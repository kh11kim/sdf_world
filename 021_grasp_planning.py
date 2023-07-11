import numpy as np
import jax
import jax.numpy as jnp
from jaxlie import SE3, SO3
import jax_dataclasses as jdc
from functools import partial

from sdf_world.sdf_world import *
from sdf_world.robots import *
from sdf_world.util import *
from flax.training import orbax_utils
import orbax
import pickle

world = SDFWorld()
panda_model = RobotModel(PANDA_URDF, PANDA_PACKAGE)
panda = Robot(world.vis, "panda", panda_model, alpha=0.5)
panda.reduce_dim([7, 8], [0.04, 0.04])

#load object
from flax import linen as nn


#grasp net
class GraspNet(nn.Module):
    hidden_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        logit = nn.Dense(features=5)(x)
        return logit

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
raw_restored = orbax_checkpointer.restore("model/grasp_net")
params = raw_restored["params"]
grasp_net = GraspNet(raw_restored["hidden_dim"])
grasp_fn = lambda x: grasp_net.apply(params, x)

with open("./sdf_world/assets/object"+'/info.pkl', 'rb') as f:
    obj_data = pickle.load(f)
scale_to_norm = obj_data["scale_to_norm"]
def grasp_reconst(g:Array):
    rot = SO3(grasp_fn(g)[1:5]).normalize()
    trans = g/scale_to_norm
    return SE3.from_rotation_and_translation(rot, trans)
grasp_logit_fn = lambda g: grasp_fn(g)[0]

# object
obj_start = Mesh(world.vis, "obj_start", "./sdf_world/assets/object/mesh.obj",
                 alpha=0.5)
d, w, h = obj_start.mesh.bounding_box.primitive.extents
obj_start.set_translate([0.4, -0.3, h/2])
frame = Frame(world.vis, "grasp_pose")

# CGN fns
class ResidualFn:
    def get_error_and_jac(self, x, state):
        pass
    def get_weight(self):
        pass

class ConstraintFn:
    def get_value_and_jac(self, x, state):
        pass
    def get_bounds(self):
        pass

@jax.jit
def residual_eval(x, state, res_fns:Tuple[ResidualFn]):
    errors, jacs = [], []
    for res_fn in res_fns:
        error, jac = res_fn.get_error_and_jac(x, state)
        errors.append(error)
        jacs.append(jac)
    return jnp.hstack(errors), jnp.vstack(jacs)

def residual_weights(res_fns:Tuple[ResidualFn]):
    weights = []
    for res_fn in res_fns:
        weights.append(res_fn.get_weight())
    return jnp.hstack(weights)

@jax.jit
def constr_eval(x, state, constr_fns:Tuple[ConstraintFn]):
    vals, jacs = [], []
    for constr_fn in constr_fns:
        val, jac = constr_fn.get_value_and_jac(x, state)
        vals.append(val)
        jacs.append(jac)
    return jnp.hstack(vals), jnp.vstack(jacs)

def constr_bounds(constr_fns:Tuple[ConstraintFn]):
    lbs, ubs = [], []
    for constr_fn in constr_fns:
        lb, ub = constr_fn.get_bounds()
        lbs.append(lb)
        ubs.append(ub)
    return jnp.hstack(lbs), jnp.hstack(ubs)

# Kinematics
def get_rotvec_angvel_map(v):
    def skew(v):
        v1, v2, v3 = v
        return jnp.array([[0, -v3, v2],
                        [v3, 0., -v1],
                        [-v2, v1, 0.]])
    vmag = jnp.linalg.norm(v)
    vskew = skew(v)
    return jnp.eye(3) \
        - 1/2*skew(v) \
        + vskew@vskew * 1/vmag**2 * (1-vmag/2 * jnp.sin(vmag)/(1-jnp.cos(vmag)))

@jax.jit
def get_ee_fk_jac(q):
    # outputs ee_posevec and analytical jacobian
    fks = panda_model.fk_fn(q)
    p_ee = fks[-1][-3:]
    rotvec_ee = SO3(fks[-1][:4]).log()
    E = get_rotvec_angvel_map(rotvec_ee)
    jac = []
    for posevec in fks[1:8]:
        p_frame = posevec[-3:]
        rot_axis = SE3(posevec).as_matrix()[:3, 2]
        lin_vel = jnp.cross(rot_axis, p_ee - p_frame)
        jac.append(jnp.hstack([lin_vel, rot_axis]))
    jac = jnp.array(jac).T
    jac = jac.at[3:, :].set(E @ jac[3:, :])
    return jnp.hstack([p_ee, rotvec_ee]), jac

robot_dim = 7
grasp_dim = 3
dim = grasp_dim + robot_dim
def to_posevec(pose:SE3):
    return jnp.hstack([pose.translation(), pose.rotation().log()])
grasp_to_posevec = lambda g: to_posevec(grasp_reconst(g))

@jdc.pytree_dataclass
class KinError(ResidualFn):
    def get_error_and_jac(self, x, state):
        grasp = x[:grasp_dim]
        grasp_pose = grasp_reconst(grasp)
        grasp_posevec = to_posevec(grasp_pose)
        q = x[-robot_dim:]
        ee, jac = get_ee_fk_jac(q)
        error = grasp_posevec - ee
        return error, jnp.hstack([jnp.zeros((6,3)), -jac])
    
    def get_weight(self):
        return np.array([1,1,1,0.3,0.3,0.3]) * 0.1

initial_pose = get_ee_fk_jac(panda.neutral)[0]
euc_dist_to_grasp = lambda g: to_posevec(obj_start.pose@grasp_reconst(g))[:3]
@jdc.pytree_dataclass
class GraspDistance(ResidualFn):
    def get_error_and_jac(self, x, state):
        grasp = x[:grasp_dim]
        grasp_pose = grasp_reconst(grasp)
        grasp_pose_wrt_world = obj_start.pose@grasp_pose
        grasp_posevec = to_posevec(grasp_pose_wrt_world)
        error = grasp_posevec[:3]
        jac = 1 / scale_to_norm * jnp.eye(3)
        return error, jnp.hstack([jac, jnp.zeros((3,7))])
    
    def get_weight(self):
        return np.array([1,1,1.])


@jdc.pytree_dataclass
class JointLimit(ConstraintFn):
    """ lb < val < ub. dval/dx = jac"""
    robot_lb: Array
    robot_ub: Array
    def get_value_and_jac(self, x, state):
        val = x
        jac = jnp.eye(dim)
        return val, jac
    
    def get_bounds(self):
        lb = jnp.tile(self.robot_lb, 1)
        ub = jnp.tile(self.robot_ub, 1)
        return lb, ub
    
@jdc.pytree_dataclass
class State:
    q0: Array
    target: Array
state = State(jnp.array(panda.neutral), jnp.zeros(3))
res_fns = [KinError(), GraspDistance()] #, 
constr_fns = [JointLimit(panda.lb, panda.ub)] #, Penetration()

x = jnp.hstack([0,0,0,panda.neutral])
panda.set_joint_angles(x[-7:])

residual_eval(x, state, res_fns)
print("heay")