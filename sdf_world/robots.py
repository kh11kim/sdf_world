from pathlib import Path
import numpy as np
from sdf_world.util import Array
import trimesh

import jax
from jax import Array
import jax.numpy as jnp
from jaxlie import SE3, SO3

import meshcat
import meshcat.geometry as g

from typing import *

from .sdf_world import *
from .robot_model import *
from .util import *

PREDEFINED_ROBOTS = [
    "gen3+hand_e",
    "panda+panda_hand"
]

GEN3_PACKAGE = Path(__file__).parent / "assets/robots/kinova_description/gen3_7dof"
GEN3_URDF = GEN3_PACKAGE / "urdf/GEN3_URDF_V12.urdf"
HANDE_PACKAGE = Path(__file__).parent / "assets/robots/kinova_description/hande_gripper"
HANDE_URDF = HANDE_PACKAGE / "urdf/robotiq_hande.urdf"

PANDA_PACKAGE = Path(__file__).parent / "assets/robots/franka_description/panda"
PANDA_URDF = PANDA_PACKAGE / "urdf/franka_panda.urdf"
PANDAHAND_PACKAGE = Path(__file__).parent / "assets/robots/franka_description/panda_hand"
PANDAHAND_URDF = PANDAHAND_PACKAGE / "urdf/hand.urdf"

def apply_base_pose(wxyzxyz:Array, base_pose:SE3):
    return (base_pose @ SE3(wxyzxyz)).as_matrix() 
apply_base_pose_batch = jax.jit(jax.vmap(apply_base_pose, (0,None)))

class Arm:
    def __init__(
        self, 
        parent, name, model:RobotModel, 
        base_pose=SE3.identity(), color="white", alpha=0.5
    ):
        self.parent = parent
        self.model = model
        self.color = Colors.read(color)
        self.alpha = alpha

        self.handle = parent[name]
        self.meshes = {
            link.name:Mesh(self.handle, link.name, link.visual_mesh_path, alpha=self.alpha) 
            if link.has_mesh else None
            for link in self.model.links.values()}
        self.base_pose = base_pose
        #state
        self.q = self.model.neutral
        self.fks_mat = None 
        self.set_joint_angles(self.q)

    def set_base_pose(self, pose:SE3):
        self.base_pose = pose
        list(self.meshes.values())[0].set_pose(self.base_pose)
        self.set_joint_angles(self.q)

    def set_joint_angles(self, q):
        self.q = q
        fks = self.model.fk_links(q)
        self.set_link_poses(fks)
    
    def set_link_poses(self, fks:Array):
        self.fks_mat = apply_base_pose_batch(fks, self.base_pose)
        for i, mesh in enumerate(self.meshes.values()):
            if i == 0: continue
            if mesh is not None:
                mesh.set_pose(np.asarray(self.fks_mat[i-1], dtype=float))


class Gripper:
    def __init__(
        self,
        parent, name, model:RobotModel,
        tool_pose_offset,
        max_width,
        is_rev_type=True,
        visualize="visual",
        color="yellow", 
        alpha=0.5,
        scale = 1.
    ):
        self.parent = parent
        self.model = model
        self.max_width = max_width
        self.is_rev_type=is_rev_type # if q is max, grasping
        self.tool_pose_offset = tool_pose_offset
        self.color = Colors.read(color)
        self.alpha = alpha
        self.handle = parent[name]
        self.fk_fn = self.model.get_general_links_fk_fn()
        self.visualize = visualize
        self.meshes = {
            link.name:Mesh(self.handle, link.name, link.visual_mesh_path, scale=scale, alpha=alpha) 
            if link.has_mesh else None
            for link in self.model.links.values()}
        self.q = self.model.lb
        self.grip()
        self.hand_pc = self.get_hand_pc()
    
    
    
    # def reload(self, vis_type="visual"):
    #     for name, mesh in self.meshes.items():
    #         if vis_type == "visual":
    #             path = self.model.links[name].visual_mesh_path
    #         else:
    #             path = self.model.links[name].collision_mesh_path
    #         mesh.reload(path=path)

    def get_bounding_box(self, name):
        hand_pc_wrt_tool_pose = self.get_hand_pc_wrt_tool_pose(20)
        min_points = hand_pc_wrt_tool_pose.min(axis=0)
        max_points = hand_pc_wrt_tool_pose.max(axis=0)
        extents = max_points - min_points
        center = (max_points + min_points) / 2
        box = Box(self.parent, name, extents, visualize=False)
        box.set_translate(center)
        return box

    def get_hand_pc(self, num_points=10):
        """wrt base pose (not tool pose)"""
        pcs = {}
        for name, mesh in self.meshes.items():
            pcs[name] = farthest_point_sampling(mesh.mesh.sample(10*num_points), num_points) 
        
        hand_pc = []
        fks = jnp.vstack([SE3.identity().parameters(), self.fk_fn(self.q)])
        for i, (name, pc) in enumerate(pcs.items()):
            link_pose = SE3(fks[i])
            hand_pc.extend(jax.vmap(link_pose.apply)(pc))
        return np.vstack(hand_pc)
    
    def get_hand_pc_wrt_tool_pose(self, num_points=10):
        hand_pc = self.get_hand_pc(num_points)
        return jax.vmap(
            self.tool_pose_offset.inverse().apply)(hand_pc)
    
    def set_tool_pose(self, pose):
        base_pose = pose @ self.tool_pose_offset.inverse()
        self.handle.set_transform(np.array(base_pose.as_matrix(), dtype=float))

    def grip(self, width=None, base_pose=SE3.identity()):
        if width is None:
            width = self.max_width
        assert width <= self.max_width
        if self.is_rev_type:
            width = self.max_width - width
        self.q = np.full(2, width/2)
        
        fks = self.fk_fn(self.q)
        fks_mat = apply_base_pose_batch(fks, base_pose)

        for i, mesh in enumerate(self.meshes.values()):
            if i == 0: continue
            if mesh is not None:
                mesh.set_pose(np.array(fks_mat[i-1], dtype=float))
        list(self.meshes.values())[0].set_pose(base_pose)

class ArmGripper:
    def __init__(self, arm:Arm, gripper:Gripper):
        self.arm = arm
        self.gripper = gripper
        self.fk_fn = self.arm.model.get_serial_links_fk_fn(gripper.tool_pose_offset)
        self.fk_fn = jax.jit(self.fk_fn)
        ee_pose = SE3(self.arm.model.fk_links(self.arm.q)[-1])
        self.gripper.grip(base_pose=ee_pose)
        self.pcs = []
        self.hand_pc = self.gripper.get_hand_pc_wrt_tool_pose(10)
        for i, mesh in enumerate(self.arm.meshes.values()):
            if i == 0: continue
            if mesh is None:
                self.pcs.append(None)
            else:
                self.pcs.append(farthest_point_sampling(mesh.mesh.sample(100), 10))
        self.pcs.append(self.hand_pc)
        
    def set_joint_angles(self, q):
        fks = self.fk_fn(q)
        self.arm.set_link_poses(fks)
        self.gripper.grip(base_pose=SE3.from_matrix(self.arm.fks_mat[-2]))
        self.arm.q = q

    def set_base_pose(self, pose:SE3):
        self.arm.set_base_pose(pose)
        self.set_joint_angles(self.arm.q)
    
    def get_fk_ee_fn(self):
        return self.arm.model.get_fk_ee_fn(self.gripper.tool_pose_offset)

    def get_robot_pc_fn(self, robot_pose:SE3=SE3.identity()):
        def get_robot_pc(q):
            fks = self.fk_fn(q)
            # fks = jnp.vstack([SE3.identity().parameters(), fks]) #include base
            
            assigned_pcs = []
            for i in range(len(fks)):
                if self.pcs[i] is None: continue
                link_pose = robot_pose @ SE3(fks[i])
                assigned_pc = jax.vmap(link_pose.apply)(self.pcs[i])
                assigned_pcs.extend(assigned_pc)
            assigned_pcs = jnp.vstack(assigned_pcs)
            return assigned_pcs
        return get_robot_pc



def get_predefined_robot(parent, robot_name):
    handle = parent[robot_name]
    if robot_name == "gen3+hand_e":
        gen3_model = RobotModel(GEN3_URDF, GEN3_PACKAGE)
        hande_model = RobotModel(HANDE_URDF, HANDE_PACKAGE, is_floating=True)
        kinova = Arm(handle, "gen3", gen3_model, alpha=0.5)
        
        tool_pose_offset= SE3.from_rotation_and_translation(
            SO3.from_z_radians(jnp.pi/2), jnp.array([0,0,0.135]))
        hande = Gripper(handle, "hande", hande_model, tool_pose_offset,
                        max_width=0.05, scale=0.001)
        robot = ArmGripper(kinova, hande)
    
    elif robot_name == "panda+panda_hand":
        panda_model = RobotModel(PANDA_URDF, PANDA_PACKAGE)
        pandahand_model = RobotModel(PANDAHAND_URDF, PANDAHAND_PACKAGE)
        tool_pose_offset = SE3.from_translation(jnp.array([0,0,0.105]))
        panda = Arm(handle, "panda", panda_model)
        panda_hand = Gripper(handle, "panda_hand", pandahand_model, 
                             tool_pose_offset, max_width=0.08, is_rev_type=False)
        robot = ArmGripper(panda, panda_hand)
    return robot