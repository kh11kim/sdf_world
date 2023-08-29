import jax_dataclasses as jdc
from jaxlie import SE3, SO3
from typing import *
import jax
from jax import Array
import jax.numpy as jnp
from .util import value_and_jacfwd
from functools import partial

@jdc.pytree_dataclass
class IKConfig:
    fk_ee: Callable # 
    robot_base_pose: SE3
    threshold: float
    damping: float
    max_iter: int

zflip_SO3 = SO3.from_z_radians(jnp.pi)
def fk_err(q:Array, target_pose:SE3, robot_pose:SE3, fk_ee_fn:Callable):
    sqrsum = lambda x: jnp.sum(x**2)
    target_pose_wrt_base = robot_pose.inverse() @ target_pose
    ee_pose = SE3(fk_ee_fn(q))
    pos_err = target_pose_wrt_base.translation() - ee_pose.translation()
    rot_err1 = ee_pose.rotation().inverse() @ target_pose_wrt_base.rotation()
    rot_err2 = rot_err1 @ zflip_SO3
    rot_errors = jnp.vstack([rot_err1.log(), rot_err2.log()])    
    rot_err = rot_errors[jax.vmap(sqrsum)(rot_errors).argmin()]
    return jnp.hstack([pos_err, rot_err])

# ik_cond
def ik_body(carry):
    q, target_pose, mag_error, i, config = carry
    vg_fk_err = jax.tree_util.Partial(
        func=value_and_jacfwd(fk_err, argnums=0), fk_ee_fn=config.fk_ee)
    error, jac_err = vg_fk_err(q, target_pose, config.robot_base_pose)
    jac = -jac_err
    hess = (jac.T @ jac + jnp.eye(7) * config.damping)
    d = jnp.linalg.solve(hess, jac.T@error)
    q = q + d
    mag_error = jnp.linalg.norm(error)
    i += 1
    return q, target_pose, mag_error, i, config

def ik_cond(carry):
    q, target_pose, mag_error, i, config = carry
    return (mag_error > config.threshold) & (i <= config.max_iter)

def get_ik_fn():
    ik = jax.tree_util.Partial(jax.lax.while_loop, cond_fun=ik_cond, body_fun=ik_body)
    return ik

