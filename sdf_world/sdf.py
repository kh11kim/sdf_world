import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
#import jax_dataclasses as jdc
from jaxlie import SE3, SO3
from typing import *

def safe_2norm(d):
    """differentiable 2-norm"""
    is_zero = jnp.allclose(d, 0.)
    d = jnp.where(is_zero, jnp.ones_like(d), d)
    l = jnp.linalg.norm(d)
    l = jnp.where(is_zero, 0., l)
    l = jnp.max(jnp.array([l, 1e-6]))
    return l

def task_space_potential(d, safe_dist):
    def free(d):
        return 0.
    def in_safe_dist(d):
        return 1/(2*safe_dist)*(d-safe_dist)**2
    def in_col(d):
        return -d + 1/2 * safe_dist
    is_in_col = d <= 0.
    is_in_safe_dist = (0. < d) & (d < safe_dist)
    switch_var = is_in_col + is_in_safe_dist*2
    return jax.lax.switch(switch_var, [free, in_col, in_safe_dist], d)

#@jdc.pytree_dataclass
class SDF:
    def _distance(self, point):
        raise NotImplementedError
    def penetration(self, point, safe_dist):
        raise NotImplementedError
    
#@jdc.pytree_dataclass
class SDFSphere(SDF):
    def _distance(self, point, center, r):
        return safe_2norm(point - center) - r

#@jdc.pytree_dataclass
class SDFBox(SDF):
    def _distance(self, point, box_pose:SE3, half_extents:Array):
        point = box_pose.inverse().apply(point)
        q = jnp.abs(point) - half_extents
        return safe_2norm(jnp.maximum(q, 0)) + \
            jnp.minimum(jnp.maximum(q[0], jnp.maximum(q[1], q[2])), 0)


#@jdc.pytree_dataclass
class SDFContainer:
    def __init__(self, sdfs:List[SDF], safe_dist:float):
        self.sdfs = sdfs
        self.safe_dist = safe_dist
    
    def distances(self, points):
        signed_distances = []
        for sdf in self.sdfs:
            signed_distances.append(jax.vmap(sdf.sdf)(points))
        return jnp.vstack(signed_distances).min(axis=0)
    
    # def distances(self, points):
    #     signed_distances = []
    #     for sdf in self.sdfs:

    def penetration(self, point):
        max_penet = 0.
        for sdf in self.sdfs:
            max_penet = jnp.maximum(max_penet, sdf.penetration(point, self.safe_dist))
        return max_penet
    
    def penetration_sum(self, points):
        return jax.vmap(self.penetration)(points).sum()

