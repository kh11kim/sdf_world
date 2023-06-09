import numpy as np
import trimesh
import meshcat
import meshcat.geometry as g
import matplotlib
from typing import *

from jax import Array
import jax.numpy as jnp
from jaxlie import SE3, SO3
from .util import *
from .sdf import *


class SDFWorld:
    def __init__(self):
        self.vis = meshcat.Visualizer()
        bg = self.vis["/Background"]
        bg.set_property("bottom_color", [0.1, 0.1, 0.1])
        bg.set_property("top_color", [0.1, 0.1, 0.1])
        self.bodies = {}

    def show_in_jupyter(self, height=400):
        return self.vis.jupyter_cell(height)
    
    def create(self, obj_class, **kwargs):
        obj = obj_class(self.vis, **kwargs)
        self.bodies[obj.name] = obj
        return obj
    
    def delete(self, body:"MeshCatObject"):
        # if isinstance(body, MeshCatObject):
        #     body = body.name
        if body.name in self.bodies:
            obj = self.bodies.pop(body.name)
            del obj
            del body
        else:
            print(f"No body named {body}")


class MeshCatObject:
    def __init__(self, vis, name):
        self.vis = vis
        self.name = name
        self.handle = self.vis[name]
        self.handle.delete()
        self.pose: SE3 = SE3.identity()
    
    def load(self):
        raise NotImplementedError()
    
    def __del__(self):
        self.handle.delete()

    def set_pose(self, pose:SE3):
        self.pose = pose
        self.handle.set_transform(to_numpy_tfmat(pose))
    
    def set_translate(self, xyz):
        assert len(xyz) == 3
        xyz = np.array(xyz)
        T = np.block([[np.eye(3),   xyz[:,None]],
                      [np.zeros(3), 1.         ]])
        self.pose = SE3.from_matrix(T)
        self.handle.set_transform(T)
    
    def reload(self, **configs):
        for name, value in configs.items():
            if name in dir(self):
                setattr(self, name, value)
        self.handle.delete()
        self.load()
        self.set_pose(self.pose)

class Box(MeshCatObject, SDFBox):
    def __init__(self, vis, name, lengths, color=None, alpha=1.):
        assert len(lengths) == 3
        super().__init__(vis=vis, name=name)
        self.lengths = np.array(lengths)
        self.color = Colors.read(color)
        self.alpha = alpha
        self.load()

    def load(self):
        obj = g.Box(self.lengths)
        material = g.MeshLambertMaterial(color=self.color, opacity=self.alpha)
        self.handle.set_object(obj, material)
    
    def reshape(self, lengths):
        self.lengths = np.array(lengths)
        self.handle.delete()
        self.load()
    
    def penetration(self, point, safe_dist):
        d = self.distance(point, self.pose, self.lengths/2)
        return task_space_potential(d, safe_dist)

class Sphere(MeshCatObject, SDFSphere):
    def __init__(self, vis, name, r, color=None, alpha=1.):
        super().__init__(vis=vis, name=name)
        self.r = r
        self.color = Colors.read(color)
        self.alpha = alpha
        self.load()

    def load(self):
        obj = g.Sphere(self.r)
        material = g.MeshLambertMaterial(color=self.color, opacity=self.alpha)
        self.handle.set_object(obj, material)
    
    def reshape(self, **config):
        for name, value in config.items():
            if name in dir(self):
                setattr(self, name, value)
        self.handle.delete()
        self.load()
        self.set_pose(self.pose)
    
    def penetration(self, point, safe_dist):
        center = self.pose.translation()
        d = self.distance(point, center, self.r)
        return task_space_potential(d, safe_dist)

class Mesh(MeshCatObject):
    def __init__(self, vis, name, path):
        super().__init__(vis=vis, name=name)
        self.path = path
        self.mesh = trimesh.load(path)
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = self.mesh.dump(True)
        self.load()

    def load(self):
        exp_obj = trimesh.exchange.obj.export_obj(self.mesh)
        obj = g.ObjMeshGeometry.from_stream(
            trimesh.util.wrap_as_stream(exp_obj))
        self.vis[self.name].set_object(obj)


class Frame(MeshCatObject):
    def __init__(self, vis, name, length=0.1, r=0.02):
        super().__init__(vis=vis, name=name)
        self.length = length
        self.r = r
        self.load()
    
    def load(self):
        axes = ["x", "y", "z"]
        colors = [0xff0000, 0x00ff00, 0x0000ff]
        length_mat = np.full((3,3), self.r)
        np.fill_diagonal(length_mat, self.length)
        tf_mat = np.diag([self.length/2]*3)
        for i in range(3):
            color = colors[i]
            obj = g.Box(length_mat[i])
            material = g.MeshLambertMaterial(color=color)
            self.handle[axes[i]].set_object(obj, material)
            self.handle[axes[i]].set_transform(mat_from_translate(tf_mat[i]))

class PointCloud(MeshCatObject):
    def __init__(self, vis, name, points, size=0.05, color="white"):
        """points(n, 3)"""
        super().__init__(vis=vis, name=name)
        length = points.shape[0]
        self.points = np.array(points)
        self.color = np.tile(Colors.read(color, return_rgb=True), length).reshape(-1, 3)
        self.size = size
        self.load()

    def load(self):
        obj = g.PointsGeometry(self.points.T, self.color.T)
        material = g.PointsMaterial(size=self.size)
        self.handle["pc"].set_object(obj, material)
        
        