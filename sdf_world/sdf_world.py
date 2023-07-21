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
        self.vis["/Background"].set_property("top_color", [0.2]*3)
        self.vis["/Background"].set_property("bottom_color", [0.2]*3)
        self.vis["/Lights/SpotLight"].set_property("visible", True)
        self.vis["/Lights/AmbientLight"].set_property("visible", False)
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
    def __init__(self, vis, name, visualize=True):
        self.vis = vis
        self.name = name
        self.handle = self.vis[name]
        self.handle.delete()
        self.pose: SE3 = SE3.identity()
        self.visualize = visualize
        if self.visualize:
            self.load()
    
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
    def __init__(self, vis, name, lengths, color="white", alpha=1., visualize=True):
        assert len(lengths) == 3
        self.lengths = np.array(lengths)
        self.color = color
        self.alpha = alpha
        super().__init__(vis=vis, name=name, visualize=visualize)

    def load(self):
        obj = g.Box(self.lengths)
        material = g.MeshLambertMaterial(color=Colors.read(self.color), opacity=self.alpha)
        self.handle.set_object(obj, material)
    
    def penetration(self, point, safe_dist):
        d = self._distance(point, self.pose, self.lengths/2)
        return task_space_potential(d, safe_dist)
    
    def sdf(self, point):
        return self._distance(point, self.pose, self.lengths/2)

class Sphere(MeshCatObject, SDFSphere):
    def __init__(self, vis, name, r, color="red", alpha=1., visualize=True):
        self.r = r
        self.color = color
        self.alpha = alpha
        super().__init__(vis=vis, name=name, visualize=visualize)

    def load(self):
        obj = g.Sphere(self.r)
        material = g.MeshLambertMaterial(color=Colors.read(self.color), opacity=self.alpha)
        self.handle.set_object(obj, material)
    
    def penetration(self, point, safe_dist):
        center = self.pose.translation()
        d = self._distance(point, center, self.r)
        return task_space_potential(d, safe_dist)
    
    def sdf(self, point):
        center = self.pose.translation()
        d = self._distance(point, center, self.r)
        return d

class Cylinder(MeshCatObject):
    def __init__(self, vis, name, h, r, color="red", alpha=1., visualize=True):
        self.h = h
        self.r = r
        self.color = color
        self.alpha = alpha
        super().__init__(vis=vis, name=name, visualize=visualize)

    def load(self):
        obj = g.Cylinder(height=self.h, radius=self.r)
        material = g.MeshLambertMaterial(color=Colors.read(self.color), opacity=self.alpha)
        self.handle.set_object(obj, material)
    

class Capsule(MeshCatObject):
    def __init__(self, vis, name, p1, p2, r, color="red", alpha=1., visualize=True):
        self.p1 = p1
        self.p2 = p2
        self.r = r
        self.color = color
        self.alpha = alpha
        super().__init__(vis=vis, name=name, visualize=visualize)

    def load(self):
        normalize = lambda v: v/jnp.linalg.norm(v)
        obj_sphere1 = g.Sphere(self.r)
        obj_sphere2 = g.Sphere(self.r)
        h = jnp.linalg.norm(self.p1 - self.p2).item()
        yaxis = normalize(self.p2 - self.p1)
        zaxis = jnp.array([0, 0.1, 1.13])  #random
        xaxis = normalize(jnp.cross(yaxis, zaxis))
        zaxis = normalize(jnp.cross(xaxis, yaxis))
        R = jnp.vstack([xaxis, yaxis, zaxis]).T
        center = (self.p1 + self.p2) / 2
        cylinder_pose = SE3.from_rotation_and_translation(SO3.from_matrix(R), center)
        T = np.array(cylinder_pose.as_matrix())
        obj_cylinder = g.Cylinder(height=h, radius=self.r)
        material = g.MeshLambertMaterial(color=Colors.read(self.color), opacity=self.alpha)
        self.handle["sphere1"].set_object(obj_sphere1, material)
        self.handle["sphere2"].set_object(obj_sphere2, material)
        self.handle["cylinder"].set_object(obj_cylinder, material)
        
        self.handle["sphere1"].set_transform(point_to_T(self.p1))
        self.handle["sphere2"].set_transform(point_to_T(self.p2))
        self.handle["cylinder"].set_transform(T.astype(float))


class Mesh(MeshCatObject):
    def __init__(self, vis, name, path, color="white", alpha=1.):
        self.path = path
        self.mesh = trimesh.load(path)
        self.color = color
        self.alpha = alpha
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = self.mesh.dump(True)
        super().__init__(vis=vis, name=name)

    def load(self):
        exp_obj = trimesh.exchange.obj.export_obj(self.mesh)
        obj = g.ObjMeshGeometry.from_stream(
            trimesh.util.wrap_as_stream(exp_obj))
        material = g.MeshLambertMaterial(
            color=Colors.read(self.color), opacity=self.alpha)
        self.vis[self.name].set_object(obj, material)


class Frame(MeshCatObject):
    def __init__(self, vis, name, length=0.1, r=0.02):
        self.length = length
        self.r = r
        super().__init__(vis=vis, name=name, visualize=True)
    
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
        # length = points.shape[0]
        self.points = np.array(points)
        self.color = color    
        self.size = size
        super().__init__(vis=vis, name=name, visualize=True)

    def get_color(self, length):
        if isinstance(self.color, str):
            color_arr = np.tile(Colors.read(self.color, return_rgb=True), length).reshape(-1, 3)
        else:
            color_arr = self.color
        return color_arr
    
    def load(self):
        color = self.get_color(self.points.shape[0])
        obj = g.PointsGeometry(self.points.T, color.T)
        material = g.PointsMaterial(size=self.size)
        self.handle["pc"].set_object(obj, material)

class DottedLine(MeshCatObject):
    def __init__(self, vis, name, points, point_size=0.01, color:str="red"):
        """points(n, 3)"""
        length = points.shape[0]
        self.points = np.array(points)
        self.color = Colors.read("red")
        self.colors = np.tile(Colors.read(color, return_rgb=True), length).reshape(-1, 3)
        self.size = point_size
        super().__init__(vis=vis, name=name, visualize=True)

    def load(self):
        point_obj = g.PointsGeometry(self.points.T, self.colors.T)
        line_material = g.MeshBasicMaterial(color=self.color)
        point_material = g.PointsMaterial(size=0.02)
        self.handle["line"].set_object(
            g.Line(point_obj, line_material)
        )
        self.handle["dot"].set_object(point_obj, point_material)