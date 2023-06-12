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

from .sdf_world import MeshCatObject
from .util import *

PANDA_PACKAGE = Path(__file__).parent / "assets/robots/panda"
PANDA_URDF = PANDA_PACKAGE / "franka_panda.urdf"
HAND_URDF = PANDA_PACKAGE / "hand.urdf"

class Link:
    def __init__(self, name, xyz=None, rpy=None, visual_mesh=None, collision_mesh=None, package=None):
        self.name = name
        self.xyz = str2arr(xyz)
        self.rpy = str2arr(rpy)
        self.parent: Joint = None
        self.child: Joint = None
        self.visual_mesh_path = visual_mesh
        self.collision_mesh_path = collision_mesh
        
        if self.has_mesh:
            self.visual_mesh_path = self.get_mesh_path(visual_mesh, package)
            self.collision_mesh_path = self.get_mesh_path(collision_mesh, package)
            self.visual_mesh = self.load_mesh(self.visual_mesh_path)
            self.collision_mesh = self.load_mesh(self.collision_mesh_path)
            self.surface_points = farthest_point_sampling(self.collision_mesh.sample(100), 20)

    @property
    def has_mesh(self):
        return (self.visual_mesh_path is not None) or (self.collision_mesh_path is not None)
    
    @property
    def has_mesh_offset(self):
        return (self.xyz is not None) or (self.rpy is not None)

    def load_mesh(self, path):
        mesh = trimesh.load(path)
        if isinstance(mesh, trimesh.Scene):
            # print(f"merging mesh scene: {path}")
            mesh = mesh.dump(True)
        return mesh
    
    def get_meshcat_obj(self, type="visual"):
        if type == "visual":
            mesh = self.visual_mesh
        else:
            raise NotImplementedError("visual mesh only")
        exp_obj = trimesh.exchange.obj.export_obj(mesh)
        return g.ObjMeshGeometry.from_stream(
            trimesh.util.wrap_as_stream(exp_obj))
    
    def get_T_link_offset(self):
        xyz = jnp.array(self.xyz)
        rot = SO3.from_rpy_radians(*self.rpy)
        return SE3.from_rotation_and_translation(rot, xyz)
    
    def get_mesh_path(self, path:str, package:Union[str,Path]):
        if isinstance(package, str):
            return path.replace("package:/", package)
        elif isinstance(package, Path):
            return path.replace("package:/", package.as_posix())
    
    def __repr__(self):
        return f"Link:{self.name}"

class Joint:
    def __init__(self, name, joint_type, xyz, rpy, parent, child, 
                 axis=None, lb=None, ub=None):
        self.name = name
        self.joint_type = joint_type
        self.xyz = str2arr(xyz)
        self.rpy = str2arr(rpy)
        self.parent: Link = parent
        self.child: Link = child
        self.axis = str2arr(axis)
        self.lb = None if lb is None else float(lb)
        self.ub = None if lb is None else float(ub)
    
    @property
    def actuated(self):
        return self.joint_type != "fixed"
    
    def get_T_offset(self):
        xyz = jnp.array(self.xyz)
        rot = SO3.from_rpy_radians(*self.rpy)
        return SE3.from_rotation_and_translation(rot, xyz)

    def get_T_joint(self, q):
        if self.joint_type == "revolute":
            #assume that revolute joint axis is 0 0 1
            return SE3.from_rotation(SO3.from_rpy_radians(0,0,q))
        elif self.joint_type == "prismatic":
            return SE3.from_translation(jnp.array(self.axis * q))

    def __repr__(self):
        return f"Joint:{self.name}"
    
class RobotModel:
    def __init__(self, urdf, package, is_floating=False):
        self.urdf = urdf
        self.packge = package
        self.links: Dict[str,Link] = {}
        self.joints: Dict[str,Joint] = {}
        self.fk_fn = None #jitted FK
        #self.fk = None #compiled FK
        self.dof = None
        self.view_col_mesh = False #not implemented
        self.view_visual_mesh = True
        self.q = None
        self.lb = None
        self.ub = None
        self.neutral = None
        self.is_floating = is_floating # This excludes the point assignment of root link

        self.parse_urdf()
        self.build()

    def parse_urdf(self):
        import xml.etree.ElementTree as ET
        tree = ET.parse(self.urdf)
        root = tree.getroot()

        #parse links
        for link in root.iter("link"):
            xyz = rpy = None
            name = link.attrib["name"]
            visual_mesh = collision_mesh = None
            if link.find("visual") is not None:
                visual_mesh = link.find("visual").find("geometry").find("mesh").attrib["filename"]
                if link.find("visual").find("origin") is not None:
                    xyz = link.find("visual").find("origin").attrib["xyz"]
                    rpy = link.find("visual").find("origin").attrib["rpy"]
            if link.find("collision") is not None:
                collision_mesh = link.find("collision").find("geometry").find("mesh").attrib["filename"]
            self.links[name] = Link(
                name, xyz, rpy, visual_mesh, collision_mesh, self.packge)
        last_link_name = name

        #parse joints
        self.joints = {}
        for joint in root.iter("joint"):
            name = joint.attrib["name"]
            joint_type = joint.attrib["type"]
            xyz = joint.find("origin").attrib["xyz"]
            rpy = joint.find("origin").attrib["rpy"]
            parent_name = joint.find("parent").attrib["link"]
            parent = self.links[parent_name]
            child_name = joint.find("child").attrib["link"]
            child = self.links[child_name]
            axis = lb = ub = None
            if joint_type != "fixed":
                axis = joint.find("axis").attrib["xyz"]
                lb = joint.find("limit").attrib["lower"]
                ub = joint.find("limit").attrib["upper"]
            self.joints[name] = Joint(name, joint_type, xyz, rpy, parent, child, axis, lb, ub)
            parent.child = self.joints[name]
            child.parent = self.joints[name]
        
        #set parameters
        self.root_link = self.get_root_link(self.links[last_link_name])
        self.dof = 0
        for joint in self.joints.values():
            if joint.joint_type != "fixed":
                self.dof += 1
        self.lb = np.array([joint.lb for joint in self.joints.values() if joint.actuated])
        self.ub = np.array([joint.ub for joint in self.joints.values() if joint.actuated])
        self.neutral = (self.lb + self.ub) / 2
        self.q = self.neutral
        
    def build(self):
        def fk(q):
            Tlink = {}
            Tlink[self.root_link.name] = SE3.identity()
            dof_idx = 0
            for link in self.links.values():
                if link.name in Tlink: continue
                parent_joint = link.parent
                parent_link = parent_joint.parent
                T = Tlink[parent_link.name] @ parent_joint.get_T_offset()
                if parent_joint.joint_type != "fixed":
                    theta = q[dof_idx]
                    T = T @ parent_joint.get_T_joint(theta)
                    dof_idx += 1
                if link.has_mesh_offset:
                    T = T @ link.get_T_link_offset()
                Tlink[link.name] = T
            return jnp.stack([T.parameters() for T in Tlink.values()], axis=0)
        self.fk_fn = jax.jit(fk)

        def ee_jac(q):
            fks = self.fk_fn(q)
            pos_jac = []
            rot_jac = []
            p_ee = fks[-1][-3:]
            for i, joint in enumerate(self.joints.values()):
                if joint.joint_type != "revolute": continue
                joint_idx = i + 1 #exclude root
                p_frame = fks[joint_idx][-3:]
                rot_axis = SE3(fks[joint_idx]).as_matrix()[:3, 2]
                lin_vel = jnp.cross(rot_axis, p_ee - p_frame)
                pos_jac.append(lin_vel)
                rot_jac.append(rot_axis)
            pos_jac = jnp.vstack(pos_jac).T
            rot_jac = jnp.vstack(rot_jac).T
            return jnp.vstack([pos_jac, rot_jac])
        self.jac_fn = jax.jit(ee_jac)

        def get_robot_surface_points(q):
            to_mat = lambda wxyzxyz: SE3(wxyzxyz).as_matrix()
            points = []
            fks = self.fk_fn(q) #vector wxyz_xyz
            for wxyzxyz, link in zip(fks, self.links.values()):
                if not link.has_mesh: continue
                if not self.is_floating and link == self.root_link: continue
                assigned_surface_points = jax.vmap(SE3(wxyzxyz).apply)(
                    link.surface_points
                )
                points.append(assigned_surface_points)
            return jnp.vstack(points)
        self.get_surface_points_fn = jax.jit(get_robot_surface_points)
        # fk_mat = lambda q: jax.vmap(to_mat)(fk(q))
        # self.fk = jax.jit(self.fk_mat_fn).lower(jnp.zeros(self.dof)).compile()
        # self.get_surface_points = \
        #     jax.jit(self.get_surface_points_fn).lower(jnp.zeros(self.dof)).compile()


    def get_root_link(self, link:Link):
        while True:
            if link.parent is None:
                break
            link = link.parent
        return link

class Robot(MeshCatObject):
    def __init__(
            self, vis, name, model:RobotModel, color="white", alpha=1., show_surface_points=False):
        self.model = model
        self.fk_to_mat = jax.jit(jax.vmap(lambda wxyzxyz: SE3(wxyzxyz).as_matrix()))
        self.full_idx = np.arange(self.model.dof)
        self.free_idx = self.full_idx.copy()
        self.fix_idx = []
        self.fix_val = []
        self.lb = model.lb
        self.ub = model.ub
        self.neutral = model.neutral
        self.show_surface_points = show_surface_points
        self.color = Colors.read(color)
        self.alpha = alpha
        self.q = self.neutral
        super().__init__(vis, name)
        self.set_joint_angles(self.q)

    def load(self):
        material = g.MeshLambertMaterial(color=self.color, opacity=self.alpha)
        for link in self.model.links.values():
            # for now, only supports visual mesh
            if link.has_mesh:
                obj = link.get_meshcat_obj("visual")
                self.handle["visual"][link.name].set_object(obj, material)

    def show(self):
        Ts = self.fk_to_mat(self.model.fk_fn(self.q))
        Ts = np.asarray(Ts, dtype=np.float64)
        i = 0
        for i, link in enumerate(self.model.links.values()):
            if not link.has_mesh: continue
            self.handle["visual"][link.name].set_transform(Ts[i])
            i += 1
        self.handle["surface_points"].delete()
        if self.show_surface_points:
            surface_points = self.model.get_surface_points(self.q).T
            colors = np.tile(
                Colors.read("red",return_rgb=True), 
                surface_points.shape[1]).reshape(-1, 3).T
            self.handle["surface_points"].set_object(
                g.PointCloud(
                    position=surface_points, color=colors, size=0.03)
            )
            
    def set_joint_angles(self, q:Array, show=True):
        assert len(q) == len(self.free_idx)
        if len(self.fix_idx) != 0:
            qnew = np.zeros(self.model.dof)
            np.put(qnew, self.free_idx, q)
            np.put(qnew, self.fix_idx, self.fix_val)
            self.q = qnew
        else:
            self.q = np.asarray(q)
        if show:
            self.show()
    
    def get_surface_points_fn(self, q):
        qnew = jnp.zeros(self.model.dof)
        qnew = qnew.at[self.free_idx].set(q)
        if len(self.fix_idx) != 0:
            qnew = qnew.at[self.fix_idx].set(self.fix_val)
        return self.model.get_surface_points_fn(qnew)
    
    def reduce_dim(self, fix_idx=[], fix_val=[]):
        assert len(fix_idx) == len(fix_val)
        self.free_idx = np.array([i for i in self.full_idx if i not in fix_idx])
        self.fix_idx = np.array(fix_idx)
        self.fix_val = np.array(fix_val)
        self.lb = self.model.lb[self.free_idx]
        self.ub = self.model.ub[self.free_idx]
        self.neutral = self.model.neutral[self.free_idx]

    def get_random_config(self):
        lb = self.model.lb[self.free_idx]
        ub = self.model.ub[self.free_idx]
        return np.random.uniform(lb, ub)
