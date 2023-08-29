from pathlib import Path
import numpy as np
import trimesh

import jax
from jax import Array
import jax.numpy as jnp
from jaxlie import SE3, SO3


from typing import *
from .util import *

class Link:
    def __init__(self, name:str, package:Path, xyz=None, rpy=None, visual_mesh=None, collision_mesh=None):
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
            #self.set_surface_points(20)

    @property
    def has_mesh(self):
        return self.visual_mesh_path is not None
    
    @property
    def has_mesh_offset(self):
        return (self.xyz is not None) or (self.rpy is not None)
    
    def get_mesh_path(self, path:str, package:Path):
        if "package:/" in path:
            return path.replace("package:/", package.as_posix())
        elif "../" in path:
            return path.replace("..", package.as_posix())

    def get_T_mesh_offset(self):
        xyz = jnp.array(self.xyz)
        rot = SO3.from_rpy_radians(*self.rpy)
        return SE3.from_rotation_and_translation(rot, xyz)
    
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
        self.ub = None if ub is None else float(ub)
    
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
    def __init__(self, urdf:Path, package:Path, is_floating=False):
        self.urdf = urdf
        self.package = package
        self.links: Dict[str,Link] = {}
        self.joints: Dict[str,Joint] = {}
        self.dof = None
        self.lb = None
        self.ub = None
        self.neutral = None
        self.is_floating = is_floating # This excludes the point assignment of root link
        self.parse_urdf()
        self.fk_links = self.get_serial_links_fk_fn()

    def get_joint_type_idx(self, joint_type_str):
        if joint_type_str == "revolute":
            return 0
        elif joint_type_str == "prismatic":
            return 1
        elif joint_type_str == "fixed":
            return 2
        elif joint_type_str == "continuous":
            return 3
        raise ValueError()
    
    def parse_urdf(self):
        def _parse_links(root):
            # parse links from urdf
            links_info = []
            for i, link in enumerate(root.iter("link")):    
                name = link.attrib["name"]
                visual_mesh = collision_mesh = None
                xyz = rpy = None
                if link.find("visual") is not None:
                    visual_mesh = link.find("visual").find("geometry").find("mesh").attrib["filename"]
                    if link.find("visual").find("origin") is not None:
                        # mesh offset
                        xyz = link.find("visual").find("origin").attrib["xyz"]
                        rpy = link.find("visual").find("origin").attrib["rpy"]
                if link.find("collision") is not None:
                    if link.find("collision").find("geometry").find("mesh") is not None:
                        collision_mesh = link.find("collision").find("geometry").find("mesh").attrib["filename"]
                    else:
                        collision_mesh = "primitive" #not implemented
                link_info = dict(
                    name=name,
                    xyz=xyz,
                    rpy=rpy,
                    visual_mesh=visual_mesh,
                    collision_mesh=collision_mesh
                )
                links_info.append(link_info)
            return links_info
        def _parse_joints(root):
            #parse joints
            joints_info = []
            for i, joint in enumerate(root.iter("joint")):
                name = joint.attrib["name"]
                joint_type = joint.attrib["type"]
                xyz = joint.find("origin").attrib["xyz"]
                rpy = joint.find("origin").attrib["rpy"]
                parent_name = joint.find("parent").attrib["link"]
                child_name = joint.find("child").attrib["link"]
                
                axis = lb = ub = None
                if joint_type == "fixed":
                    axis = "0 0 0"
                    lb = ub = 0.
                else: 
                    axis = joint.find("axis").attrib["xyz"]
                    if joint_type == "continuous":
                        lb = -np.pi
                        ub = np.pi
                    else:
                        limit = joint.find("limit")
                        if limit is not None:
                            if "lower" in limit.attrib:
                                lb = limit.attrib["lower"]
                            if "upper" in limit.attrib:
                                ub = limit.attrib["upper"]    
                joint_info = dict(
                    name=name, joint_type=joint_type, xyz=xyz, rpy=rpy,
                    parent_name=parent_name, child_name=child_name,
                    axis=axis, lb=lb, ub=ub
                )
                joints_info.append(joint_info)
            return joints_info
    
        #load xml
        import xml.etree.ElementTree as ET
        tree = ET.parse(self.urdf)
        root = tree.getroot()

        links_info = _parse_links(root)
        joints_info = _parse_joints(root)

        #load links
        for link_info in links_info:
            link = Link(
                link_info["name"],
                self.package,
                link_info["xyz"],
                link_info["rpy"],
                link_info["visual_mesh"], 
                link_info["collision_mesh"])
            self.links[link.name] = link
        
        #load joints
        for joint_info in joints_info:
            joint = Joint(
                joint_info["name"], 
                joint_info["joint_type"], 
                joint_info["xyz"], 
                joint_info["rpy"], 
                self.links[joint_info["parent_name"]],
                self.links[joint_info["child_name"]], 
                joint_info["axis"], 
                joint_info["lb"], 
                joint_info["ub"])
            self.links[joint_info["parent_name"]].child = joint
            self.links[joint_info["child_name"]].parent = joint
            #joint.parent = self.links[joint_info["parent_name"]]
            self.joints[joint.name] = joint

        # get T_offsets:
        self.T_offsets = []
        self.joint_axes, self.joint_types = [], []
        for joint in self.joints.values():
            rot = SO3.from_rpy_radians(*joint.rpy)
            xyz = joint.xyz
            self.T_offsets.append(
                SE3.from_rotation_and_translation(rot, xyz).parameters())
            self.joint_axes.append(joint.axis)
            self.joint_types.append(self.get_joint_type_idx(joint.joint_type))
        self.T_offsets = np.array(self.T_offsets)
        self.joint_axes = np.array(self.joint_axes)
        self.joint_types = np.array(self.joint_types)
        self.dim = len(self.joints)
        self.movable_joints = np.arange(self.dim)[self.joint_types != 2]
        self.lb = np.array([joint.lb for joint in self.joints.values()])[self.movable_joints]
        self.ub = np.array([joint.ub for joint in self.joints.values()])[self.movable_joints]
        self.neutral = (self.lb + self.ub)/2
        
    def get_serial_links_fk_fn(self, tool_pose_offset=None):
        def qmap(q):
            result = jnp.zeros(self.dim)
            return result.at[self.movable_joints].set(q)
        def get_joint_transform(theta, axis, joint_type_idx):
            def revolute(theta, axis): # cond 0
                return SE3.from_rotation(SO3.from_rpy_radians(*(axis*theta)))
            def prismatic(theta, axis): # cond 1
                return SE3.from_translation((axis*theta))
            def fixed(theta, axis): #cond 2
                return SE3.identity()
            return jax.lax.switch(joint_type_idx, [revolute, prismatic, fixed, revolute], theta, axis).parameters()
        def calc_frame(prev_frame, inputs):
            T_offset, T_joint = inputs
            frame = SE3(prev_frame) @ SE3(T_offset) @ SE3(T_joint)
            return frame.parameters(), frame.parameters()
        
        def fk_links(q):
            joints = qmap(q)
            T_joints = jax.vmap(get_joint_transform, (0,0,0))(
                joints, self.joint_axes, self.joint_types)
            inputs = jnp.stack([self.T_offsets, T_joints], axis=1)
            base_frame = SE3.identity().parameters()
            _, fks = jax.lax.scan(calc_frame, base_frame, inputs)
            if tool_pose_offset is not None:
                ee = (SE3(fks[-1]) @ tool_pose_offset).parameters()
                return jnp.vstack([fks, ee])
            else:
                return fks
        return fk_links

    def get_general_links_fk_fn(self):
        def fk_links(q):
            Tlink = {}
            theta_idx = 0
            for i, link in enumerate(self.links.values()):
                if i == 0:
                    Tlink[link.name] = SE3.identity() # assume the first one is base
                    continue
                parent_joint = link.parent
                parent_link = parent_joint.parent
                T = Tlink[parent_link.name] @ parent_joint.get_T_offset()
                if parent_joint.joint_type != "fixed":
                    theta = q[theta_idx]
                    T = T @ parent_joint.get_T_joint(theta)
                    theta_idx += 1
                if link.has_mesh_offset:
                    T = T @ link.get_T_mesh_offset()
                Tlink[link.name] = T
            return jnp.stack([T.parameters() for T in Tlink.values()], axis=0)[1:]
        return fk_links

    def get_fk_point_fn(self):
        @jax.custom_jvp
        def fk_point(q, link_idx=7, local_pos=jnp.zeros(3)):
            fks = self.fk_links(q)
            return SE3(fks[link_idx]).apply(local_pos)

        @fk_point.defjvp
        def fk_point_jvp(primals, tangents):
            def get_lin_vel_wrt_joint_frame(target_point, joint_frame):
                joint_to_target = target_point - joint_frame[-3:]
                rot_axis = SO3(joint_frame[:4]).as_matrix()[:3,2]
                return jnp.cross(rot_axis, joint_to_target)
            q, link_idx, local_pos = primals
            qdot, _, _ = tangents
            fks = self.fk_links(q)
            point = SE3(fks[link_idx]).apply(local_pos)
            lin_jac = jax.vmap(get_lin_vel_wrt_joint_frame, in_axes=(None,0))(
                point, fks[self.movable_joints]).T
            masking = np.tile(np.arange(7), 3).reshape(-1,7) + 1
            masking = jnp.where(masking > link_idx+1, 0, 1)
            return point, (masking * lin_jac) @ qdot
        return fk_point
    
    def get_fk_ee_fn(self, tool_pose_offset=None):
        def skew(v):
            v1, v2, v3 = v
            return jnp.array([[0, -v3, v2],
                            [v3, 0., -v1],
                            [-v2, v1, 0.]])
        
        fk_fn = self.get_serial_links_fk_fn(tool_pose_offset)

        @jax.custom_jvp
        def fk_ee(q):
            fks = fk_fn(q)
            return fks[-1]

        @fk_ee.defjvp
        def fk_ee_jvp(primals, tangents):
            q, = primals
            q_dot, = tangents
            fks = fk_fn(q)

            qtn, p_ee = fks[-1][:4], fks[-1][-3:]
            w, xyz = qtn[0], qtn[1:]
            geom_jac = []
            for posevec in fks[self.movable_joints]:
                p_frame = posevec[-3:]
                rot_axis = SE3(posevec).as_matrix()[:3, 2]
                lin_vel = jnp.cross(rot_axis, p_ee - p_frame)
                geom_jac.append(jnp.hstack([lin_vel, rot_axis]))
            geom_jac = jnp.array(geom_jac).T  #geom_jacobian
            H = jnp.hstack([-xyz[:,None], skew(xyz)+jnp.eye(3)*w])
            rot_jac = 0.5*H.T@geom_jac[3:,:]
            jac = jnp.vstack([rot_jac, geom_jac[:3,:]])
            return fks[-1], jac@q_dot
        return fk_ee
