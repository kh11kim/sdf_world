import time
from pybullet_suite import *
import trimesh
import pandas as pd
import tqdm
import open3d as o3d

class GraspSim:
    def __init__(self, hand:Gripper, obj:Body, obj_mesh:trimesh.Trimesh):
        self.world = hand.world
        self.hand = hand
        self.obj = obj
        self.mesh = obj_mesh

    def execute2(self, visualize=False, surface_points=100000, grasp_samples=100000):
        print("Approach grasp sampling...")
        points, faces = trimesh.sample.sample_surface_even(self.mesh, surface_points)
        normals = self.mesh.face_normals[faces]
        grasps = []
        for _ in tqdm.tqdm(range(grasp_samples)):
            i = np.random.randint(0, len(points))
            point, normal = points[i], normals[i]
            self.obj.set_base_pose(Pose.identity())
            self.hand.remove()
            depth, grasp_pose_cand = self.generate_approach_grasp(point, normal)
            if grasp_pose_cand is None:
                continue
            self.hand.reset(grasp_pose_cand)
            result = self.simulate_grasp(grasp_pose_cand, view=visualize) # view=true if visualization
            if result is None:
                continue
            grasps.append((point, normal, depth, *result))

        columns = ["x", "y", "z", "a1", "a2", "a3", "g1", "g2", "g3", "depth", "width"]
        graspable_points = []
        for point, normal, depth, (width, grasp_pose) in grasps:
            approach_vec = - normal
            grasp_vec = grasp_pose.rot.as_matrix()[:,1]
            graspable_points.append((*point, *approach_vec, *grasp_vec, depth, width))
        np.hstack([points, ])
        df_succ = grasps

    def execute(self, visualize=False, surface_points=100000, grasp_samples=100000):
        print("Antipodal grasp sampling...")
        points, faces = trimesh.sample.sample_surface_even(self.mesh, surface_points)
        normals = self.mesh.face_normals[faces]
        grasps = []
        for _ in tqdm.tqdm(range(grasp_samples)):
            i = np.random.randint(0, len(points))
            point, normal = points[i], normals[i]
            self.obj.set_base_pose(Pose.identity())
            self.hand.remove()
            grasp_pose_cand = self.generate_antipodal_grasp(point, normal)
            if grasp_pose_cand is None:
                continue
            self.hand.reset(grasp_pose_cand)
            result = self.simulate_grasp(grasp_pose_cand, view=visualize) # view=true if visualization
            if result is None:
                continue
            grasps.append((i, *result))
        
        print("Convert to approach grasp...")
        df_succ = self.convert_to_approach_grasp(grasps)
        df_fail = self.get_grasp_failure_data(df_succ, points)
        print("-----Result-----")
        print(f"succ:{len(df_succ)}, fail:{len(df_fail)}")
        return df_succ, df_fail
    
    def convert_to_approach_grasp(self, antipodal_grasps):
        columns = ["i", "x", "y", "z", "a1", "a2", "a3", "g1", "g2", "g3", "depth", "width"]
        graspable_points = []
        self.hand.remove()
        self.obj.set_base_pose(Pose.identity())
        for i, (idx, width, grasp) in tqdm.tqdm(enumerate(antipodal_grasps)):
            tcp = grasp.trans
            grasp_vec = grasp.rot.as_matrix()[:,1] #yaxis
            approach_vec = grasp.rot.as_matrix()[:,-1]

            ray_origin = tcp - approach_vec
            ray_direction = approach_vec
            locations, index_ray, index_tri = self.mesh.ray.intersects_location(
                ray_origins=ray_origin[None, :],
                ray_directions=ray_direction[None,:])
            if len(locations) != 0:
                idx = np.linalg.norm(locations - ray_origin, axis=1).argmin()
                surface_point = locations[idx] #np.array(result[0][3])
                depth = np.linalg.norm(surface_point - tcp)
                graspable_points.append((idx, *surface_point, *approach_vec, *grasp_vec, depth, width))
        graspable_points = np.array(graspable_points)

        # make dataframe
        df_succ = pd.DataFrame(graspable_points, columns=columns)
        return df_succ

    def get_grasp_failure_data(self, df_succ, surface_points):
        # generate failure data with KDTree
        graspable_points = df_succ.loc[:,["x", "y", "z"]].to_numpy(dtype=float)
        graspable_points_o3d = o3d.utility.Vector3dVector(graspable_points)
        pc = o3d.geometry.PointCloud(graspable_points_o3d)
        pc_tree = o3d.geometry.KDTreeFlann(pc)
        failure_points = []
        for i, point in enumerate(surface_points):
            _, idx, _ = pc_tree.search_knn_vector_3d(point.reshape(3,1), 1)
            d = np.linalg.norm(graspable_points[idx] - point)
            if d >= 0.002:
                failure_points.append([*point])
        df_failure = pd.DataFrame(failure_points, columns=["x", "y", "z"]) #, "nx", "ny", "nz"
        return df_failure

    def apply_random_cone_rotation(self, vector, max_angle=np.pi/6):
        rand_rotvec = Rotation.random().as_rotvec()
        random_rot = Rotation.from_rotvec(rand_rotvec / (np.pi*2) * max_angle)
        return Pose(random_rot).transform_vector(vector/np.linalg.norm(vector))

    def get_antipodal_point(self, point, penetration_vector):
        ray_from = point + penetration_vector
        result = self.world.physics_client.rayTest(ray_from, point)
        success = result[0][0] != -1
        if success:
            return result[0][3]
        return None

    def get_grasp_orn_w_pitch(self, y, pitch):
        x_ = np.array([1,0,0])
        if np.linalg.norm(x_ - y) < 0.001:
            x_ = np.array([0,0,1])
        z = np.cross(x_, y)
        x = np.cross(y, z)
        rotmat = np.vstack([x,y,z]).T
        rot = Rotation.from_matrix(rotmat) * Rotation.from_euler("zyx", [0,pitch,0])
        return rot

    def get_grasp_orn_w_yaw(self, z, yaw):
        x_ = np.array([1,0,0])
        if np.linalg.norm(x_ - z) < 0.001:
            x_ = np.array([0,1,0])
        y = np.cross(z, x_)
        x = np.cross(y, z)
        rotmat = np.vstack([x,y,z]).T
        rot = Rotation.from_matrix(rotmat) * Rotation.from_euler("zyx", [yaw,0,0])
        return rot

    def generate_antipodal_grasp(self, point, normal):
        # point, normal = grasp["point"], grasp["normal"]
        pitch_grid = np.linspace(0, np.pi*2, 10, endpoint=False)

        grasp_vec = self.apply_random_cone_rotation(-normal) #normalized
        point2 = self.get_antipodal_point(point, grasp_vec)
        if point2 is None:
            return None #fail
        w = np.linalg.norm(point - point2)
        if w >= 0.08:
            return None  #failed on the sampled point
        tcp = (point + point2)/2
    
        np.random.shuffle(pitch_grid)
        for pitch in pitch_grid:
            rot = self.get_grasp_orn_w_pitch(grasp_vec, pitch)
            self.hand.reset(Pose(rot, tcp))
            if self.hand.is_grasp_candidate(self.obj):
                return Pose(rot, tcp)
        return None

    def generate_approach_grasp(self, point, normal):
        depth = np.random.uniform(0.01, 0.04)
        yaw = np.random.uniform(0, np.pi*2)
        rot = self.get_grasp_orn_w_yaw(-normal, yaw)
        grasp_pose = Pose(rot, point) * Pose(trans=[0,0,depth])
        self.hand.reset(grasp_pose)
        if self.hand.is_grasp_candidate(self.obj):
            return depth, grasp_pose
        return None


    def simulate_grasp(self, grasp_pose, view=True):
        #simulate
        self.obj.set_base_pose(Pose.identity())
        self.hand.reset(grasp_pose)
        for _ in range(100):
            self.hand.grip(0, control=True)
            self.world.step()
            if view:
                time.sleep(0.005)
        
        #check
        if not self.hand.detect_contact():
            return None
        width = self.hand.body.get_joint_angles().sum()
        self.hand.grip()
        self.hand.reset(self.hand.get_tcp_pose())
        if not self.hand.is_grasp_candidate(self.obj):
            return None
        final_grasp_pose = self.obj.get_base_pose().inverse() * self.hand.get_tcp_pose()
        return width, final_grasp_pose

    # def get_dataframe(self, grasps, scale):
    #     col_names=[
    #         "px", "py", "pz", 
    #         "rot11", "rot21", "rot31", 
    #         "rot12", "rot22", "rot32",
    #         "rot13", "rot23", "rot33",
    #         "width", "obj_scale"
    #     ]
    #     rot_col_names = ["rot11", "rot21", "rot31", 
    #         "rot12", "rot22", "rot32",
    #         "rot13", "rot23", "rot33"]
    #     df = pd.DataFrame(columns=col_names)
    #     if len(grasps) == 0:
    #         return df
    #     for i, grasp in enumerate(grasps):
    #         #df.loc[i, "label"] = grasp["label"]
    #         df.loc[i, ["px", "py", "pz"]] = grasp["point"]
    #         if "pose" in grasp:
    #             df.loc[i, rot_col_names] = grasp["pose"].rot.as_matrix().T.reshape(-1)
    #         if "width" in grasp:
    #             df.loc[i, "width"] = grasp["width"]
    #     df["scale"] = scale
    #     return df
    
    # def get_fail_dataframe(self, failures, scale):
    #     df = pd.DataFrame(failures, columns=["px", "py", "pz"])
    #     df["scale"] = scale
    #     return df