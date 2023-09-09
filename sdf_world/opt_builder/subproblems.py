import numpy as np
from typing import *
from .planning_instances import *
from .constraints import *


# Subproblems
class SubProblem:
    robots: List[str] = []
    home_configs: Dict[str,Config] = {}
    num_viapoints: int = 0
    def __init__(self):
        self.idx = None
        self.name = None
        #problem
        self.instances: List[Instance] = []
        self.constraints: List[Constraint] = []
    def set_constraints(self):
        pass
    def set_instances(self):
        pass
    @classmethod
    def get_other_robot(cls, robot):
        return [x for x in cls.robots if x!= robot][0]
    @property
    def is_not_single_arm(self):
        return len(self.robots) > 1 
    

# Keyframe
class Keyframe(SubProblem):
    def __init__(self, idx):
        #scene
        super().__init__()
        self.idx = idx
        self.obj_pose: Pose = None
        self.configs: Dict[str, Config] = {} #one config for each robot
        
class Home(Keyframe):
    def __init__(self, idx, obj_pose:Pose, configs:Dict[str,Config]):
        super().__init__(idx=idx)
        assert obj_pose.is_var == False
        assert all([config.is_var==False for config in configs.values()])
        self.obj_pose = obj_pose
        self.configs = configs
    def set_instances(self, is_relaxed=False):
        self.instances.append(self.obj_pose)
        if not is_relaxed:
            self.instances.extend(list(self.configs.values()))
    def set_constraints(self):
        pass # no constraint

class PickPlace(Keyframe):
    def __init__(self, idx, robot, grasp, obj_pose):
        super().__init__(idx=idx)
        self.robot = robot
        self.grasp = grasp
        self.obj_pose = obj_pose
        self.prefix = "pickplace"
    def set_instances(self):
        name = f"{self.prefix}_{self.robot}_{self.idx}"
        self.config = Config(name, self.robot, is_var=True)
        self.configs[self.robot] = self.config
        self.instances.append(self.config)
        if self.is_not_single_arm:
            other_robot = self.get_other_robot(self.robot)
            other_config_name = f"{self.prefix}_{self.robot}_{self.idx}"
            self.config_other = Config(
                other_config_name, 
                other_robot, 
                is_var=False,  # if pick, the other robot is going home(constant)
                value=self.home_configs[other_robot])
            self.configs[other_robot] = self.config_other
            self.instances.append(self.config_other)
    def set_constraints(self):
        constr_name = f"{self.prefix}_{self.robot}_{self.idx}"
        self.constraints.append(
            KinematicsError(
                constr_name, self.grasp, self.obj_pose, self.config))
        self.constraints.append(
            RobotColDistance(constr_name, self.config))
        
class RelaxedPickPlace(Keyframe):
    def __init__(self, idx, robot, grasp, obj_pose):
        super().__init__(idx=idx)
        self.robot = robot
        self.grasp = grasp
        self.obj_pose = obj_pose
        self.prefix = "rpickplace"
    def set_instances(self):
        pass
    def set_constraints(self):
        constr_name = f"{self.prefix}_{self.robot}_{self.idx}"
        self.constraints.append(
            InvManipulability(constr_name, self.grasp, self.obj_pose))
        self.constraints.append(
            HandColDistance(constr_name, self.grasp, self.obj_pose))

class Handover(Keyframe):
    def __init__(self, idx, robot_from, robot_to, grasp_from, grasp_to):
        super().__init__(idx=idx)
        self.robot_from = robot_from
        self.robot_to = robot_to
        self.grasp_from = grasp_from
        self.grasp_to = grasp_to
        self.obj_pose = None
        self.prefix = "handover"
    def set_instances(self):
        config_robot_from_name = f"{self.prefix}_{self.robot_from}_{self.idx}"
        config_robot_to_name = f"{self.prefix}_{self.robot_to}_{self.idx}"
        pose_obj_name = f"{self.prefix}_{self.idx}"
        self.config_from = Config(config_robot_from_name, 
                                  self.robot_from, is_var=True)
        self.config_to = Config(config_robot_to_name, 
                                self.robot_to, is_var=True)
        self.obj_pose = Pose(pose_obj_name, is_var=True)
        self.configs[self.robot_from] = self.config_from
        self.configs[self.robot_to] = self.config_to
        self.instances.extend([self.config_from, self.obj_pose, self.config_to])
    def set_constraints(self):
        config_robot_from_name = f"{self.prefix}_{self.robot_from}_{self.idx}"
        config_robot_to_name = f"{self.prefix}_{self.robot_to}_{self.idx}"
        self.constraints.extend([
            KinematicsError(config_robot_from_name, self.grasp_from, self.obj_pose, self.config_from),
            KinematicsError(config_robot_to_name, self.grasp_to, self.obj_pose, self.config_to),])
        


# Action
class Action(SubProblem):
    def __init__(self, name:str):
        super().__init__()
        self.name = name
        self.instances = []
        self.idx = None
    def set_instances(self):
        self.configs: Dict[str, List[Config]] = {r:[] for r in self.robots}

    def set_waypoints(self, prev_keyframe:Keyframe, next_keyframe:Keyframe):
        for i in range(self.num_viapoints):
            for r in self.robots:
                if not prev_keyframe.configs[r].is_var and \
                    not next_keyframe.configs[r].is_var:
                    continue
                action_name = self.__class__.__name__
                name = f"{action_name}_{self.idx}_{r}_via{i}"
                waypoint = Config(name, r, is_var=True)
                self.configs[r].append(waypoint)
                self.instances.append(waypoint) #default:waypoint instances
    def set_constraints(self):
        waypoint_constrs = []
        for i in range(self.num_viapoints):
            for r in self.robots:
                waypoint = self.configs[r][i]
                action_name = self.__class__.__name__
                name = f"{action_name}_{self.idx}_{r}_via{i}"
                waypoint_constrs.append(RobotColDistance(name, waypoint))
                #for waypoint in self.configs[r]:
        self.constraints.extend(waypoint_constrs)
        
class MoveFree(Action):
    def __init__(self, move_robot):
        super().__init__(name="movefree")
        self.robot = move_robot #move robot
    def set_instances(self):
        super().set_instances() #add waypoints
    def set_constraints(self):
        super().set_constraints() #add waypoint constr
    
class MoveHold(Action):
    def __init__(self, hold_robot):
        super().__init__(name="movehold")
        self.robot = hold_robot
    def set_instances(self):
        super().set_instances()
        self.grasp = Grasp(f"{self.idx}", self.robot, is_var=True)
        self.instances.extend([self.grasp])
    def set_constraints(self):
        name = f"movehold_{self.idx}_{self.robot}"
        self.constraints.append(GraspProb(name, self.grasp))
        super().set_constraints()