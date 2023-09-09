import numpy as np
from typing import *
from .planning_instances import *

LB_INV_MANIPULABILITY = 0.1
LB_GRASP_LOGIT_THRESHOLD = 0
LB_SAFE_DISTANCE = 0.05
UB_UNBOUNDED = np.inf

# constraints
class Constraint:
    def __init__(self, name, constr_type, dim, inputs:List, lb, ub):
        self.name = name
        self.constr_type = constr_type
        self.dim = dim
        self.inputs: List[Instance] = inputs
        self.coord: np.ndarray = None
        
        if isinstance(lb, (int, float)) and dim != 1:
            self.lb = np.full(dim, lb)
        else: self.lb = lb
        if isinstance(ub, (int, float)) and dim != 1:
            self.ub = np.full(dim, ub)
        else: self.ub = ub
        
    def __repr__(self):
        prefix = "Constr:"
        return f"{prefix}_{self.constr_type}_{self.name}"
    @property
    def num_inputs(self):
        return len(self.inputs)
    
class KinematicsError(Constraint):
    def __init__(self, name, grasp, obj_pose, config):
        assert isinstance(grasp, Grasp)
        assert isinstance(obj_pose, Pose)
        assert isinstance(config, Config)
        super().__init__(
            name, "kinematics_error", 6, [grasp, obj_pose, config], lb=0, ub=0)

class InvManipulability(Constraint):
    def __init__(self, name, grasp, obj_pose):
        assert isinstance(grasp, Grasp)
        assert isinstance(obj_pose, Pose)
        super().__init__(
            name, "inv_manipulability", 1, [grasp, obj_pose], 
            LB_INV_MANIPULABILITY, UB_UNBOUNDED)
        
class GraspProb(Constraint):
    def __init__(self, name, grasp):
        assert isinstance(grasp, Grasp)
        super().__init__(
            name, "grasp_prob", 1, [grasp], 
            LB_GRASP_LOGIT_THRESHOLD, UB_UNBOUNDED)
        
class RobotColDistance(Constraint):
    def __init__(self, name, config):
        assert isinstance(config, Config)
        super().__init__(
            name, "robot_col_distance", 1, [config], 
            LB_SAFE_DISTANCE, UB_UNBOUNDED)

class HandColDistance(Constraint):
    def __init__(self, name, grasp, obj_pose):
        assert isinstance(grasp, Grasp)
        assert isinstance(obj_pose, Pose)
        super().__init__(
            name, "hand_col_distance", 1, [grasp, obj_pose], 
            LB_SAFE_DISTANCE, UB_UNBOUNDED)

# class Constant(Constraint):
#     def __init__(self, param:Instance):
#         super().__init__(
#             param.name, "constant", param.dim, [param], param.value, param.value
#         )