import numpy as np
from typing import *

# Planning instances
class Instance:
    def __init__(self, name, dim, is_var, value=None):
        assert (not is_var and value is not None) or is_var
        self.name = name
        self.dim = dim
        self.is_var = is_var
        self.value = value
        self.coord: np.ndarray = None
        self.lb: np.ndarray = None
        self.ub: np.ndarray = None
    def __repr__(self):
        prefix = "Var:" if self.is_var else "Param:"
        return prefix+self.name
    
class Grasp(Instance):
    def __init__(self, name, robot, is_var, value=None):
        super().__init__(
            name="grasp_"+name,
            dim=3,
            is_var=is_var,
            value=value)
        self.lb = -np.ones(3)
        self.ub = np.ones(3)
        self.robot = robot

class Config(Instance):
    def __init__(self, name, robot, is_var, value=None):
        super().__init__(
            name="config_"+name,
            dim=7,
            is_var=is_var,
            value=value)
        self.lb = -np.ones(7)*np.pi
        self.ub = np.ones(7)*np.pi
        self.robot = robot

class Pose(Instance):
    def __init__(self, name, is_var, value=None):
        super().__init__(
            name="pose_"+name,
            dim=7,
            is_var=is_var,
            value=value)
        self.lb = np.array([-1,-1,-0.5, -1, -1, -1, -1]) #ws
        self.ub = np.array([1,1,1.5, 1, 1, 1, 1])
        
class PosePos(Instance):
    def __init__(self, name, is_var, value=None):
        super().__init__(
            name="pose_"+name,
            dim=3,
            is_var=is_var,
            value=value)
        
        