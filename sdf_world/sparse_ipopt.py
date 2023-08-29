import numpy as np
import jax.numpy as jnp
import jax
from jax import Array
import cyipopt
from dataclasses import dataclass, field
from typing import *

@dataclass
class Variable:
    name: str
    coord: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    
    @property
    def dim(self): return len(self.coord)

@dataclass
class Parameter:
    name: str
    coord: np.ndarray
    dim: int
    value: np.ndarray

    @property
    def lb(self): return np.full(self.dim, -np.inf)
    @property
    def ub(self): return np.full(self.dim, np.inf)

@dataclass
class Constraint:
    name: str
    coord: np.ndarray
    inputs: List[Variable]
    fn: "Function"
    lb: np.ndarray
    ub: np.ndarray
    no_deriv_names: List[str]
    jac_indices: np.ndarray

    @property
    def dim(self): return len(self.coord)

@dataclass
class Function:
    name: str
    in_dims: List[int]
    out_dim: int
    eval_fn: Callable
    jac_fn: Callable
    jac_out_argnums: Iterable
    custom_jac_indices: Optional[List[np.ndarray]]
    constraints: List[Constraint] = field(default_factory=list)

class SparseIPOPT():
    def __init__(self, check_fn=False):
        self.x_info: Dict[str, Union[Variable, Parameter]] = {}
        self.c_info: Dict[str,Constraint] = {}
        self.fn_info: Dict[str,Function] = {}
        self.obj_info: Dict = {}
        self.param_info: Dict[str, Array] = {}
        self.check_fn = check_fn

        self.x_idx, self.c_idx = 0, 0
        self.param_info: Dict[str, np.ndarray] = {}

    @property
    def xdim(self):
        return sum([x.dim for x in self.x_info.values()])
    @property
    def cdim(self):
        return sum([c.dim for c in self.c_info.values()])
    @property
    def input_info(self):
        return {**self.x_info, **self.param_info}
    
    def print_x_info(self):
        print({var.name: var.dim for var in self.x_info.values()})
    
    def split_solution(self, sol):
        return {var.name:sol[var.coord] for var in self.x_info.values()}

    def add_variable(self, name, dim, lb=-np.inf, ub=np.inf):
        assert name not in self.x_info
        assert isinstance(lb, float) or len(lb) == dim
        assert isinstance(ub, float) or len(ub) == dim

        if isinstance(lb, float): lb = np.full(dim, lb)
        if isinstance(ub, float): ub = np.full(dim, ub)

        coord = np.arange(self.x_idx, self.x_idx+dim)
        self.x_info[name] = Variable(
            name, coord, lb, ub)
        
        self.x_idx += dim
    
    def add_parameter(self, name, dim, value=None):
        assert name not in self.param_info
        #TODO: name also should not be in "variable"
        if value is None:
            value = np.zeros(dim)
        coord = np.arange(self.x_idx, self.x_idx+dim)
        self.x_info[name] = Parameter(name, coord, dim, value)
        self.x_idx += dim
    
    def change_parameter(self, name, value):
        assert self.x_info[name].dim == len(value)
        self.x_info[name].value = value
    
    def get_init_value(self, x_init_dict:Dict):
        for var in self.x_info.values():
            if var.name in x_init_dict:
                continue
            elif isinstance(var, Parameter):
                x_init_dict[var.name] = var.value
            elif var.name not in x_init_dict:
                raise ValueError(f"{var.name} : Not in x_init_dict")
        return np.hstack([x_init_dict[varname] for varname in self.x_info])

    def set_objective(self, fn_name, input_x_names):
        self.obj_info["fn"] = self.fn_info[fn_name]
        self.obj_info["inputs"] = [self.x_info[name] for name in input_x_names]
    
    def set_debug_callback(self, debug_callback:Callable):
        self.obj_info["debug_cb"] = debug_callback

    def set_constr(self, name, cfn_name, input_x_names, lb, ub, no_deriv_names=[]):
        c_fn = self.fn_info[cfn_name]
        cdim = c_fn.out_dim
        assert name not in self.c_info
        assert isinstance(lb, float) or len(lb) == cdim
        assert isinstance(ub, float) or len(ub) == cdim
        if isinstance(lb, float): lb = np.full(cdim, lb)
        if isinstance(ub, float): ub = np.full(cdim, ub)

        vars = [self.x_info[name] for name in input_x_names]
        c_coord = np.arange(self.c_idx, self.c_idx+cdim)

        jac_indices = []
        for i, var in enumerate(vars):
            if var.name in no_deriv_names: continue
            if isinstance(var, Parameter): continue
            
            if c_fn.custom_jac_indices is not None:
                row, col = c_fn.custom_jac_indices[i]
            else:
                row, col = np.indices((cdim, var.dim)).reshape(2, -1)
            row_offset, col_offset = c_coord[0], var.coord[0] # offset
            jac_indices.append(np.vstack([row+row_offset, col+col_offset]))

        self.c_info[name] = Constraint(
            name, c_coord, vars, 
            c_fn, lb, ub,
            no_deriv_names, jac_indices)
        c_fn.constraints.append(self.c_info[name])
        self.c_idx += cdim

    def register_fn(
            self, 
            name, in_dims, out_dim, 
            eval_fn, jac_fn, 
            jac_out_argnums=None,
            custom_jac_indices=None):
        xdummies = [jnp.zeros(dim) for dim in in_dims]
        if self.check_fn:
            assert eval_fn(*xdummies).size == out_dim
            assert len(jac_fn(*xdummies)) == len(in_dims)
        
        if jac_out_argnums is None:
            jac_out_argnums = np.arange(len(in_dims))
        
        if custom_jac_indices is not None:
            assert len(custom_jac_indices) == len(jac_out_argnums)
        self.fn_info[name] = Function(
            name, in_dims, out_dim, eval_fn, jac_fn, 
            jac_out_argnums, custom_jac_indices)
    
    def get_objective_fn(self, compile=True):
        no_obj = False
        if "fn" not in self.obj_info: 
            objective = lambda x: 0.
            no_obj = True
        else:
            def objective(x):        
                xs = {var.name:x[var.coord] for var in self.x_info.values()}
                fn_input = [xs[var.name] for var in self.obj_info["inputs"]]    
                val = self.obj_info["fn"].eval_fn(*fn_input)
                return val
        
        if "debug_cb" in self.obj_info:
            def objective_debug(x):
                xs = {var.name:x[var.coord] for var in self.x_info.values()}
                self.obj_info["debug_cb"](xs)    
                return objective(x)
            return objective_debug
        elif compile and not no_obj:
            return jax.jit(objective)
        return objective
    
    def get_gradient_fn(self, compile=True):
        no_obj = False
        if "fn" not in self.obj_info: 
            gradient = lambda x: np.zeros(self.xdim)
            no_obj = True
        else:
            grad_value_dict = {var.name: np.zeros(var.dim) for var in self.x_info.values()}
            def gradient(x):
                xs = {var.name:x[var.coord] for var in self.x_info.values()}
                fn_input = [xs[var.name] for var in self.obj_info["inputs"]]    
                grads = self.obj_info["fn"].jac_fn(*fn_input)
                for var, grad in zip(self.obj_info['inputs'], grads):
                    grad_value_dict[var.name] = grad
                return jnp.hstack(grad_value_dict.values())
        if compile and not no_obj:
            return jax.jit(gradient)
        return gradient      
    
    def get_constraint_fn(self, compile=True):
        def constraints(x):
            xs = {var.name:x[var.coord] for var in self.x_info.values()}
            result = []
            for constr in self.c_info.values():
                fn_input = [xs[var.name] for var in constr.inputs]    
                out = constr.fn.eval_fn(*fn_input)
                result.append(out)
            return jnp.hstack(result)
        if compile:
            return jax.jit(constraints)
        return constraints
    
    def get_jacobian_fn(self, compile=True):
        def jacobian(x):
            xs = {var.name:x[var.coord] for var in self.x_info.values()}
            result = []
            for constr in self.c_info.values():
                fn_input = [xs[var.name] for var in constr.inputs]    
                jacs = constr.fn.jac_fn(*fn_input)
                
                for i in constr.fn.jac_out_argnums:
                    var = constr.inputs[i]
                    #for i, var in enumerate(constr.inputs):
                    if var.name in constr.no_deriv_names: continue
                    elif isinstance(var, Parameter): continue
                    result.append(jacs[i].flatten())
            return jnp.hstack(result)
        if compile:
            return jax.jit(jacobian)
        return jacobian

    
    def get_jacobian_structure(self):
        rows, cols = [], []
        for constr in self.c_info.values():
            for jac_idx in constr.jac_indices:
                if jac_idx is None: continue
                rows.append(jac_idx[0])
                cols.append(jac_idx[1])
        rows = np.hstack(rows)
        cols = np.hstack(cols)
        return rows, cols

    def print_sparsity(self):
        row, col = self.get_jacobian_structure()
        jac_struct = np.full((self.cdim, self.xdim), -1, dtype=int)
        jac_struct[row, col] = 1
        for row in jac_struct:
            row_str = ""
            for val in row:
                if val == -1: row_str += "-"
                else: row_str += f"o"
            print(row_str)
    
    def build(self, compile=True):
        lb = np.hstack([x.lb for x in self.x_info.values()])
        ub = np.hstack([x.ub for x in self.x_info.values()])
        fns = {
            "objective": self.get_objective_fn(compile),
            "gradient": self.get_gradient_fn(compile)}
        if self.cdim != 0:
            cl = np.hstack([c.lb for c in self.c_info.values()])
            cu = np.hstack([c.ub for c in self.c_info.values()])
            row, col = self.get_jacobian_structure()
            jac_struct_fn = lambda : (row, col)
            fns["constraints"] = self.get_constraint_fn(compile)
            fns["jacobian"] = self.get_jacobian_fn(compile)

        class Prob:
            pass
        prob = Prob()
        xdummy = jnp.zeros(self.xdim)
        for fn_name, fn in fns.items():
            print(f"compiling {fn_name} ...")
            fn(xdummy)
            setattr(prob, fn_name, fn)
        
        
        if self.cdim != 0:
            setattr(prob, "jacobianstructure", jac_struct_fn)
            ipopt = cyipopt.Problem(
                n=self.xdim, m=self.cdim,
                problem_obj=prob,
                lb=lb, ub=ub, cl=cl, cu=cu
            )
            self.print_sparsity()
        else:
            ipopt = cyipopt.Problem(
                n=self.xdim, m=self.cdim,
                problem_obj=prob,
                lb=lb, ub=ub)
            print("no constraints")
            
        # default option
        ipopt.add_option("acceptable_iter", 2)
        ipopt.add_option("acceptable_tol", np.inf) #release
        ipopt.add_option("acceptable_obj_change_tol", 0.1)
        ipopt.add_option("acceptable_constr_viol_tol", 1.)
        #ipopt.add_option("acceptable_dual_inf_tol", 1.) 
        ipopt.add_option('mu_strategy', 'adaptive')
        ipopt.add_option("print_level", 1)
        
        return ipopt