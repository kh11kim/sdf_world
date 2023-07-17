import cyipopt
import jax
import jax.numpy as jnp
import numpy as np
from typing import *
from dataclasses import dataclass, field

@dataclass
class Variable:
    index: int
    offset: int
    name: str
    dim: int
    lb: np.ndarray
    ub: np.ndarray

@dataclass
class Parameter:
    name: str
    value: np.ndarray

@dataclass
class Objective:
    fn_info: "Function"
    inputs: List[Union[Variable,Parameter]]

@dataclass
class Constraint:
    name: str
    dim: int
    inputs: List[Union[Variable,Parameter]]
    lb: np.ndarray
    ub: np.ndarray
    fn_info: "Function"
    index: Optional[int] = field(default_factory=lambda :None)
    offset: Optional[int] = field(default_factory=lambda :None)

@dataclass
class Function:
    name: str
    in_dims: List[int]
    out_dim: int
    fn: Callable
    jacs: List[Callable]
    constraints: List[Constraint] = field(default_factory=list)


class SparseIPOPTBuilder:
    def __init__(self):
        self.x_info: Dict[str,Variable] = {}
        self.fn_info: Dict[str,Function] = {}
        self.c_info: Dict[str,Constraint] = {}
        self.obj_info: Optional[Objective] = None
        self.param_info: Dict[str, Parameter] = {}
        self.cfn_list: List[Function] = []
        self.xidx = 0
        self.x_offset = 0
    
    @property
    def input_info(self):
        return {**self.x_info, **self.param_info}
    @property
    def xdim(self):
        return sum([x.dim for x in self.x_info.values()])
    @property
    def cdim(self):
        return sum([c.dim for c in self.c_info.values()])
    @property
    def xnames(self):
        return [x.name for x in self.x_info.values()]
    @property
    def xoffsets(self):
        return [x.offset for x in self.x_info.values()]
    @property
    def xdims(self):
        return [x.dim for x in self.x_info.values()]
    @property
    def param_dict(self):
        return {param.name:param.value for param in self.param_info.values()}
    
    def as_vector(self, val, dim):
        if isinstance(val, float):
            return np.full(dim, val)
        assert len(val) == dim
        return val
    
    def add_variable(self, name, dim, lb=-np.inf, ub=np.inf):
        assert name not in self.x_info
        assert isinstance(lb, float) or len(lb) == dim
        assert isinstance(ub, float) or len(ub) == dim
        if isinstance(lb, float):
            lb = np.full(dim, lb)
        if isinstance(ub, float):
            ub = np.full(dim, ub)
        var = Variable(
            self.xidx, self.x_offset,
            name, dim, lb, ub)
        self.x_info[name] = var
        self.xidx += 1
        self.x_offset += dim
    
    def add_parameter(self, name, value):
        self.param_info[name] = Parameter(name, np.array(value))
    
    def register_fn(self, name, in_dims, out_dim, fn, jacs):
        assert len(in_dims) == len(jacs), "The number of jacobian function and input variables should be the same."
        self.fn_info[name] = Function(name, in_dims, out_dim, fn, jacs)
    
    def add_objective(self, input_x_names, fn_name):
        fn = self.fn_info[fn_name]
        assert fn.out_dim == 1
        inputs = [self.input_info[x_name] for x_name in input_x_names]
        self.obj_info = Objective(fn, inputs)

    def add_constr(self, name, input_x_names, cfn_name, lb, ub):
        assert name not in self.c_info
        cfn = self.fn_info[cfn_name]
        assert isinstance(lb, float) or len(lb) == cfn.out_dim
        assert isinstance(ub, float) or len(ub) == cfn.out_dim
        if isinstance(lb, float):
            lb = np.full(cfn.out_dim, lb)
        if isinstance(ub, float):
            ub = np.full(cfn.out_dim, ub)
        
        inputs = [self.input_info[x_name] for x_name in input_x_names]
        constr = Constraint(name, cfn.out_dim , inputs, lb, ub, cfn)
        self.c_info[name] = constr
        self.cfn_list.append(cfn)
        cfn.constraints.append(constr)
        # -> we should reindex the constraints: first clustering cfns, then constraints
    
    def get_objective_fn(self):
        ndo = (self.xnames, self.xdims, self.xoffsets)
        def objective(x):
            input_dict = {name:x[offset:offset+dim] 
                                for name, dim, offset in zip(*ndo)}
            input_dict.update(self.param_dict)
            input_names = [x.name for x in self.obj_info.inputs]
            inputs = [input_dict[name] for name in input_names]
            return self.obj_info.fn_info.fn(*inputs)
        if isinstance(self.obj_info, Objective):
            return objective
        #if obj not set, return dummy
        return lambda x: 0.
    
    def get_gradient_fn(self):
        zip_x_ndo = zip(self.xnames, self.xdims, self.xoffsets)
        grad_dict = {x.name:np.zeros(x.dim) for x in self.x_list}

        def gradient(x):
            input_dict = {name:x[offset:offset+dim] 
                                for name, dim, offset in zip_x_ndo}
            input_dict.update(self.param_dict)
            input_names = [x.name for x in self.obj_info.inputs]
            inputs = [input_dict[name] for name in input_names]
            #inputs = [x.name for x in self.obj_info.inputs]
            for i, xname in enumerate(input_names):
                grad_fn = self.obj_info.fn_info.jacs[i]
                if grad_fn is None: continue
                grad = grad_fn(*inputs)
                grad_dict[xname] = grad
            return jnp.hstack(list(grad_dict.values()))
        
        if isinstance(self.obj_info, Objective):
            return gradient
        #if obj not set, return dummy
        return lambda x: np.zeros(self.xdim)
    
    def get_constr_fn(self):
        zip_x_ndo = zip(self.xnames, self.xdims, self.xoffsets)
        # constraint evaluation
        def constraints(x):
            input_dict = {name:x[offset:offset+dim] 
                                for name, dim, offset in zip_x_ndo}
            input_dict.update(self.param_dict)

            cvals = []
            for cfn in self.cfn_list:
                input_batch = []
                num_inputs = len(cfn.in_dims)
                for i in range(num_inputs):
                    inputs = []
                    for constr in cfn.constraints:
                        input_name = constr.inputs[i].name
                        inputs.append(input_dict[input_name])
                    input_batch.append(jnp.vstack(inputs))
                cval = jax.vmap(cfn.fn)(*input_batch)
                cvals.append(cval.flatten())
            return jnp.hstack(cvals, dtype=float)
        return constraints

    def get_jacobian_fn(self):
        zip_x_ndo = zip(self.xnames, self.xdims, self.xoffsets)
        # jacobian evaluation
        def jacobian(x):
            input_dict = {name:x[offset:offset+dim] 
                                for name, dim, offset in zip_x_ndo}
            input_dict.update(self.param_dict)

            jac_vals = []
            for cfn in self.cfn_list:
                input_batch = []
                num_inputs = len(cfn.in_dims)
                #prepare input batches
                param_masks = []
                for i in range(num_inputs):
                    inputs = []
                    param_mask = []
                    for j, constr in enumerate(cfn.constraints):
                        if isinstance(constr.inputs[i], Variable):
                            param_mask.append(j)
                        input_name = constr.inputs[i].name
                        inputs.append(input_dict[input_name])
                    param_masks.append(np.array(param_mask, dtype=int))
                    input_batch.append(jnp.vstack(inputs))
                
                for i in range(num_inputs):
                    jac_fn = cfn.jacs[i]
                    if jac_fn is None: continue
                    param_mask = param_masks[i]
                    jac_val = jax.vmap(jac_fn)(*input_batch)[param_mask].flatten()
                    jac_vals.append(jac_val)
            return jnp.hstack(jac_vals, dtype=float)
        return jacobian
    
    def freeze(self):
        #reindexing var/constraints
        self.x_list = [x for x in self.x_info.values()]
        cidx = 0
        c_offset = 0
        self.c_list = []
        for cfn in self.fn_info.values():
            for constr in cfn.constraints:
                constr.index = cidx
                constr.offset = c_offset
                cidx += 1
                c_offset += constr.dim
                self.c_list.append(constr)
        
        #set jac structure
        jac_rows, jac_cols = [], []
        for cfn in self.fn_info.values():
            num_input = len(cfn.in_dims)
            for i in range(num_input):
                for constr in cfn.constraints:
                    if isinstance(constr.inputs[i], Parameter): continue
                    xdim = constr.fn_info.in_dims[i]
                    cdim = constr.dim
                    row, col = np.indices((cdim, xdim))
                    jac_rows.append(row.flatten()+constr.offset)
                    jac_cols.append(col.flatten()+constr.inputs[i].offset)
        self.jac_rows, self.jac_cols = np.hstack(jac_rows), np.hstack(jac_cols)

    def print_sparsity(self):
        jac_structure = np.full((self.cdim, self.xdim), -1, dtype=int)
        jac_structure[self.jac_rows, self.jac_cols] = np.arange(len(self.jac_rows))
        print("\nSparsity pattern:")
        for row in jac_structure:
            row_str = ""
            for val in row:
                if val >= 10:
                    row_str += f"{val} "
                elif val == -1:
                    row_str += "-- "
                else:
                    row_str += f"0{val} "
            print(row_str)
    
    def build(self, compile_obj=False):
        lb = np.hstack([x.lb for x in self.x_info.values()])
        ub = np.hstack([x.ub for x in self.x_info.values()])
        cl = np.hstack([c.lb for c in self.c_list])
        cu = np.hstack([c.ub for c in self.c_list])
        fns = {
            "objective": self.get_objective_fn(),
            "gradient": self.get_gradient_fn(),
            "constraints": self.get_constr_fn(),
            "jacobian": self.get_jacobian_fn(),
        }

        class Prob:
            pass
        prob = Prob()
        xdummy = jnp.zeros(self.xdim)
        for fn_name, fn in fns.items():
            if (not compile_obj) and (fn_name == "objective"):
                setattr(prob, fn_name, fn)
            else:
                setattr(prob, fn_name, jax.jit(fn).lower(xdummy).compile())
        jac_struct_fn = lambda : (self.jac_rows, self.jac_cols)
        setattr(prob, "jacobianstructure", jac_struct_fn)
            
        ipopt = cyipopt.Problem(
            n=self.xdim, m=self.cdim,
            problem_obj=prob,
            lb=lb, ub=ub,
            cl=cl, cu=cu
        )
        self.print_sparsity()
        return ipopt
        