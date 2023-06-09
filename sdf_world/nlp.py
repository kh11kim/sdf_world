import time
import numpy as np
import jax
import jax.numpy as jnp
from typing import *
import cyipopt


class Variable:
    def __init__(self, name, dim, lower, upper):
        self.name = name
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.st = None
        self.ed = None
    
    @property
    def indices(self):
        return np.arange(self.st, self.ed, dtype=int)
    
    def __repr__(self):
        return f"{self.name}({self.dim})"

class Constraint:
    def __init__(self, name, dim, wrt, fn, lower, upper, jac_sparsity): #, jac
        self.name = name
        self.dim = dim
        self.wrt = wrt
        self.fn = fn
        self.lower = lower
        self.upper = upper
        self.jac_sparsity = jac_sparsity
    
    def __repr__(self):
        wrt = ",".join(self.wrt)
        return f"{self.name}({self.dim})[{wrt}]"


class NLP:  
    def __init__(self):
        self.vars = OrderedDict()
        self.constrs = OrderedDict()
        self.obj_fn = None
        self.finalized = False
        self.ipopt = None

    @property
    def dim(self):
        return sum([var.dim for var in self.vars.values()])
    @property
    def constr_dim(self):
        return sum([constr.dim for constr in self.constrs.values()])
    @property
    def lb(self):
        return np.hstack([var.lower for var in self.vars.values()])
    @property
    def ub(self):
        return np.hstack([var.upper for var in self.vars.values()])
    @property
    def cl(self):
        return np.hstack([constr.lower for constr in self.constrs.values()])
    @property
    def cu(self):
        return np.hstack([constr.upper for constr in self.constrs.values()])
    @property
    def xrand(self):
        return np.random.random(self.dim)
    
    def set_lower_upper(self, dim, lower, upper):
        if lower is None:
            lower = np.full(dim, -np.inf)
        elif np.isscalar(lower):
            lower = np.full(dim, lower)
        if upper is None:
            upper = np.full(dim, np.inf)
        elif np.isscalar(upper):
            upper = np.full(dim, upper)
        return lower, upper
        
    def add_var(self, name, dim, lower=None, upper=None):
        lower, upper = self.set_lower_upper(dim, lower, upper)
        self.vars[name] = Variable(name, dim, lower, upper)
    
    def add_con(
            self, name, dim, wrt, fn, 
            eq=None, lower=None, upper=None,
            est_sparsity=False
        ): #, jac=None
        if eq is not None:
            lower, upper = self.set_lower_upper(dim, eq, eq)
        else:
            lower, upper = self.set_lower_upper(dim, lower, upper)
        
        jac_sparsity = None
        if est_sparsity:
            jac_sparsity = {}
            for i, var in enumerate(wrt):
                x = np.random.random(self.dim)
                inputs = [np.random.random(self.vars[vname].dim) for vname in wrt]
                sparsity = jax.jacrev(fn, argnums=i)(*inputs) != 0.
                jac_sparsity[var] = sparsity
        self.constrs[name] = Constraint(name, dim, wrt, fn, lower, upper, jac_sparsity) #, jac
    
    def add_objective(self, fn):
        self.obj_fn = fn

    def finalize(self):
        st = 0
        for var in self.vars.values():
            var.st = st
            var.ed = st + var.dim
            st = st + var.dim
        
        st = 0
        for constr in self.constrs.values():
            constr.st = st
            constr.ed = st + constr.dim
            st = st + constr.dim
        self.finalized = True
        
    def print_sparsity(self, element=False):
        vars_str = ", ".join([var.__repr__() for var in self.vars.values()])

        if not element:
            print(f"variables: {vars_str}")
            for cname, constr in self.constrs.items():
                str_row = f"{cname}({constr.dim})\t\t: "
                for vname, var in self.vars.items():
                    if vname in constr.wrt:
                        str_row += "o"
                    else:
                        str_row += "x"
                    str_row += " | "
                print(str_row)
        else:
            print(self.get_jac_sparsity_pattern().astype(int))
    
    def get_objective_fn(self):
        assert self.finalized == True
        def objective(x):
            xdict = {var.name:x[var.st:var.ed] for var in self.vars.values()}
            #inputs = [xdict[var] for var in self.vars.keys()]
            return self.obj_fn(**xdict).flatten()[0]
        return objective
        
    def get_jac_sparsity_pattern(self, return_index=False):
        bool_index = []
        for constr in self.constrs.values():
            bool_row = []
            for vname, var in self.vars.items():
                if vname in constr.wrt:
                    if constr.jac_sparsity is not None:
                        jac_bool_block = constr.jac_sparsity[vname] #estimated sparsity
                    else:
                        jac_bool_block = np.full((constr.dim, var.dim), True)
                else:
                    jac_bool_block = np.full((constr.dim, var.dim), False)
                bool_row.append(jac_bool_block)
            bool_index.append(np.hstack(bool_row))
        bool_index = np.vstack(bool_index)
        if return_index:
            return np.nonzero(bool_index)
        return bool_index
    
    def get_constr_fn(self):
        #constraints
        def constraints(x):
            xdict = {var.name:x[var.st:var.ed] for var in self.vars.values()}
            output = []
            for constr in self.constrs.values():
                inputs = {var:xdict[var] for var in constr.wrt}
                output.append(constr.fn(**inputs))
            return jnp.hstack(output)
        return constraints
        
    def get_jac_fn(self, sparse=True):
        constr_fn = self.get_constr_fn()
        jac = jax.jacrev(constr_fn)
        if not sparse:
            return jac
        rows, cols = self.get_jac_sparsity_pattern(return_index=True)
        jac_sparse = lambda x: jac(x)[rows, cols]
        return jac_sparse
    
    def get_jac_structure_fn(self):
        rows, cols = self.get_jac_sparsity_pattern(return_index=True)
        def jacobianstructure():
            return rows, cols
        return jacobianstructure
    
    def build(self):
        self.finalize()
        fns = {}
        fns["objective"] = self.get_objective_fn()
        fns["gradient"] = jax.grad(fns["objective"])
        fns["constraints"] = self.get_constr_fn()
        fns["jacobian"] = self.get_jac_fn()
        fns["jacobianstructure"] = self.get_jac_structure_fn()
        x = jnp.zeros(self.dim)
        for name, fn in fns.items():
            if name in ["objective", "gradient", "constraints", "jacobian"]:
                setattr(self, name, jax.jit(fn).lower(x).compile())
            if name in ["jacobianstructure"]:
                setattr(self, name, fn)
        
        self.ipopt = cyipopt.Problem(
            n=self.dim, m=self.constr_dim,
            problem_obj=self,
            lb=self.lb, ub=self.ub,
            cl=self.cl, cu=self.cu
        )
    
    def solve(self, x0, verbosity=5, tol=1e-1, viol_tol=1e-1):
        if self.ipopt is None:
            self.build()
        self.ipopt.add_option("acceptable_tol", tol)
        self.ipopt.add_option("acceptable_constr_viol_tol", viol_tol)
        self.ipopt.add_option("acceptable_iter", 2)
        self.ipopt.add_option("print_level", verbosity)
        tic = time.process_time()
        xsol, info = self.ipopt.solve(x0)
        elapsed = time.process_time() - tic
        print(f"elapsed: {elapsed}")
        return xsol, info

"""gradient_descent code"""
# x = prob.xrand

# optimizer = optax.adam(learning_rate=1e-2)
# opt_state = optimizer.init(x)
# obj_fn = prob.get_objective_fn()
# constr_fn = prob.get_constr_fn()
# def merit_fn(x):
#     upper_viol = jnp.clip(constr_fn(x) - prob.cu, a_min=0.)
#     lower_viol = jnp.clip(prob.cl - constr_fn(x), a_min=0.)
#     return obj_fn(x) + upper_viol.sum() + lower_viol.sum()
# grad_fn = jax.jit(jax.grad(merit_fn))
# merit_fn = jax.jit(merit_fn)

#loop
# x_grad = grad_fn(x)
# updates, opt_state = optimizer.update(x_grad, opt_state, x)
# x = optax.apply_updates(x, updates)





"""legacy"""

    # def get_constr_jac_fn1(self):
    #     def jacobian(x):
    #         xdict = {var.name:x[var.st:var.ed] for var in self.vars.values()}
    #         # rows, cols = [], []
    #         vals = []
    #         for constr in self.constrs.values():
    #             inputs = [xdict[var] for var in constr.wrt]
    #             vals_row = []
    #             for name, var in self.vars.items():
    #                 #for var, fn in constr.jac.items():
    #                 # row = self.variables[var].indices
    #                 # for col in range(constr.st, constr.ed):        
    #                 #     rows.append(row)
    #                 #     cols.append(np.ones_like(row) * col)
    #                 if name in constr.jac:
    #                     val = constr.jac[name](*inputs)
    #                 else:
    #                     val = np.zeros((constr.dim, var.dim), val)
    #                 vals_row.append(val)
    #             vals.append(jnp.hstack(vals_row))
    #         # rows = jnp.hstack(rows)
    #         # cols = jnp.hstack(cols)
    #         return jnp.vstack(vals)
    #     return jacobian
    
    # def get_jac_structure_fn(self):
    #     rows, cols = [], []
    #     for constr in self.constrs.values():
    #         for var, fn in constr.jac.items():
    #             col = self.vars[var].indices
    #             for row in range(constr.st, constr.ed):        
    #                 cols.append(col)
    #                 rows.append(np.ones_like(col) * row)
    #     rows = np.hstack(rows)
    #     cols = np.hstack(cols)
    #     def jacobianstructure():
    #         return rows, cols
    #     return jacobianstructure