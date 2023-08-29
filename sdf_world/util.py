import numpy as np
import jax
from jaxlie import SE3, SO3
from jax import Array
import jax.numpy as jnp
import matplotlib
from typing import *
from functools import partial
from matplotlib.colors import rgb2hex, hex2color

class Colors:
    colors=dict(
        red=0xF60000,
        orange=0xFF8C00,
        yellow=0xFFEE00,
        green=0x4DE94C,
        blue=0x3783FF,
        purple=0x4815AA,
        grey=0x868686,
        gray=0x868686,
        darkgray=0x333333,
        white=0xF6F6F6,
        black=0x272727,
    )
    
    @staticmethod
    def read(x, return_rgb=False):
        if isinstance(x, list) or isinstance(x, np.ndarray):
            assert len(x) == 3
            rgb = x
        elif isinstance(x, str):
            rgb = Colors.from_str(x)
        elif isinstance(x, int):
            rgb = hex2color(x) #hex
        elif x is None:
            rgb = Colors.get_random_color() #random
        else:
            raise NotImplementedError("?")
        if return_rgb:
            return rgb
        return Colors.rgb2hex(x)
        
    @staticmethod
    def from_str(string):
        """output:rgb"""
        if string in Colors.colors:
            return Colors.hex2rgb(Colors.colors[string])
        elif string == "random":
            return Colors.get_random_color()
        else:
            print("No matched color. A random color selected.")
            return Colors.get_random_color()

    @staticmethod
    def rgb2hex(rgb_val):
        return int(rgb2hex(rgb_val).replace("#", "0x"), 16)
    
    @staticmethod
    def hex2rgb(hex_val):
        return hex2color(hex(hex_val).replace("0x", "#"))
    
    @staticmethod
    def get_random_color():
        cmap = matplotlib.colormaps["gist_rainbow"]
        rgb = cmap(np.random.random())[:3]
        return rgb

to_numpy = lambda x: np.array(x, dtype=np.float64)
def to_numpy_tfmat(x):
    if isinstance(x, SE3):
        return to_numpy(x.as_matrix())
    elif isinstance(x, Array):
        if x.shape == (7,):
            return to_numpy(SE3(x).as_matrix())
        else:
            return to_numpy(SE3.from_matrix(x))
    elif isinstance(x, np.ndarray):
        return x
    raise NotImplementedError(type(x))

def to_SE3(x):
    if isinstance(x, np.ndarray):
        return SE3.from_matrix(x)
    raise NotImplementedError(type(x))

def mat_from_translate(translate):
    translate = jnp.array(translate)
    return to_numpy(SE3.from_translation(translate).as_matrix())

def str2arr(string):
    if string is None:
        return None
    return np.array(string.split(" ")).astype(float)

def farthest_point_sampling(points, num_samples):
    farthest_points = np.zeros((num_samples, 3))
    farthest_points[0] = points[np.random.randint(len(points))]
    distances = np.full(points.shape[0], np.inf)
    for i in range(1, num_samples):
        distances = np.minimum(distances, np.linalg.norm(points - farthest_points[i - 1], axis=1))
        farthest_points[i] = points[np.argmax(distances)]
    return farthest_points

def fibonacci_sphere(num_samples):
    phi = jnp.pi * (3. - jnp.sqrt(5.))  # golden angle in radians
    indices = jnp.arange(num_samples)
    y = 1 - (indices / (num_samples - 1)) * 2
    r = jnp.sqrt(1 - y*y)
    theta = phi * indices
    x = jnp.cos(theta) * r
    z = jnp.sin(theta) * r
    points = jnp.vstack([x, y, z]).T
    return points

def super_fibonacci_spiral(n):
    phi = jnp.sqrt(2)
    psi = 1.533751168755204288118041
    indices = jnp.arange(n)
    s = indices + 1/2
    t = s/n
    d = 2*jnp.pi*s
    r = jnp.sqrt(t)
    R = jnp.sqrt(1-t)
    alpha = d/phi
    beta = d/psi
    qtns = jnp.vstack([r*jnp.sin(alpha), r*jnp.cos(alpha), R*jnp.sin(beta), R*jnp.cos(beta)])
    return qtns.T

def to_spherical_coord(xyz):
    x, y, z = xyz
    r = jnp.linalg.norm(xyz)
    theta = jnp.arccos(z/r)
    phi = jnp.sign(y)*jnp.arccos(x/jnp.sqrt(x**2+y**2))
    return jnp.array([r, theta, phi])
def to_cartesian_coord(r_theta_phi):
    r, theta, phi = r_theta_phi
    x = r*jnp.sin(theta)*jnp.cos(phi)
    y = r*jnp.sin(theta)*jnp.sin(phi)
    z = r*jnp.cos(theta)
    return jnp.array([x,y,z])

def point_to_T(p):
    return np.array(SE3.from_translation(p).as_matrix(), dtype=float)


from jax._src import linear_util as lu
from jax._src.api_util import argnums_partial
from jax._src.api import _std_basis, _jacfwd_unravel, _jvp

def value_and_jacfwd(fun, argnums):
    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args,
                                            require_static_args_hashable=False)
        pushfwd: Callable = partial(_jvp, f_partial, dyn_args)
        y, jac = jax.vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
        
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = jax.tree_map(partial(_jacfwd_unravel, example_args), y, jac)
        return y, jac_tree
    return jacfun