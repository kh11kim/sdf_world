from jax import lax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Sequence
import orbax
from flax.training import orbax_utils
from flax.training.train_state import TrainState

class LipLinear(nn.Module):
    """A linear layer with Lipschitz regularization."""
    features: int
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the LipLinear layer.
        
        Args:
            inputs (jnp.ndarray): The input tensor.
        
        Returns:
            jnp.ndarray: The output tensor.
        """
        kernel = self.param('kernel',
            self.kernel_init,
            (jnp.shape(inputs)[-1], self.features))
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,))
        else:   bias = None
        c = self.param('c', self.bias_init, 1)
        absrowsum = jnp.sum(jnp.abs(kernel), axis=0)
        scale = jnp.minimum(1.0, nn.softplus(c)/absrowsum)
        y = lax.dot_general(inputs, kernel*scale,
                        (((inputs.ndim - 1,), (0,)), ((), ())))
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y

class MLP(nn.Module):
    dims: Sequence[int]
    skip_layer: int = 0 #if 0, no skip connection
    linear: nn.Module = nn.Dense
    actv_fn: Callable = nn.relu
    out_actv_fn: Callable = None

    def setup(self):
        self.nin = self.dims[0]
        layers = []
        for i, dim in enumerate(self.dims[1:]):
            if i == self.skip_layer and i != 0:
                dim += self.nin
            layers += [self.linear(dim)]
        self.layers = layers
    
    def __call__(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer and i != 0:
                x = jnp.hstack([x, inputs])
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.actv_fn(x)

        if self.out_actv_fn is not None:
            x = self.out_actv_fn(x)
        return x

class Hyperparam:
    """Class to store hyperparameters for a model.
    
    Usage:
        hp = Hyperparam()
        hp.key1 = value1
        hp.key2 = value2
        ...
        if key in hp:
            # do something
        print(hp)
        print(hp.to_str())
    """
    def __contains__(self, key):
        return key in self.__dict__.keys()
    
    def __repr__(self):
        return self.__dict__.__repr__()
    
    def as_str(self):
        result = ""
        for key, item in self.__dict__.items():
            if key == "layers":
                result += key + ":" + "_".join([str(l) for l in item])
            else:
                result += key + ":" + str(item)
            result += ","
        return result[:-1]
    
    def as_dict(self):
        return self.__dict__
    
    @classmethod
    def from_dict(cls, d:dict):
        result = cls()
        for k, v in d.items():
            setattr(result, k, v)
        return result

def get_mlp(hp:Hyperparam):
    skip_layer = 0 if "skip_layer" not in hp else hp.skip_layer
    linear = nn.Dense if "linear" not in hp else getattr(nn, hp.linear)
    actv_fn = nn.relu if "actv_fn" not in hp else getattr(nn, hp.actv_fn)
    out_actv_fn = None if "out_actv_fn" not in hp else getattr(nn, hp.out_actv_fn)
    return MLP(hp.dims, skip_layer, linear=linear, actv_fn=actv_fn, out_actv_fn=out_actv_fn)

def get_mlp_by_path(path):
    net_dict = load(path)
    hp = Hyperparam.from_dict(net_dict['hp'])
    net = get_mlp(hp)
    net_bind = net.bind({"params":net_dict['params']})
    return net_bind

def save(path, state:TrainState, hp:Hyperparam, force=False):
    params = state.params.unfreeze()
    params['hp'] = hp.as_dict()
    
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    orbax_checkpointer.save(path, params, save_args=save_args, force=force)

def load(path):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    return orbax_checkpointer.restore(path)

class ManipNet(nn.Module):
    hidden_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return nn.softplus(x)