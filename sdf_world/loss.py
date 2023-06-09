
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from typing import Dict

def lipschitz_loss(params):
    loss = 1.
    for layer in params['params'].keys():
        loss *= nn.softplus(params['params'][layer]['c'])
    return loss[0]

def l2_loss_fn(state:TrainState, params:Dict, batch:tuple):
    x, y = batch
    y_pred = state.apply_fn(params, x).squeeze()
    loss = optax.l2_loss(y_pred, y).mean()
    return loss