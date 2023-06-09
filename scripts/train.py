import os
import jax
from flax.training.train_state import TrainState
from typing import Callable, Tuple
import numpy as np
import orbax.checkpoint
from torch.utils.data import DataLoader
from functools import partial

from torch.utils.tensorboard import SummaryWriter

def train_step_fn(state:TrainState, batch:Tuple, loss_fn:Callable):
    """
    Function to perform a single step of training.

    Args:
        state (TrainState): The current state of the model.
        batch (tuple): A tuple of data points used for training.
        loss_fn (function): A function that calculates the loss.

    Returns:
        state (TrainState): The updated state of the model.
        loss (float): The calculated loss.
    """
    grad_fn = jax.value_and_grad(loss_fn, argnums=1)
    loss, grads = grad_fn(state, state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss

def eval_step_fn(state:TrainState, batch, loss_fn):
    """
    Function to evaluate a model on a given batch of data.

    Args:
        state (TrainState): The current state of the model.
        batch (tuple): A tuple of data points used for evaluation.
        loss_fn (function): A function that calculates the loss.

    Returns:
        loss (float): The calculated loss.
    """
    loss = loss_fn(state, state.params, batch)
    return loss

def trainer(
    state:TrainState, 
    train_loader:DataLoader, 
    val_loader:DataLoader,
    loss_fn:Callable,
    num_epochs:int,
    exp_str:str,
    log_dir="./log", 
    ckpt_dir="./checkpoint", 
    save_interval_steps=10,
):
    """
    Train a model using the given parameters.

    Parameters:
    state (TrainState): The current training state of the model.
    train_loader (DataLoader): The DataLoader object for the training data.
    val_loader (DataLoader): The DataLoader object for the validation data.
    loss_fn (Callable): The loss function to use for training and evaluation.
    num_epochs (int): The number of epochs to train for.
    exp_str (str): A string representing the experiment name.
    log_dir (str): The directory to save logs in. Defaults to "./log".
    ckpt_dir (str): The directory to save checkpoints in. Defaults to "./checkpoint".

    Returns:
    state (TrainState): The updated training state of the model.
    """
    # Compile functions with JAX
    train_step = jax.jit(partial(train_step_fn, loss_fn=loss_fn))
    eval_step = jax.jit(partial(eval_step_fn, loss_fn=loss_fn))

    # Set up logging and checkpointing
    log_dir = os.path.join(log_dir, exp_str)
    ckpt_dir = os.path.join(ckpt_dir, exp_str)
    writer = SummaryWriter(log_dir=log_dir)
    ckpt_options = orbax.checkpoint.CheckpointManagerOptions(save_interval_steps=save_interval_steps, max_to_keep=10, create=True)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer, ckpt_options)

    # Training loop
    for epoch in range(num_epochs):
        train_losses = []
        val_losses = []
        # training
        for i, batch in enumerate(train_loader):
            state, loss = train_step(state, batch)
            train_losses += [float(loss)]
            print(f'TRAIN: EPOCH {epoch+1}/{num_epochs} | BATCH {i}/{len(train_loader)} | LOSS: {np.mean(train_losses)}')
        
        # validation
        for i, batch in enumerate(val_loader):
            loss = eval_step(state, batch)
            val_losses += [float(loss)]
            print(f'VAL: EPOCH {epoch+1}/{num_epochs} | BATCH {i}/{len(val_loader)} | LOSS: {np.mean(val_losses)}')
        
        # logging and checkpoint saving
        writer.add_scalar('loss/train', np.mean(train_losses), epoch)
        writer.add_scalar('loss/val', np.mean(val_losses), epoch)
        checkpoint_manager.save(epoch, state.params)
    return state

