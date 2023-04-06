"""Optimization and parameter handling."""

import torch.autograd as autograd
import torch
import copy
import torch.nn as nn
import wandb
from torch.optim import Adam
import pdb
from string import digits
import time

def gradient(loss: torch.tensor,
             params: list,
             name,
             second_order: bool = False,
             ) -> torch.tensor:
    """Compute gradient.

    Compute gradient of loss with respect to parameters.

    Parameters
    ----------
    loss : torch.tensor
        Scalar that depends on params.
    params : list
        Sequence of tensors that the gradient will be computed with
        respect to
    name: str
        Name of the model, whose gradient norm will be uploaded to wandb.
    second_order : bool
        Select to compute second or higher order derivatives.

    Returns
    -------
    torch.tensor
        Flattened gradient.

    Examples
    --------
    loss = torch.abs(model(y) - y)
    grad = gradient(loss, model.parameters())

    """
    try:
        grad = autograd.grad(loss, params.values(), retain_graph=True,
                             create_graph=second_order)
    except RuntimeError:
        pdb.set_trace()

    # wandb.log({f' Grad norm {name}': torch.norm(
    #     nn.utils.parameters_to_vector(grad)).cpu()})

    return nn.utils.parameters_to_vector(grad)


def params_update_shalllow(params: list,
                           grad: torch.tensor,
                           lr: float
                           ) -> list:
    """Apply gradient descent update to params.

    It rewrites the params to save the updated params. Do not use this
    when computing higher order derivatives.

    Parameters
    ----------
    params : list
        Sequences of tensors. These are the base parameters that will
        be updated.
    grad : torch.tensor
        Flattened tensor containing the gradient.
    lr : float
        Learning rate.

    Returns
    -------
    list
        The updated parameters.

    Examples
    --------
    grad = torch.ones(10)
    w, b = torch.rand(5), torch.rand(b)
    params = {'weight': nn.Parameter(w), 'bias': nn.Parameter(b)}
    lr = 0.1
    new_params = params_update_deep(params, grad, lr)

    """
    params_updt = copy.copy(params)
    start, end = 0, 0
    for name, param in params.items():
        start = end
        end = start + param.numel()
        update = grad[start:end].reshape(param.shape)
        params_updt[name] = param - lr * update
    return params_updt


def GD_full_update(params: dict,
                   losses: list,
                   keys: list,
                   lr: float,
                   ) -> list:
    """Compute GD for multiple models.

    Compute and update parameters of models in the params list with
    respect to the losses in the loss list. Do not use for second or
    higher order derivatives.

    Parameters
    ----------
    params : list
        Each element of the dictionary has the parameters of a model.
    losses : list
        Each element of the list is a scalar tensor. It should match
        the order of the params dict, e.g., first element of loss,
        should correspond to first params.
    lr : float
        Learning rate.

    Returns
    -------
    list
        Update dictionary with all parameters.

    """
    for loss, key in zip(losses, keys):
        grad = gradient(loss, params[key], key)
        params[key] = params_update_shalllow(params[key], grad, lr)
    return params


def Adam_update(params: list[dict],
                losses: list[float],
                keys: list[str],
                optimizers: list,
                ):

    for loss, key in zip(losses, keys):
        optimizers[key].zero_grad()
        loss.backward(retain_graph=True)
        if 'Critic' in key:
            nn.utils.clip_grad_norm_(params[key].values(), 50)
        if 'LengthPolicy' in key:
            nn.utils.clip_grad_norm_(params[key].values(), 100)
        optimizers[key].step()

    return params
    
def set_optimizers(params, keys, lr):
    optimizers = {}
    for key in keys:
        if 'Level' in key:
            val = key[5:]
            parameters = (*params[f'Encoder{val}'].values(),
                          *params[f'Decoder{val}'].values())
        else:
            parameters = params[key].values()

        optimizers[key] = Adam(parameters, lr=lr)

    return optimizers
        
