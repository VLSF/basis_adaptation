import jax.numpy as jnp
import equinox as eqx

from jax import vmap
from jax.tree_util import tree_map

def compute_loss(model, input, target, x):
    output = vmap(model, in_axes=(0, None))(input, x)
    l = jnp.mean(jnp.linalg.norm((output - target).reshape(input.shape[0], -1), axis=1))
    return l

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)
jit_apply_model = eqx.filter_jit(lambda u, x, model: vmap(model, in_axes=(0, None))(u, x))

def make_step_scan(carry, n, optim):
    model, features, targets, x, opt_state = carry
    loss, grads = compute_loss_and_grads(model, features[n], targets[n], x)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, x, opt_state], loss

def compute_error(carry, ind):
    model, features, targets, x = carry
    prediction = model(features[ind], x)
    error = jnp.linalg.norm((prediction - targets[ind]).reshape(targets.shape[1], -1)) / jnp.linalg.norm((targets[ind]).reshape(targets.shape[1], -1))
    return carry, error