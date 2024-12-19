import jax.numpy as jnp
import equinox as eqx

from jax import random
from jax.nn import gelu

def normalize_conv(A):
    A = eqx.tree_at(lambda x: x.weight, A, A.weight*jnp.sqrt(2/A.weight.shape[1]))
    A = eqx.tree_at(lambda x: x.bias, A, jnp.zeros_like(A.bias))
    return A

class split_skip(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    convs1: list
    convs2: list

    def __init__(self, N_layers, N_features, D, key):
        n_in, n_processor, n_out = N_features
        
        keys = random.split(key, 3 + 2*N_layers)
        self.encoder = normalize_conv(eqx.nn.Conv(D, n_in, n_processor, 1, key=keys[-1]))
        self.decoder = normalize_conv(eqx.nn.Conv(D, n_processor, n_out, 1, key=keys[-2]))
        self.convs1 = [normalize_conv(eqx.nn.Conv(D, n_processor, n_processor, 1, key=key)) for key in keys[:N_layers]]
        self.convs2 = [normalize_conv(eqx.nn.Conv(D, n_processor, n_processor, 1, key=key)) for key in keys[N_layers:2*N_layers]]
        
    def __call__(self, u, x):
        u = jnp.concatenate([x, u], 0)
        u = self.encoder(u)
        for i in range(len(self.convs1)):
            u += gelu(self.convs2[i](gelu(self.convs1[i](self.space_mixer(u, x, i)))))
        u = self.decoder(u)
        return u

    def space_mixer(self, u, x, i):
        return u