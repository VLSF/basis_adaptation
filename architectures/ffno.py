import jax.numpy as jnp

from architectures import split_skip
from jax import random
from jax.lax import dot_general, dynamic_slice_in_dim

class FFNO(split_skip.split_skip):
    A: jnp.array

    def __init__(self, N_layers, N_features, N_modes, D, key):
        keys = random.split(key)
        super().__init__(N_layers, N_features, D, keys[0])
        self.A = random.normal(keys[1], [N_layers, N_features[1], N_features[1], N_modes, D], dtype=jnp.complex64)*jnp.sqrt(2/N_features[1])

    def space_mixer(self, v, x, j):
        u = 0
        N = v.shape
        for i in range(self.A[j].shape[-1]):
            u_ = jnp.fft.rfft(v, axis=i+1)
            N_modes = min(self.A[j].shape[-2], u_.shape[i+1])
            u_ = dynamic_slice_in_dim(u_, 0, N_modes, axis=i+1)
            u_ = dot_general(self.A[j][:, :, :N_modes, i], u_, (((1,), (0,)), ((2, ), (i+1, ))))
            u_ = jnp.moveaxis(u_, 0, i+1)
            u += jnp.fft.irfft(u_, axis=i+1, n=N[i+1])
        return u