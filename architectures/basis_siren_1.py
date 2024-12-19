import jax.numpy as jnp
import equinox as eqx

from jax import random
from jax.lax import dot_general
from architectures import siren, split_skip

class SL_siren_layer(eqx.Module):
    NN: eqx.Module
    A: jnp.array

    def __init__(self, N_features, N_modes, D, N_features_siren, N_layers_siren, key, eps=1e-1):
        keys = random.split(key)
        self.A = random.normal(keys[0], [N_features, N_features, N_modes, D])*jnp.sqrt(2/N_features)
        self.NN = siren.siren([1, N_features_siren, N_modes], N_layers_siren, keys[1])

    def __call__(self, v, x, axis, eps_=1e-3):
        basis = self.NN(jnp.expand_dims(x, 1))
        res = dot_general(basis, v, (((0,), (axis+1,)), ((), ())))
        res = dot_general(self.A[..., axis], res, (((1,), (1,)), ((2,), (0,))))
        res = dot_general(basis, res, (((1,), (0,)), ((), ()))) / x.shape[0]
        res = jnp.moveaxis(res, 0, axis+1)
        return res

class SLNO_siren(split_skip.split_skip):
    sl_basis_layers: list

    def __init__(self, N_layers, N_features, N_modes, D, N_features_siren, N_layers_siren, key, eps=1e-2):
        keys = random.split(key, N_layers+1)
        super().__init__(N_layers, N_features, D, keys[-1])
        self.sl_basis_layers = [SL_siren_layer(N_features[1], N_modes, D,  N_features_siren, N_layers_siren, key, eps=eps) for key in keys[:-1]]

class SLNO_siren_1D(SLNO_siren):
    def space_mixer(self, v, x, j):
        u = self.sl_basis_layers[j](v, x[0, :], 0)
        return u

class SLNO_siren_2D(SLNO_siren):
    def space_mixer(self, v, x, j):
        u = 0
        u += self.sl_basis_layers[j](v, x[0, :, 0], 0)
        u += self.sl_basis_layers[j](v, x[1, 0, :], 1)
        return u

class SLNO_siren_3D(SLNO_siren):
    def space_mixer(self, v, x, j):
        u = 0
        u += self.sl_basis_layers[j](v, x[0, :, 0, 0], 0)
        u += self.sl_basis_layers[j](v, x[1, 0, :, 0], 1)
        u += self.sl_basis_layers[j](v, x[2, 0, 0, :], 2)
        return u