# Training architecture with adaptive basis layer. Separate bases for "analysis" and "synthesis", learnable for each individual layer. Initialised as random siren network.# Training FFNO to get a baseline accuracy for selected hyperparameters.

import jax.numpy as jnp
import equinox as eqx
import optax
import argparse
import os.path
import sys
import time
import hashlib

from jax.tree_util import tree_map, tree_flatten
from jax import random
from jax.lax import scan

from training_loops import training_loop
from architectures import basis_siren_2

def normalise_field(field):
    # field has shape (N_samples, N_features, N_x, N_y, ...)
    original_shape = field.shape
    field = field.reshape(field.shape[0], field.shape[1], -1)
    field = field / jnp.max(jnp.linalg.norm(field, axis=2, keepdims=True), axis=0, keepdims=True)
    return field.reshape(original_shape)

def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
       "-path_to_dataset": {
            "help": "path to dataset in the .npz format"
        },
        "-path_to_results": {
            "help": "path to folder where to save results"
        },
        "-learning_rate": {
            "default": 1e-4,
            "type": float,
            "help": "learning rate"
        },
        "-gamma": {
            "default": 0.5,
            "type": float,
            "help": "decay parameter for the exponential decay of learning rate"
        },
        "-N_batch": {
            "default": 20,
            "type": int,
            "help": "number of samples used to average gradient"
        },
        "-N_train": {
            "default": 5000,
            "type": int,
            "help": "number of samples in the training set"
        },
        "-N_test": {
            "default": 5000,
            "type": int,
            "help": "number of samples in the test set"
        },
        "-N_updates": {
            "default": 10000,
            "type": int,
            "help": "number of updates of the model weights"
        },
        "-N_drop": {
            "default": 10000 // 2,
            "type": int,
            "help": "number of updates after which learning rate is multiplied by chosen learning rate decay"
        },
        "-N_features": {
            "default": 32,
            "type": int,
            "help": "number of features in a hidden layer"
        },
        "-N_layers": {
            "default": 4,
            "type": int,
            "help": "number of layers"
        },
        "-N_modes": {
            "default": 16,
            "type": int,
            "help": "number of basis function"
        },
        "-N_features_siren": {
            "default": 20,
            "type": int,
            "help": "number of features in a hidden siren layer"
        },
        "-N_layers_siren": {
            "default": 3,
            "type": int,
            "help": "number of layers siren"
        },
        "-key": {
            "default": 14,
            "type": int,
            "help": "PRNGKey for network init and training set reshuffle"
        }
    }

    for key in args:
        parser.add_argument(key, **args[key])

    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args = vars(parser.parse_args())
    script_name = sys.argv[0].split(".")[0]

    header = ",".join([key for key in args.keys()])
    header += ",hash,final_loss,model_size,train_error,test_error,training_time"
    if not os.path.isfile(f"{args['path_to_results']}/results.csv"):
        with open(f"{args['path_to_results']}/results.csv", "w") as f:
            f.write(header)

    data = jnp.load(args["path_to_dataset"])
    features = normalise_field(data["features"])
    targets = normalise_field(data["targets"])
    coordinates = data["coordinates"]

    keys = random.split(random.PRNGKey(args["key"]), 2)
    D = features.ndim-2
    NN_args = [args["N_layers"], [features.shape[1] + coordinates.shape[0], args["N_features"], targets.shape[1]], args["N_modes"], features.ndim-2, args["N_features_siren"], args["N_layers_siren"], keys[0]]
    if D == 1:
        model = basis_siren_2.SLNO_siren_1D(*NN_args)
    elif D == 2:
        model = basis_siren_2.SLNO_siren_2D(*NN_args)
    elif D == 3:
        model = basis_siren_2.SLNO_siren_3D(*NN_args)
    else:
        raise ValueError("Only cases D=1, D=2 and D=3 are available")
    model_size = sum(tree_map(jnp.size, tree_flatten(model)[0], is_leaf=eqx.is_array))

    sc = optax.exponential_decay(args["learning_rate"], args["N_drop"], args["gamma"])
    optim = optax.lion(learning_rate=sc)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    ind_train, ind_test = jnp.arange(args["N_train"]), -jnp.arange(1, args["N_test"]+1)
    n = random.choice(keys[1], ind_train, shape = (args["N_updates"], args["N_batch"]))
    carry = [model, features, targets, coordinates, opt_state]

    make_step_scan = lambda a, b: training_loop.make_step_scan(a, b, optim)
    start = time.time()
    res, losses = scan(make_step_scan, carry, n)
    stop = time.time()
    training_time = stop - start
    model, opt_state = res[0], res[-1]

    _, train_errors = scan(training_loop.compute_error, [model, features, targets, coordinates], ind_train)
    _, test_errors = scan(training_loop.compute_error, [model, features, targets, coordinates], ind_test)
    train_error, test_error = jnp.mean(train_errors), jnp.mean(test_errors)

    # saving model and opt state
    exp_hash = hashlib.sha256(str.encode(script_name + "".join([str(args[k]) for k in args.keys()]))).hexdigest()

    eqx.tree_serialise_leaves(f"{args['path_to_results']}/model_{exp_hash}.eqx", model)
    eqx.tree_serialise_leaves(f"{args['path_to_results']}/opt_state_{exp_hash}.eqx", opt_state)

    data = "\n" + ",".join([str(args[key]) for key in args.keys()])
    data += f",{exp_hash},{losses[-1]},{model_size},{jnp.mean(train_error)},{jnp.mean(test_error)},{training_time}"

    with open(f"{args['path_to_results']}/results.csv", "a") as f:
        f.write(data)
