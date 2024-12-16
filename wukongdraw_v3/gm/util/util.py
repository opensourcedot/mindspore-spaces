import importlib
import random
from functools import partial
from inspect import isfunction

import numpy as np
from packaging import version

import mindspore as ms
from mindspore import nn
from mindspore.train.amp import AMP_BLACK_LIST, AMP_WHITE_LIST, _auto_black_list, _auto_white_list


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def expand_dims_like(x, y):
    dim_diff = y.dim() - x.dim()
    if dim_diff > 0:
        for _ in range(dim_diff):
            x = x.unsqueeze(-1)
    return x


def count_params(model, verbose=False):
    total_params = sum([p.asnumpy().size for _, p in model.parameters_and_names()])  # tensor.numel()
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def append_zero(x):
    return np.concatenate([x, np.zeros([1], dtype=x.dtype)])


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    for i in range(dims_to_append):
        x = x[..., None]
    return x


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


def new_version():
    return version.parse(ms.__version__) >= version.parse("2.2")
