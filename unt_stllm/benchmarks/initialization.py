import hashlib
import json
import random

import numpy as np
import torch


def snapshot_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def digest_state(state):
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        result.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def node_parameter(name):
    return name == 'identity' or name.endswith('.relation.persistent')


def capture_initialization(model):
    state = model.state_dict()
    reset = {name: value for name, value in state.items() if node_parameter(name)}
    shared = {name: value for name, value in state.items() if not node_parameter(name)}
    rng = torch.get_rng_state().numpy().tobytes() + repr(np.random.get_state()).encode() + repr(random.getstate()).encode()
    if torch.cuda.is_available():
        rng += b''.join(x.cpu().numpy().tobytes() for x in torch.cuda.get_rng_state_all())
    return dict(target_node_initial_sha256=digest_state(reset),
                initial_shared_sha256=digest_state(shared), initial_rng_sha256=hashlib.sha256(rng).hexdigest())


def copy_shared(model, source):
    current = model.state_dict()
    if set(source) != set(current):
        raise ValueError('Source and target state keys differ')
    merged = dict(current)
    reset, copied = [], []
    for name, value in source.items():
        if not isinstance(value, torch.Tensor) or value.dtype != current[name].dtype:
            raise ValueError('Source state dtype differs at ' + name)
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('Source state is non-finite at ' + name)
        if node_parameter(name):
            if value.ndim != 2 or current[name].ndim != 2:
                raise ValueError('Invalid node state')
            reset.append(name)
        else:
            if value.shape != current[name].shape:
                raise ValueError('Shared shape differs at ' + name)
            merged[name] = value
            copied.append(name)
    if not reset or not copied:
        raise ValueError('Empty reset or shared parameter selection')
    model.load_state_dict(merged, strict=True)
    return dict(source_sha256=digest_state(source), reset=reset, transferred=copied)
