# Adapted from balboa. Copyright (c) 2026 Tzu-Mao Li. See THIRD_PARTY_NOTICES.md.
"""GPU buffer uploads and compute shader loading used by this project."""

import numpy as np
import slangpy as spy


class ComputeShader:
    def __init__(self, module, kernel):
        self.module = module
        self.kernel = kernel


def load_compute_shader(device, filename, entry_name="compute_main"):
    module = device.load_module(filename)
    entry = module.entry_point(entry_name)
    program = device.link_program([module], [entry])
    kernel = device.create_compute_kernel(program)
    reflected_module = spy.Module.load_from_module(device, module)
    return ComputeShader(reflected_module, kernel)


def structured_typed_buffer_from_numpy(device, values, dtype):
    values = np.ascontiguousarray(values)
    if len(values) == 0:
        values = np.zeros((1, *values.shape[1:]), dtype=values.dtype)
    tensor = spy.Tensor.empty(device, shape=(len(values),), dtype=dtype)
    tensor.copy_from_numpy(values)
    return tensor.storage


def create_structured_buffer(device, dtype, values):
    tensor = spy.Tensor.empty(device, shape=(max(len(values), 1),), dtype=dtype)
    cursor = tensor.cursor()
    for index, value in enumerate(values):
        cursor[index].write(value)
    cursor.apply()
    return tensor.storage
