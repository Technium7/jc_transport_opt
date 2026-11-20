"""NumPy to PyTorch conversion that does not depend on NumPy 2 dtype hooks.

Torch 2.1 raises ``Could not infer dtype of numpy.float32`` against NumPy 2.
Copying through a typed buffer avoids that path.
"""

from __future__ import annotations

import numpy as np
import torch


def to_tensor(array, dtype=None, device=None):
    array = np.asarray(array)
    if dtype is None:
        if array.dtype == np.bool_:
            dtype = torch.bool
        elif np.issubdtype(array.dtype, np.integer):
            dtype = torch.long
        elif array.dtype == np.float64:
            dtype = torch.float64
        else:
            dtype = torch.float32
    if dtype in (torch.bool, torch.uint8) and dtype is torch.bool:
        raw = np.ascontiguousarray(array, dtype=np.uint8)
        out = torch.frombuffer(bytearray(raw.tobytes()), dtype=torch.uint8).clone().reshape(raw.shape).bool()
    elif dtype in (torch.long, torch.int64):
        raw = np.ascontiguousarray(array, dtype=np.int64)
        out = torch.frombuffer(bytearray(raw.tobytes()), dtype=torch.long).clone().reshape(raw.shape)
    elif dtype in (torch.float64, torch.double):
        raw = np.ascontiguousarray(array, dtype=np.float64)
        out = torch.frombuffer(bytearray(raw.tobytes()), dtype=torch.float64).clone().reshape(raw.shape)
    else:
        raw = np.ascontiguousarray(array, dtype=np.float32)
        out = torch.frombuffer(bytearray(raw.tobytes()), dtype=torch.float32).clone().reshape(raw.shape)
    if device is not None:
        out = out.to(device)
    return out


def tensor_to_numpy(tensor):
    """Copy a CPU tensor into a NumPy array without ``Tensor.numpy()``."""
    import ctypes

    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bool:
        flat = tensor.reshape(-1).tolist()
        return np.array(flat, dtype=np.bool_).reshape(tensor.shape)
    kind = {
        torch.float32: (np.float32, 4),
        torch.float64: (np.float64, 8),
        torch.int64: (np.int64, 8),
        torch.int32: (np.int32, 4),
    }
    if tensor.dtype not in kind:
        raise TypeError("unsupported tensor dtype {}".format(tensor.dtype))
    np_dtype, item = kind[tensor.dtype]
    raw = tensor.view(-1)
    blob = bytearray(int(raw.numel()) * item)
    buf = (ctypes.c_char * len(blob)).from_buffer(blob)
    ctypes.memmove(ctypes.addressof(buf), raw.data_ptr(), len(blob))
    return np.frombuffer(blob, dtype=np_dtype).reshape(tensor.shape).copy()
