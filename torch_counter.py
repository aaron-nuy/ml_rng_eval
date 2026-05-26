import ctypes
import os
import sys
import torch

_lib = None

def _init_lib():
    global _lib
    if _lib is not None:
        return

    native_torch_lib_directory = os.path.join(os.path.dirname(torch.__file__), 'lib')
    native_torch_lib_path = os.path.join(native_torch_lib_directory, 'libtorch_cpu.so')

    if not os.path.exists(native_torch_lib_path):
        sys.exit(f"Couldn't find native torch library at: {native_torch_lib_path}")

    _lib = ctypes.CDLL(native_torch_lib_path)

    _lib.get_pytorch_rng_call_count.argtypes = []
    _lib.get_pytorch_rng_call_count.restype = ctypes.c_uint64

    _lib.reset_pytorch_rng_call_count.argtypes = []
    _lib.reset_pytorch_rng_call_count.restype = None


def get_count() -> int:
    _init_lib()
    return _lib.get_pytorch_rng_call_count()


def reset_count() -> None:
    _init_lib()
    _lib.reset_pytorch_rng_call_count()