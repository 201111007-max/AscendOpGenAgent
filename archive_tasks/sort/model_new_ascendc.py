import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _kv_sort_ext as _ext  # noqa: E402


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, keys: torch.Tensor, values: torch.Tensor):
        assert keys.dtype == torch.int32
        assert values.dtype == torch.int32
        assert keys.dim() == 1 and values.dim() == 1
        assert keys.shape[0] == values.shape[0]
        assert keys.is_contiguous() and values.is_contiguous()

        sorted_keys, sorted_values, _ = _ext.run_kv_sort(keys, values, -1)
        return sorted_keys, sorted_values
