import sys
from pathlib import Path

import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from design.tile_level.kv_sort import kv_sort as tl_kv_sort


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _build_kernel(self, keys: torch.Tensor):
        n = keys.shape[0]
        return tl_kv_sort(n, dtype="int32")

    def forward(self, keys: torch.Tensor, values: torch.Tensor):
        assert keys.ndim == 1 and values.ndim == 1
        assert keys.shape[0] == values.shape[0]
        assert keys.dtype == torch.int32
        assert values.dtype == torch.int32

        kernel = self._build_kernel(keys)
        sorted_keys, sorted_values = kernel(keys, values)
        return sorted_keys, sorted_values
