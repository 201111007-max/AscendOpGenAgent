import torch
import torch.nn as nn

from flash_attention.design.tile_level.flash_attention import (
    flash_attention_fwd as tl_flash_attention_fwd,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _build_kernel(self, q: torch.Tensor):
        batch, heads, seq_len, dim = q.shape
        return tl_flash_attention_fwd(
            batch,
            seq_len,
            heads,
            dim,
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
        assert q.shape == k.shape == v.shape
        assert q.dtype == k.dtype == v.dtype == torch.float16

        kernel = self._build_kernel(q)
        return kernel(q, k, v)
