import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs top-k and top-p filtering.
    torch_npu.npu_top_k_top_p(logits, p, k) -> torch.Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, logits: torch.Tensor, p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Performs top-k and top-p filtering on logits.

        Args:
            logits (torch.Tensor): Data to process. Must be 2D.
                                   dtype: float32, float16, bfloat16, format: ND.
                                   Supports non-contiguous tensors.
            p (torch.Tensor): Top-p threshold tensor. Range: [0, 1].
                              dtype: float32, float16, bfloat16 (must match logits).
                              Must be 1D with size matching logits' first dimension.
                              format: ND, supports non-contiguous tensors.
            k (torch.Tensor): Top-k threshold tensor. Range: [1, 1024], max <= logits.size(1).
                              dtype: int32. Must be 1D with size matching logits' first dimension.
                              format: ND, supports non-contiguous tensors.

        Returns:
            torch.Tensor: Filtered logits tensor.
        """
        return torch_npu.npu_top_k_top_p(logits, p, k)
