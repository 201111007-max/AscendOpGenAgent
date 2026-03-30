import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs interleave RoPE (Rotary Position Embedding).
    torch_npu.npu_interleave_rope(x, cos, sin) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Performs interleave RoPE on input tensor.

        Args:
            x (torch.Tensor): Input tensor to process. Must be 4D with shape (B, N, S, D).
                              dtype: bfloat16, float16, format: ND.
                              Does not support non-contiguous tensors.
            cos (torch.Tensor): RoPE cosine component. Must be 4D with shape (B, N, S, D).
                                S can be 1 or same as x's S. dtype and format must match x.
                                Does not support non-contiguous tensors.
            sin (torch.Tensor): RoPE sine component. Shape, dtype and format must match cos.
                                Does not support non-contiguous tensors.

        Returns:
            torch.Tensor: Output tensor after interleave RoPE, same shape as input x.
        """
        return torch_npu.npu_interleave_rope(x, cos, sin)
