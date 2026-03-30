import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs Rotary Position Embedding multiplication.
    torch_npu.npu_rotary_mul(input, r1, r2, rotary_mode='half') -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, rotary_mode: str = 'half') -> torch.Tensor:
        """
        Applies rotary position embedding multiplication to the input tensor.

        Args:
            input (torch.Tensor): Input tensor, must be 4D. Supports float16, bfloat16, float32.
            r1 (torch.Tensor): Cosine rotation coefficient, must be 4D. Supports float16, bfloat16, float32.
            r2 (torch.Tensor): Sine rotation coefficient, must be 4D. Supports float16, bfloat16, float32.
            rotary_mode (str, optional): Computation mode, supports 'half' and 'interleave'. Default: 'half'.

        Returns:
            torch.Tensor: Output tensor with rotary position embedding applied.
        """
        return torch_npu.npu_rotary_mul(input, r1, r2, rotary_mode=rotary_mode)
