import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs Group Normalization with Swish activation.
    torch_npu.npu_group_norm_swish(input, num_groups, weight, bias, eps=1e-5, swish_scale=1.0) -> (Tensor, Tensor, Tensor)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, num_groups: int, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5, swish_scale: float = 1.0) -> tuple:
        """
        Applies group normalization followed by swish activation.

        Args:
            input (torch.Tensor): Input tensor for group normalization, supports 2-8D tensors.
                                  Supports float16, float32, bfloat16.
            num_groups (int): Number of groups to divide the first dimension into.
                              The first dimension must be divisible by num_groups.
            weight (torch.Tensor): Weight tensor, must be 1D with size equal to input's first dimension.
                                   Supports float16, float32, bfloat16, must match input dtype.
            bias (torch.Tensor): Bias tensor, must be 1D with size equal to input's first dimension.
                                 Supports float16, float32, bfloat16, must match input dtype.
            eps (float, optional): Value added to denominator for numerical stability. Default: 1e-5.
            swish_scale (float, optional): Scale value for swish computation. Default: 1.0.

        Returns:
            tuple: (output tensor, mean, rstd) where output is the normalized result with swish.
        """
        return torch_npu.npu_group_norm_swish(input, num_groups, weight, bias, eps=eps, swish_scale=swish_scale)
