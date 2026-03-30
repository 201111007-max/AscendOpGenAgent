import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs SwiGLU with quantization.
    torch_npu.npu_swiglu_quant(x, *, smooth_scales=None, offsets=None, group_index=None, activate_left=False, quant_mode=0, group_list_type=0, dst_type=None) -> (Tensor, Tensor)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, smooth_scales: torch.Tensor = None, offsets: torch.Tensor = None,
                group_index: torch.Tensor = None, activate_left: bool = False, quant_mode: int = 0,
                group_list_type: int = 0, dst_type = None) -> tuple:
        """
        Performs SwiGLU with quantization.

        Args:
            x (torch.Tensor): Target tensor. Must be >1D, last axis must be even and <= 8192.
                              dtype: float16, bfloat16, float32, format: ND.
                              For int4 quantization, last dim must be multiple of 4.
            smooth_scales (torch.Tensor, optional): Smooth quantization scale.
                                                    dtype: float32, format: ND. Shape: [G, N] or [G, ].
            offsets (torch.Tensor, optional): Quantization offset. Not used in dynamic quantization.
                                              dtype: float, format: ND. Shape must match smooth_scales.
            group_index (torch.Tensor, optional): Group index tensor (cumsum or count mode).
                                                  dtype: int32, format: ND. Shape: [G, ].
                                                  Must be non-decreasing, max <= product of non-last dims.
            activate_left (bool, optional): Whether to activate left in SwiGLU. Default: False.
            quant_mode (int, optional): Quantization type. 0: static, 1: dynamic. Default: 0.
            group_list_type (int, optional): Group index type. 0: cumsum, 1: count. Default: 0.
            dst_type: Output quantization type. Supports int8 and int4. None means int8. Default: None.

        Returns:
            tuple: (output tensor, quantization parameters) after SwiGLU quantization.
        """
        return torch_npu.npu_swiglu_quant(x, smooth_scales=smooth_scales, offsets=offsets,
                                          group_index=group_index, activate_left=activate_left,
                                          quant_mode=quant_mode, group_list_type=group_list_type,
                                          dst_type=dst_type)
