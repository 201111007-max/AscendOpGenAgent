import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs dequantization followed by SwiGLU and quantization.
    torch_npu.npu_dequant_swiglu_quant(x, *, weight_scale=None, activation_scale=None, bias=None, quant_scale=None, quant_offset=None, group_index=None, activate_left=False, quant_mode=0, swiglu_mode=0, clamp_limit=7.0, glu_alpha=1.702, glu_bias=1.0) -> (Tensor, Tensor)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, weight_scale: torch.Tensor = None, activation_scale: torch.Tensor = None,
                bias: torch.Tensor = None, quant_scale: torch.Tensor = None, quant_offset: torch.Tensor = None,
                group_index: torch.Tensor = None, activate_left: bool = False, quant_mode: int = 0,
                swiglu_mode: int = 0, clamp_limit: float = 7.0, glu_alpha: float = 1.702,
                glu_bias: float = 1.0) -> tuple:
        """
        Performs dequantization followed by SwiGLU and quantization.

        Args:
            x (torch.Tensor): Target tensor. Must be 2D with shape [TokensNum, 2H], last axis even.
                              dtype: int32, bfloat16, format: ND.
            weight_scale (torch.Tensor, optional): Weight dequantization scale. Must be 2D [groupNum, 2H].
                                                   dtype: float32, format: ND. Required when x is int32.
            activation_scale (torch.Tensor, optional): Per-token weight dequantization scale.
                                                       Must be 1D [TokensNum]. dtype: float32.
                                                       Required when x is int32.
            bias (torch.Tensor, optional): Bias variable. dtype: int32, format: ND.
                                           Not effective when group_index is not None.
            quant_scale (torch.Tensor, optional): Smooth quantization scale. Must be 2D [groupNum, H].
                                                  dtype: float32, float16, bfloat16, format: ND.
            quant_offset (torch.Tensor, optional): Quantization offset.
                                                   dtype: float32, float16, bfloat16, format: ND.
                                                   Not effective when group_index is not None.
            group_index (Tensor, optional): Group tokens count (count mode only). Must be 1D.
                                            dtype: int64, format: ND.
            activate_left (bool, optional): Whether to activate left in SwiGLU. Default: False.
            quant_mode (int, optional): Quantization type. 0: static, 1: dynamic. Default: 0.
                                        When group_index is not None, only dynamic (1) is supported.
            swiglu_mode (int, optional): SwiGLU mode. 0: traditional, 1: variant with clamp/alpha/bias.
                                         Default: 0.
            clamp_limit (float, optional): SwiGLU output gate limit. Default: 7.0.
            glu_alpha (float, optional): GLU activation coefficient. Default: 1.702.
            glu_bias (float, optional): SwiGLU computation bias. Default: 1.0.

        Returns:
            tuple: (output tensor, quantization parameters) after dequant-SwiGLU-quant.
        """
        return torch_npu.npu_dequant_swiglu_quant(x, weight_scale=weight_scale, activation_scale=activation_scale,
                                                   bias=bias, quant_scale=quant_scale, quant_offset=quant_offset,
                                                   group_index=group_index, activate_left=activate_left,
                                                   quant_mode=quant_mode, swiglu_mode=swiglu_mode,
                                                   clamp_limit=clamp_limit, glu_alpha=glu_alpha, glu_bias=glu_bias)
