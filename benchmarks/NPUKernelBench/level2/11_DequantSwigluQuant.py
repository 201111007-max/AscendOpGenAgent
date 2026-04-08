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

    # PyTorch native implementation of forward function
    # def forward(self, x: torch.Tensor, weight_scale: torch.Tensor = None, activation_scale: torch.Tensor = None,
    #             bias: torch.Tensor = None, quant_scale: torch.Tensor = None, quant_offset: torch.Tensor = None,
    #             group_index: torch.Tensor = None, activate_left: bool = False, quant_mode: int = 0,
    #             swiglu_mode: int = 0, clamp_limit: float = 7.0, glu_alpha: float = 1.702,
    #             glu_bias: float = 1.0) -> tuple:
    #     x_float = x.float()
    # 
    #     if weight_scale is not None:
    #         x_float = x_float * weight_scale
    # 
    #     if activation_scale is not None:
    #         x_float = x_float * activation_scale
    # 
    #     if bias is not None:
    #         x_float = x_float + bias.float()
    # 
    #     half_size = x_float.shape[-1] // 2
    #     x_left = x_float[..., :half_size]
    #     x_right = x_float[..., half_size:]
    # 
    #     if swiglu_mode == 0:
    #         if activate_left:
    #             activated = torch.sigmoid(x_left) * x_left
    #             output = activated * x_right
    #         else:
    #             activated = torch.sigmoid(x_right) * x_right
    #             output = activated * x_left
    #     else:
    #         x_left_clamped = torch.clamp(x_left, -clamp_limit, clamp_limit)
    #         x_right_clamped = torch.clamp(x_right, -clamp_limit, clamp_limit)
    # 
    #         if activate_left:
    #             activated = (x_left_clamped + glu_bias) * torch.sigmoid(glu_alpha * (x_left_clamped + glu_bias))
    #             output = activated * x_right_clamped
    #         else:
    #             activated = (x_right_clamped + glu_bias) * torch.sigmoid(glu_alpha * (x_right_clamped + glu_bias))
    #             output = activated * x_left_clamped
    # 
    #     if quant_scale is not None:
    #         scales = quant_scale
    #         if scales.dim() == 1:
    #             scales = scales.unsqueeze(-1)
    #         output = output * scales
    # 
    #     if quant_mode == 1:
    #         if group_index is not None:
    #             quant_scales_list = []
    #             quantized_output_list = []
    # 
    #             group_boundaries = [0]
    #             cumsum = 0
    #             for count in group_index.tolist():
    #                 cumsum += count
    #                 group_boundaries.append(cumsum)
    # 
    #             for i in range(len(group_boundaries) - 1):
    #                 start_idx = group_boundaries[i]
    #                 end_idx = group_boundaries[i + 1]
    # 
    #                 group_data = output[start_idx:end_idx]
    # 
    #                 max_val = torch.max(torch.abs(group_data))
    #                 scale = max_val / 127.0
    #                 quant_scales_list.append(scale)
    # 
    #                 quantized_group = torch.round(group_data / scale).to(torch.int8)
    #                 quantized_output_list.append(quantized_group)
    # 
    #             quantized_output = torch.cat(quantized_output_list, dim=0)
    #             quant_scales = torch.stack(quant_scales_list)
    #         else:
    #             max_val = torch.max(torch.abs(output))
    #             quant_scales = max_val / 127.0
    #             quantized_output = torch.round(output / quant_scales).to(torch.int8)
    #     else:
    #         if quant_scale is not None:
    #             quant_scales = quant_scale
    #         else:
    #             quant_scales = torch.ones(1, dtype=torch.float32, device=output.device)
    # 
    #         quantized_output = torch.round(output / quant_scales).to(torch.int8)
    # 
    #     return quantized_output, quant_scales

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
