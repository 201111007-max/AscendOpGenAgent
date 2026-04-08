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

    # PyTorch native implementation of forward function
    # def forward(self, x: torch.Tensor, smooth_scales: torch.Tensor = None, offsets: torch.Tensor = None,
    #             group_index: torch.Tensor = None, activate_left: bool = False, quant_mode: int = 0,
    #             group_list_type: int = 0, dst_type = None) -> tuple:
    #     x_float = x.float()
    # 
    #     half_size = x_float.shape[-1] // 2
    #     x_left = x_float[..., :half_size]
    #     x_right = x_float[..., half_size:]
    # 
    #     if activate_left:
    #         activated = torch.sigmoid(x_left) * x_left
    #         output = activated * x_right
    #     else:
    #         activated = torch.sigmoid(x_right) * x_right
    #         output = activated * x_left
    # 
    #     if smooth_scales is not None and group_index is not None:
    #         if group_list_type == 0:
    #             group_boundaries = [0] + group_index.tolist()
    #         else:
    #             group_boundaries = [0]
    #             cumsum = 0
    #             for count in group_index.tolist():
    #                 cumsum += count
    #                 group_boundaries.append(cumsum)
    # 
    #         output_scaled = output.clone()
    #         for i in range(len(group_boundaries) - 1):
    #             start_idx = group_boundaries[i]
    #             end_idx = group_boundaries[i + 1]
    #             if start_idx < end_idx:
    #                 scale = smooth_scales[i]
    #                 output_scaled[start_idx:end_idx] = output[start_idx:end_idx] * scale
    # 
    #         output = output_scaled
    #     elif smooth_scales is not None:
    #         output = output * smooth_scales
    # 
    #     if quant_mode == 1:
    #         if group_index is not None:
    #             quant_scales_list = []
    #             quantized_output_list = []
    # 
    #             if group_list_type == 0:
    #                 group_boundaries = [0] + group_index.tolist()
    #             else:
    #                 group_boundaries = [0]
    #                 cumsum = 0
    #                 for count in group_index.tolist():
    #                     cumsum += count
    #                     group_boundaries.append(cumsum)
    # 
    #             for i in range(len(group_boundaries) - 1):
    #                 start_idx = group_boundaries[i]
    #                 end_idx = group_boundaries[i + 1]
    # 
    #                 if start_idx >= end_idx:
    #                     continue
    # 
    #                 group_data = output[start_idx:end_idx]
    # 
    #                 if group_data.numel() > 0:
    #                     max_val = torch.max(torch.abs(group_data))
    #                     if max_val == 0:
    #                         scale = torch.tensor(1.0, dtype=torch.float32, device=output.device)
    #                     else:
    #                         scale = max_val / 127.0
    #                     quant_scales_list.append(scale)
    # 
    #                     quantized_group = torch.round(group_data / scale).to(torch.int8)
    #                     quantized_output_list.append(quantized_group)
    # 
    #             if quantized_output_list:
    #                 quantized_output = torch.cat(quantized_output_list, dim=0)
    #                 quant_scales = torch.stack(quant_scales_list)
    #             else:
    #                 quantized_output = torch.zeros_like(output, dtype=torch.int8)
    #                 quant_scales = torch.ones(1, dtype=torch.float32, device=output.device)
    #         else:
    #             if output.numel() > 0:
    #                 max_val = torch.max(torch.abs(output))
    #                 if max_val == 0:
    #                     quant_scales = torch.tensor(1.0, dtype=torch.float32, device=output.device)
    #                 else:
    #                     quant_scales = max_val / 127.0
    #             else:
    #                 quant_scales = torch.tensor(1.0, dtype=torch.float32, device=output.device)
    #             quantized_output = torch.round(output / quant_scales).to(torch.int8)
    #     else:
    #         if smooth_scales is not None:
    #             quant_scales = smooth_scales
    #         else:
    #             quant_scales = torch.ones(1, dtype=torch.float32, device=output.device)
    # 
    #         quantized_output = torch.round(output / quant_scales).to(torch.int8)
    # 
    #     return quantized_output, quant_scales

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
