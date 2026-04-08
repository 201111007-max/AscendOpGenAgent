import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs quantized scatter operation.
    torch_npu.npu_quant_scatter(input, indices, updates, quant_scales, quant_zero_points=None, axis=0, quant_axis=1, reduce='update') -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    # PyTorch native implementation of forward function
    # def forward(self, input: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor,
    #             quant_scales: torch.Tensor, quant_zero_points: torch.Tensor = None,
    #             axis: int = 0, quant_axis: int = 1, reduce: str = 'update') -> torch.Tensor:
    #     if axis < 0:
    #         axis = updates.ndim + axis
    #     if quant_axis < 0:
    #         quant_axis = updates.ndim + quant_axis
    # 
    #     quant_scales_expanded = quant_scales
    #     while quant_scales_expanded.ndim < updates.ndim:
    #         quant_scales_expanded = quant_scales_expanded.unsqueeze(0)
    # 
    #     quantized_updates = torch.round(updates / quant_scales_expanded).to(torch.int8)
    # 
    #     output = input.clone()
    # 
    #     indices_int64 = indices.to(torch.int64)
    # 
    #     for i, idx in enumerate(indices_int64):
    #         idx_val = idx.item()
    #         slices = [slice(None)] * output.ndim
    #         slices[axis] = idx_val
    # 
    #         update_slices = [slice(None)] * quantized_updates.ndim
    #         update_slices[0] = i
    # 
    #         output[tuple(slices)] = quantized_updates[tuple(update_slices)]
    # 
    #     return output

    def forward(self, input: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor,
                quant_scales: torch.Tensor, quant_zero_points: torch.Tensor = None,
                axis: int = 0, quant_axis: int = 1, reduce: str = 'update') -> torch.Tensor:
        """
        Performs quantized scatter operation.

        Args:
            input (torch.Tensor): Source data tensor. dtype: int8, format: ND.
                                  Supports non-contiguous tensors. Must be 3-8D.
            indices (torch.Tensor): Index tensor. dtype: int32, format: ND.
                                    Supports non-contiguous tensors.
            updates (torch.Tensor): Update data tensor. format: ND, supports non-contiguous.
                                    dtype: bfloat16, float16.
            quant_scales (torch.Tensor): Quantization scale tensor. format: ND, supports non-contiguous.
                                         dtype: bfloat16, float32.
            quant_zero_points (torch.Tensor, optional): Quantization offset tensor. format: ND.
                                                        dtype: bfloat16, int32.
            axis (int, optional): Axis on updates for updating. Default: 0.
            quant_axis (int, optional): Axis on updates for quantization. Default: 1.
            reduce (str, optional): Data operation mode. Currently only supports 'update'. Default: 'update'.

        Returns:
            torch.Tensor: Output tensor after quantized scatter operation.
        """
        return torch_npu.npu_quant_scatter(input, indices, updates, quant_scales,
                                           quant_zero_points=quant_zero_points, axis=axis,
                                           quant_axis=quant_axis, reduce=reduce)
