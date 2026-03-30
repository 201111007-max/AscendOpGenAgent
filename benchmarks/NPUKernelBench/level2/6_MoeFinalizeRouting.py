import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that finalizes routing for MoE (Mixture of Experts).
    torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row, export_for_source_row, drop_pad_mode=0) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, expanded_permuted_rows: torch.Tensor, skip1: torch.Tensor, skip2: torch.Tensor,
                bias: torch.Tensor, scales: torch.Tensor, expanded_src_to_dst_row: torch.Tensor,
                export_for_source_row: torch.Tensor, drop_pad_mode: int = 0) -> torch.Tensor:
        """
        Finalizes routing for MoE.

        Args:
            expanded_permuted_rows (torch.Tensor): Result processed by experts, must be 2D.
                                                   dtype: float16, bfloat16, float32, format: ND.
                                                   drop_pad_mode 0/2: shape (NUM_ROWS*K, H)
                                                   drop_pad_mode 1/3: shape (E, C, H)
            skip1 (torch.Tensor): Sum input param 1, can be None. Must be 2D, same dtype and shape as output.
            skip2 (torch.Tensor): Sum input param 2, can be None. Must be 2D, same dtype and shape as output.
                                  If skip1 is None, skip2 must also be None.
            bias (torch.Tensor): Expert bias, can be None. Must be 2D, same dtype as expanded_permuted_rows.
                                 Shape: (E, H).
            scales (torch.Tensor): Expert weights, can be None. Must be 2D, same dtype as expanded_permuted_rows.
                                   Shape: (NUM_ROWS, K).
            expanded_src_to_dst_row (torch.Tensor): Index of each expert's processing result. Must be 1D.
                                                    dtype: int32. Shape: (NUM_ROWS*K).
            export_for_source_row (torch.Tensor): Expert number for each row, can be None. Must be 2D.
                                                  dtype: int32. Shape: (NUM_ROWS, K). Range: [0, E-1].
            drop_pad_mode (int, optional): Drop mode and arrangement. Range: [0, 3]. Default: 0.
                                           0: non-drop mode, column arrangement
                                           1: drop mode, column arrangement
                                           2: non-drop mode, row arrangement
                                           3: drop mode, row arrangement

        Returns:
            torch.Tensor: Output tensor after finalizing MoE routing.
        """
        return torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2, bias, scales,
                                                   expanded_src_to_dst_row, export_for_source_row,
                                                   drop_pad_mode=drop_pad_mode)
