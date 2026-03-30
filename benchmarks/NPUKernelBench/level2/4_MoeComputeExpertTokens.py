import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes expert tokens for MoE (Mixture of Experts).
    torch_npu.npu_moe_compute_expert_tokens(sorted_expert_for_source_row, num_expert) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, sorted_expert_for_source_row: torch.Tensor, num_expert: int) -> torch.Tensor:
        """
        Computes expert tokens for MoE routing.

        Args:
            sorted_expert_for_source_row (torch.Tensor): Result processed by experts, must be 1D.
                                                         dtype: int32, format: ND. Shape must be < 2147483647.
            num_expert (int): Total number of experts.

        Returns:
            torch.Tensor: Computed expert tokens tensor.
        """
        return torch_npu.npu_moe_compute_expert_tokens(sorted_expert_for_source_row, num_expert)
