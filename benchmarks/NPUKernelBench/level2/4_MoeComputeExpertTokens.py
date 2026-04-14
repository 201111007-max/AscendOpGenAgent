import torch
import torch.nn as nn
import torch_npu
import json
import os

class Model(nn.Module):
    """
    Simple model that computes expert tokens for MoE (Mixture of Experts).
    torch_npu.npu_moe_compute_expert_tokens(sorted_expert_for_source_row, num_expert) -> Tensor
    Pure PyTorch implementation (replacing torch_npu.npu_moe_compute_expert_tokens):

    def forward(self, sorted_expert_for_source_row, num_expert):
        # sorted_expert_for_source_row: [N] int32 tensor with expert indices
        # num_expert: total number of experts

        # Count tokens per expert using bincount
        expert_tokens = torch.bincount(
            sorted_expert_for_source_row.long(),
            minlength=num_expert
        ).to(torch.int32)

        return expert_tokens
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


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "4_MoeComputeExpertTokens.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        sorted_expert_info = inputs[0]
        num_expert_info = inputs[1]
        
        sorted_expert_for_source_row = torch.randint(0, num_expert_info["value"], sorted_expert_info["shape"], dtype=torch.int32)
        num_expert = num_expert_info["value"]
        input_groups.append([sorted_expert_for_source_row, num_expert])
    return input_groups


def get_init_inputs():
    return []
