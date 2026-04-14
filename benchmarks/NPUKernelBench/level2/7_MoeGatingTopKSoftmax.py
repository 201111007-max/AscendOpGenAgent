import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs gating top-k softmax for MoE (Mixture of Experts).
    torch_npu.npu_moe_gating_top_k_softmax(x, finished=None, k=1) -> (Tensor, Tensor, Tensor)

    Pure PyTorch implementation (replacing torch_npu.npu_moe_gating_top_k_softmax):

        def forward(self, x, finished=None, k=1):
            # x: [..., NUM_EXPERTS] input logits
            # finished: [...] bool tensor indicating finished positions (optional)
            # k: number of top experts to select

            original_shape = x.shape
            num_experts = original_shape[-1]

            # Flatten all but last dimension
            x_flat = x.view(-1, num_experts)  # [B, NUM_EXPERTS]

            # Apply finished mask if provided
            if finished is not None:
                finished_flat = finished.view(-1)
                # Mask finished positions with very negative values
                x_flat = torch.where(
                    finished_flat.unsqueeze(-1),
                    torch.tensor(float('-inf'), device=x.device, dtype=x.dtype),
                    x_flat
                )

            # Compute softmax over all experts
            softmax_output = torch.softmax(x_flat, dim=-1)  # [B, NUM_EXPERTS]

            # Get top-k experts and their indices
            topk_values, topk_indices = torch.topk(softmax_output, k, dim=-1)  # [B, k]

            # Reshape outputs back to original batch dimensions
            output_shape = original_shape[:-1]
            softmax_output = softmax_output.view(*output_shape, num_experts)
            topk_indices = topk_indices.view(*output_shape, k)
            topk_values = topk_values.view(*output_shape, k)

            return softmax_output, topk_indices, topk_values
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, finished: torch.Tensor = None, k: int = 1) -> tuple:
        """
        Performs gating top-k softmax for MoE.

        Args:
            x (torch.Tensor): Input tensor for computation. Must be 2D or 3D.
                              dtype: float16, bfloat16, float32, format: ND.
            finished (torch.Tensor, optional): Rows in input that need computation. Must be 2D or 3D.
                                               dtype: bool, shape: gating_shape[:-1], format: ND.
            k (int, optional): Top-k value. Range: 0 < k <= x.size(-1), k <= 1024. Default: 1.

        Returns:
            tuple: (output tensor, topk_indices, topk_weights) for MoE gating.
        """
        return torch_npu.npu_moe_gating_top_k_softmax(x, finished=finished, k=k)


def get_input_groups():
    """Generate input groups from JSON test cases."""
    json_path = os.path.join(os.path.dirname(__file__), os.path.splitext(os.path.basename(__file__))[0] + '.json')
    input_groups = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            inputs = case['inputs']
            tensors = {}
            for inp in inputs:
                if inp['type'] == 'tensor':
                    name = inp['name']
                    dtype_str = inp.get('dtype', 'float32')
                    shape = inp.get('shape')
                    if shape is None:
                        tensors[name] = None
                    elif dtype_str == 'bool':
                        tensors[name] = (torch.rand(shape) > 0.5).to(torch.bool)
                    elif dtype_str in ('int32', 'int64', 'int8'):
                        max_val = {'int32': 1000, 'int64': 10000, 'int8': 127}.get(dtype_str, 100)
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}[dtype_str]
                        tensors[name] = torch.randint(0, max_val, shape, dtype=dtype)
                    else:
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}.get(dtype_str, torch.float32)
                        tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    tensors[inp['name']] = inp['value']

            # Build input list in order matching forward signature
            group = []
            for inp in inputs:
                group.append(tensors[inp['name']])
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
