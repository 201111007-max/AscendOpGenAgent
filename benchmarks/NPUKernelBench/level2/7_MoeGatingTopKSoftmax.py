import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs gating top-k softmax for MoE (Mixture of Experts).
    torch_npu.npu_moe_gating_top_k_softmax(x, finished=None, k=1) -> (Tensor, Tensor, Tensor)
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
