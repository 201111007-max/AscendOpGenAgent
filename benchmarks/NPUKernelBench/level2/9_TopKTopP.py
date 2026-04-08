import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs top-k and top-p filtering.
    torch_npu.npu_top_k_top_p(logits, p, k) -> torch.Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    # PyTorch native implementation of forward function
    # def forward(self, logits: torch.Tensor, p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    #     batch_size, vocab_size = logits.shape
    #     device = logits.device
    #     dtype = logits.dtype
    # 
    #     sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=False)
    # 
    #     k_int = k.to(torch.int64)
    #     k_int = torch.clamp(k_int, 1, vocab_size)
    # 
    #     gather_indices = (vocab_size - k_int).unsqueeze(1)
    #     gather_indices = torch.clamp(gather_indices, 0, vocab_size - 1)
    #     top_k_values = torch.gather(sorted_logits, 1, gather_indices)
    # 
    #     top_k_mask = sorted_logits < top_k_values
    # 
    #     sorted_logits_filtered = sorted_logits.clone()
    #     sorted_logits_filtered[top_k_mask] = float('-inf')
    # 
    #     probs = F.softmax(sorted_logits_filtered, dim=-1)
    # 
    #     cumsum_probs = torch.cumsum(probs, dim=-1)
    # 
    #     p_expanded = p.unsqueeze(1)
    #     top_p_mask = cumsum_probs <= (1.0 - p_expanded)
    # 
    #     top_p_mask[:, -1] = False
    # 
    #     sorted_logits_filtered[top_p_mask] = float('-inf')
    # 
    #     output = torch.empty_like(logits)
    #     for i in range(batch_size):
    #         output[i, sorted_indices[i]] = sorted_logits_filtered[i]
    # 
    #     return output

    def forward(self, logits: torch.Tensor, p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Performs top-k and top-p filtering on logits.

        Args:
            logits (torch.Tensor): Data to process. Must be 2D.
                                   dtype: float32, float16, bfloat16, format: ND.
                                   Supports non-contiguous tensors.
            p (torch.Tensor): Top-p threshold tensor. Range: [0, 1].
                              dtype: float32, float16, bfloat16 (must match logits).
                              Must be 1D with size matching logits' first dimension.
                              format: ND, supports non-contiguous tensors.
            k (torch.Tensor): Top-k threshold tensor. Range: [1, 1024], max <= logits.size(1).
                              dtype: int32. Must be 1D with size matching logits' first dimension.
                              format: ND, supports non-contiguous tensors.

        Returns:
            torch.Tensor: Filtered logits tensor.
        """
        return torch_npu.npu_top_k_top_p(logits, p, k)
