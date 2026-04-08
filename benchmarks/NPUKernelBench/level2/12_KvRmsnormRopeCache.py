import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs KV RMSNorm and RoPE with cache operations.
    torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index, k_cache, ckv_cache, *, k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None, epsilon=1e-5, cache_mode='Norm', is_output_kv=False) -> (Tensor, Tensor, Tensor, Tensor)
    """
    def __init__(self):
        super(Model, self).__init__()

    # PyTorch native implementation of forward function
    # def forward(self, kv: torch.Tensor, gamma: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    #             index: torch.Tensor, k_cache: torch.Tensor, ckv_cache: torch.Tensor,
    #             k_rope_scale: torch.Tensor = None, c_kv_scale: torch.Tensor = None,
    #             k_rope_offset: torch.Tensor = None, c_kv_offset: torch.Tensor = None,
    #             epsilon: float = 1e-5, cache_mode: str = 'Norm', is_output_kv: bool = False) -> tuple:
    #     B, _, S, hidden_size = kv.shape
    #     rms_size = gamma.shape[0]
    #     rope_size = hidden_size - rms_size
    # 
    #     rms_in = kv[..., :rms_size]
    #     rope_in = kv[..., rms_size:]
    # 
    #     rms_mean = torch.mean(rms_in ** 2, dim=-1, keepdim=True)
    #     rms_normalized = rms_in / torch.sqrt(rms_mean + epsilon)
    #     y = gamma * rms_normalized
    # 
    #     rope_reshaped = rope_in.reshape(B, 1, S, rope_size // 2, 2)
    # 
    #     rope_transposed = rope_reshaped.transpose(-1, -2)
    # 
    #     rope_interleaved = rope_transposed.reshape(B, 1, S, rope_size)
    # 
    #     cos_expanded = cos
    #     sin_expanded = sin
    # 
    #     if cos_expanded.shape[2] == 1 and S > 1:
    #         cos_expanded = cos_expanded.expand(B, 1, S, rope_size)
    #     if sin_expanded.shape[2] == 1 and S > 1:
    #         sin_expanded = sin_expanded.expand(B, 1, S, rope_size)
    # 
    #     rope_rotated = torch.zeros_like(rope_interleaved)
    #     rope_rotated[..., :rope_size // 2] = -rope_interleaved[..., rope_size // 2:]
    #     rope_rotated[..., rope_size // 2:] = rope_interleaved[..., :rope_size // 2]
    # 
    #     k_embed = rope_interleaved * cos_expanded + rope_rotated * sin_expanded
    # 
    #     k_cache_out = k_cache.clone()
    #     ckv_cache_out = ckv_cache.clone()
    # 
    #     if cache_mode == 'Norm':
    #         if index.numel() > 0:
    #             if index.dim() == 1:
    #                 for i, idx in enumerate(index):
    #                     idx_val = idx.item()
    #                     if idx_val < k_cache_out.shape[0]:
    #                         if i < B * S:
    #                             b_idx = i // S
    #                             s_idx = i % S
    #                             if b_idx < B and s_idx < S:
    #                                 k_cache_out[idx_val] = k_embed[b_idx, 0, s_idx]
    #                                 ckv_cache_out[idx_val] = y[b_idx, 0, s_idx]
    #             elif index.dim() == 2:
    #                 for b in range(min(B, index.shape[0])):
    #                     for s in range(min(S, index.shape[1])):
    #                         idx = index[b, s].item()
    #                         if idx < k_cache_out.shape[0]:
    #                             k_cache_out[idx] = k_embed[b, 0, s]
    #                             ckv_cache_out[idx] = y[b, 0, s]
    # 
    #     if is_output_kv:
    #         k_embed_out = k_embed
    #         y_out = y
    #     else:
    #         k_embed_out = None
    #         y_out = None
    # 
    #     return k_cache_out, ckv_cache_out, k_embed_out, y_out

    def forward(self, kv: torch.Tensor, gamma: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                index: torch.Tensor, k_cache: torch.Tensor, ckv_cache: torch.Tensor,
                k_rope_scale: torch.Tensor = None, c_kv_scale: torch.Tensor = None,
                k_rope_offset: torch.Tensor = None, c_kv_offset: torch.Tensor = None,
                epsilon: float = 1e-5, cache_mode: str = 'Norm', is_output_kv: bool = False) -> tuple:
        """
        Performs KV RMSNorm and RoPE with cache operations.

        Args:
            kv (torch.Tensor): Input feature tensor. Must be 4D [batch_size, 1, seq_len, hidden_size].
                               hidden_size = rms_size + rope_size.
                               dtype: bfloat16, float16, format: BNSD.
            gamma (torch.Tensor): RMS normalization scale parameter. Must be 1D [rms_size].
                                  dtype: bfloat16, float16, format: ND.
            cos (torch.Tensor): RoPE cosine component. Must be 4D [batch_size, 1, seq_len, rope_size].
                                dtype: bfloat16, float16, format: ND.
            sin (torch.Tensor): RoPE sine component. Must be 4D [batch_size, 1, seq_len, rope_size].
                                dtype: bfloat16, float16, format: ND.
            index (torch.Tensor): Cache index tensor for locating write positions in caches.
                                  dtype: int64, format: ND. Shape depends on cache_mode.
            k_cache (torch.Tensor): Storage for quantized/non-quantized key vectors.
                                    dtype: bfloat16, float16, int8, format: ND. Shape depends on cache_mode.
            ckv_cache (torch.Tensor): Storage for quantized/non-quantized compressed KV vectors.
                                      dtype: bfloat16, float16, int8, format: ND. Shape depends on cache_mode.
            k_rope_scale (torch.Tensor, optional): K RoPE quantization scale. Must be 1D [rope_size].
                                                   dtype: float32, format: ND. Required in quantization mode.
            c_kv_scale (torch.Tensor, optional): Compressed KV quantization scale. Must be 1D [rms_size].
                                                 dtype: float32, format: ND. Required in quantization mode.
            k_rope_offset (torch.Tensor, optional): K RoPE quantization offset. Must be 1D [rope_size].
                                                    dtype: float32, format: ND. Required in quantization mode.
            c_kv_offset (torch.Tensor, optional): Compressed KV quantization offset. Must be 1D [rms_size].
                                                  dtype: float32, format: ND. Required in quantization mode.
            epsilon (float, optional): Small value for RMS normalization to prevent division by zero.
                                       Default: 1e-5.
            cache_mode (str, optional): Cache mode. Options: 'Norm', 'PA', 'PA_BNSD', 'PA_NZ', 
                                        'PA_BLK_BNSD', 'PA_BLK_NZ'. Default: 'Norm'.
            is_output_kv (bool, optional): Whether to output processed k_embed_out and y_out.
                                           Default: False. Only effective in PA modes.

        Returns:
            tuple: (k_cache, ckv_cache, k_embed_out, y_out) tensors. Last two are None if is_output_kv=False.
        """
        return torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                                    k_rope_scale=k_rope_scale, c_kv_scale=c_kv_scale,
                                                    k_rope_offset=k_rope_offset, c_kv_offset=c_kv_offset,
                                                    epsilon=epsilon, cache_mode=cache_mode,
                                                    is_output_kv=is_output_kv)
