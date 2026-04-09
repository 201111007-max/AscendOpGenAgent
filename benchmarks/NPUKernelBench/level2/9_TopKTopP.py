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
    #     ori_dtype = logits.dtype
    #     p = p.to(torch.float32)

    #     logits_sort, logits_idx = logits.sort(dim=-1, descending=False, stable=True)
    #     kth_idx = logits_sort.size(1) - k.to(torch.long)
    #     kth_value = logits_sort.gather(1, kth_idx.unsqueeze(dim=1))
    #     top_k_mask = logits_sort < kth_value
    #     logits_sort.masked_fill_(top_k_mask, -float("inf"))

    #     softmax_res = logits_sort.to(torch.float32).softmax(dim=-1)
    #     cumsum_res = softmax_res.cumsum(dim=-1)
    #     top_p_mask = cumsum_res <= 1 - p.unsqueeze(dim=1)
    #     top_p_mask[:, -1] = False
    #     logits_sort.masked_fill_(top_p_mask, -float("inf"))

    #     logits = torch.empty_like(logits_sort).scatter_(dim=-1, index=logits_idx, src=logits_sort)

    #     return logits.to(ori_dtype)

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

INPUT_CASES = [{'inputs': [{'dtype': 'float16',
              'name': 'logits',
              'required': True,
              'shape': [16, 2048],
              'type': 'tensor'},
             {'dtype': 'float16', 'name': 'p', 'required': True, 'shape': [16], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [16], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'logits',
              'required': True,
              'shape': [32, 4096],
              'type': 'tensor'},
             {'dtype': 'float32', 'name': 'p', 'required': True, 'shape': [32], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [32], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'logits',
              'required': True,
              'shape': [64, 8192],
              'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'p', 'required': True, 'shape': [64], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [64], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'logits',
              'required': True,
              'shape': [8, 1024],
              'type': 'tensor'},
             {'dtype': 'float16', 'name': 'p', 'required': True, 'shape': [8], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [8], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'logits',
              'required': True,
              'shape': [128, 16384],
              'type': 'tensor'},
             {'dtype': 'float32', 'name': 'p', 'required': True, 'shape': [128], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [128], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'logits',
              'required': True,
              'shape': [4, 512],
              'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'p', 'required': True, 'shape': [4], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [4], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'logits',
              'required': True,
              'shape': [256, 32768],
              'type': 'tensor'},
             {'dtype': 'float16', 'name': 'p', 'required': True, 'shape': [256], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [256], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'logits',
              'required': True,
              'shape': [2, 256],
              'type': 'tensor'},
             {'dtype': 'float32', 'name': 'p', 'required': True, 'shape': [2], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [2], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'logits',
              'required': True,
              'shape': [512, 65536],
              'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'p', 'required': True, 'shape': [512], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [512], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'logits',
              'required': True,
              'shape': [24, 2048],
              'type': 'tensor'},
             {'dtype': 'float16', 'name': 'p', 'required': True, 'shape': [24], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [24], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'logits',
              'required': True,
              'shape': [48, 4096],
              'type': 'tensor'},
             {'dtype': 'float32', 'name': 'p', 'required': True, 'shape': [48], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [48], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'logits',
              'required': True,
              'shape': [96, 8192],
              'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'p', 'required': True, 'shape': [96], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [96], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'logits',
              'required': True,
              'shape': [12, 1024],
              'type': 'tensor'},
             {'dtype': 'float16', 'name': 'p', 'required': True, 'shape': [12], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [12], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'logits',
              'required': True,
              'shape': [192, 16384],
              'type': 'tensor'},
             {'dtype': 'float32', 'name': 'p', 'required': True, 'shape': [192], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [192], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'logits',
              'required': True,
              'shape': [6, 512],
              'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'p', 'required': True, 'shape': [6], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [6], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'logits',
              'required': True,
              'shape': [384, 32768],
              'type': 'tensor'},
             {'dtype': 'float16', 'name': 'p', 'required': True, 'shape': [384], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [384], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'logits',
              'required': True,
              'shape': [3, 256],
              'type': 'tensor'},
             {'dtype': 'float32', 'name': 'p', 'required': True, 'shape': [3], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [3], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'logits',
              'required': True,
              'shape': [768, 65536],
              'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'p', 'required': True, 'shape': [768], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [768], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'logits',
              'required': True,
              'shape': [20, 2048],
              'type': 'tensor'},
             {'dtype': 'float16', 'name': 'p', 'required': True, 'shape': [20], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [20], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'logits',
              'required': True,
              'shape': [40, 4096],
              'type': 'tensor'},
             {'dtype': 'float32', 'name': 'p', 'required': True, 'shape': [40], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [40], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'logits',
              'required': True,
              'shape': [80, 8192],
              'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'p', 'required': True, 'shape': [80], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [80], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'logits',
              'required': True,
              'shape': [10, 1024],
              'type': 'tensor'},
             {'dtype': 'float16', 'name': 'p', 'required': True, 'shape': [10], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [10], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'logits',
              'required': True,
              'shape': [160, 16384],
              'type': 'tensor'},
             {'dtype': 'float32', 'name': 'p', 'required': True, 'shape': [160], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [160], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'logits',
              'required': True,
              'shape': [5, 512],
              'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'p', 'required': True, 'shape': [5], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [5], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'logits',
              'required': True,
              'shape': [320, 32768],
              'type': 'tensor'},
             {'dtype': 'float16', 'name': 'p', 'required': True, 'shape': [320], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [320], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'logits',
              'required': True,
              'shape': [1, 256],
              'type': 'tensor'},
             {'dtype': 'float32', 'name': 'p', 'required': True, 'shape': [1], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [1], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'logits',
              'required': True,
              'shape': [640, 65536],
              'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'p', 'required': True, 'shape': [640], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [640], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'logits',
              'required': True,
              'shape': [28, 2048],
              'type': 'tensor'},
             {'dtype': 'float16', 'name': 'p', 'required': True, 'shape': [28], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [28], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'logits',
              'required': True,
              'shape': [56, 4096],
              'type': 'tensor'},
             {'dtype': 'float32', 'name': 'p', 'required': True, 'shape': [56], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [56], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'logits',
              'required': True,
              'shape': [112, 8192],
              'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'p', 'required': True, 'shape': [112], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [112], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'logits',
              'required': True,
              'shape': [14, 1024],
              'type': 'tensor'},
             {'dtype': 'float16', 'name': 'p', 'required': True, 'shape': [14], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [14], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'logits',
              'required': True,
              'shape': [224, 16384],
              'type': 'tensor'},
             {'dtype': 'float32', 'name': 'p', 'required': True, 'shape': [224], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [224], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'logits',
              'required': True,
              'shape': [7, 512],
              'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'p', 'required': True, 'shape': [7], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [7], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'logits',
              'required': True,
              'shape': [448, 32768],
              'type': 'tensor'},
             {'dtype': 'float16', 'name': 'p', 'required': True, 'shape': [448], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [448], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'logits',
              'required': True,
              'shape': [9, 256],
              'type': 'tensor'},
             {'dtype': 'float32', 'name': 'p', 'required': True, 'shape': [9], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [9], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'logits',
              'required': True,
              'shape': [896, 65536],
              'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'p', 'required': True, 'shape': [896], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [896], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'logits',
              'required': True,
              'shape': [36, 2048],
              'type': 'tensor'},
             {'dtype': 'float16', 'name': 'p', 'required': True, 'shape': [36], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [36], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'logits',
              'required': True,
              'shape': [72, 4096],
              'type': 'tensor'},
             {'dtype': 'float32', 'name': 'p', 'required': True, 'shape': [72], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [72], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'logits',
              'required': True,
              'shape': [144, 8192],
              'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'p', 'required': True, 'shape': [144], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [144], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'logits',
              'required': True,
              'shape': [18, 1024],
              'type': 'tensor'},
             {'dtype': 'float16', 'name': 'p', 'required': True, 'shape': [18], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [18], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'logits',
              'required': True,
              'shape': [288, 16384],
              'type': 'tensor'},
             {'dtype': 'float32', 'name': 'p', 'required': True, 'shape': [288], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [288], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'logits',
              'required': True,
              'shape': [11, 512],
              'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'p', 'required': True, 'shape': [11], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [11], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'logits',
              'required': True,
              'shape': [576, 32768],
              'type': 'tensor'},
             {'dtype': 'float16', 'name': 'p', 'required': True, 'shape': [576], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [576], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'logits',
              'required': True,
              'shape': [13, 256],
              'type': 'tensor'},
             {'dtype': 'float32', 'name': 'p', 'required': True, 'shape': [13], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [13], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'logits',
              'required': True,
              'shape': [1152, 65536],
              'type': 'tensor'},
             {'dtype': 'bfloat16',
              'name': 'p',
              'required': True,
              'shape': [1152],
              'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [1152], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'logits',
              'required': True,
              'shape': [44, 2048],
              'type': 'tensor'},
             {'dtype': 'float16', 'name': 'p', 'required': True, 'shape': [44], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [44], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'logits',
              'required': True,
              'shape': [88, 4096],
              'type': 'tensor'},
             {'dtype': 'float32', 'name': 'p', 'required': True, 'shape': [88], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [88], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'logits',
              'required': True,
              'shape': [176, 8192],
              'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'p', 'required': True, 'shape': [176], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [176], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'logits',
              'required': True,
              'shape': [22, 1024],
              'type': 'tensor'},
             {'dtype': 'float16', 'name': 'p', 'required': True, 'shape': [22], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [22], 'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'logits',
              'required': True,
              'shape': [352, 16384],
              'type': 'tensor'},
             {'dtype': 'float32', 'name': 'p', 'required': True, 'shape': [352], 'type': 'tensor'},
             {'dtype': 'int32', 'name': 'k', 'required': True, 'shape': [352], 'type': 'tensor'}]}]

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _make_boxes(shape, dtype):
    leading_shape = tuple(shape[:-1])
    mins = torch.rand(*leading_shape, 2, dtype=torch.float32)
    sizes = torch.rand(*leading_shape, 2, dtype=torch.float32) + 0.05
    maxs = mins + sizes
    boxes = torch.cat([mins, maxs], dim=-1)
    return boxes.to(dtype=dtype)


def _make_tensor(spec):
    dtype = _DTYPE_MAP[spec["dtype"]]
    shape = spec["shape"]
    name = spec["name"]
    value_range = spec.get("range")

    if dtype == torch.bool:
        return torch.randint(0, 2, tuple(shape), dtype=torch.int64).to(torch.bool)

    if name in {"boxes", "bboxes", "gtboxes"} and shape and shape[-1] == 4 and dtype in {
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    }:
        return _make_boxes(shape, dtype)

    if value_range is not None:
        low, high = value_range
        if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            high_exclusive = high + 1
            return torch.randint(low, high_exclusive, tuple(shape), dtype=dtype)
        return torch.empty(tuple(shape), dtype=dtype).uniform_(low, high)

    if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        return torch.randint(0, 17, tuple(shape), dtype=dtype)

    return torch.randn(*shape, dtype=dtype)


def _make_tensor_list(spec):
    dtype = _DTYPE_MAP[spec["dtype"]]
    return [torch.randn(*shape, dtype=dtype) for shape in spec["shapes"]]


def _make_arg(spec):
    spec_type = spec["type"]
    if spec_type == "tensor":
        return _make_tensor(spec)
    if spec_type == "tensor_list":
        return _make_tensor_list(spec)
    if spec_type == "attr":
        return spec["value"]
    raise ValueError(f"Unsupported input spec type: {spec_type}")


def get_input_groups():
    input_groups = []
    for case in INPUT_CASES:
        input_groups.append([_make_arg(spec) for spec in case["inputs"]])
    return input_groups


def get_init_inputs():
    return []
