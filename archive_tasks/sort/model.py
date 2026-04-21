import torch
import torch.nn as nn


class Model(nn.Module):
    """Reference key-value sort: sort by keys ascending, reorder values accordingly."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, keys: torch.Tensor, values: torch.Tensor):
        sorted_indices = torch.argsort(keys, stable=True)
        return keys[sorted_indices], values[sorted_indices]


KV_SORT_CASES = [
    {"n": 1, "key_range": 10, "seed": 1001},
    {"n": 7, "key_range": 5, "seed": 1002},
    {"n": 16, "key_range": 10, "seed": 1003},
    {"n": 33, "key_range": 20, "seed": 1004},
    {"n": 64, "key_range": 15, "seed": 1005},
    {"n": 100, "key_range": 30, "seed": 1006},
    {"n": 128, "key_range": 20, "seed": 1007},
    {"n": 255, "key_range": 50, "seed": 1008},
    {"n": 512, "key_range": 50, "seed": 1009},
    {"n": 999, "key_range": 80, "seed": 1010},
    {"n": 1024, "key_range": 60, "seed": 1011},
    {"n": 2048, "key_range": 100, "seed": 1012},
    {"n": 3000, "key_range": 100, "seed": 1013},
    {"n": 4097, "key_range": 100, "seed": 1014},
    {"n": 7777, "key_range": 200, "seed": 1015},
    {"n": 8192, "key_range": 150, "seed": 1016},
    {"n": 12345, "key_range": 200, "seed": 1017},
    {"n": 16384, "key_range": 200, "seed": 1018},
    {"n": 32768, "key_range": 300, "seed": 1019},
    {"n": 50000, "key_range": 500, "seed": 1020},
]


def get_input_groups():
    input_groups = []
    for case in KV_SORT_CASES:
        g = torch.Generator().manual_seed(case["seed"])
        n = case["n"]
        keys = torch.randint(0, case["key_range"], (n,), dtype=torch.int32, generator=g)
        values = torch.arange(n, dtype=torch.int32)
        input_groups.append([keys, values])
    return input_groups


def get_init_inputs():
    return []
