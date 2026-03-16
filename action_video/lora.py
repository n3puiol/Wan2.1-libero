"""
Minimal LoRA (Low-Rank Adaptation) implementation for nn.Linear layers.
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a low-rank additive update."""

    def __init__(self, original: nn.Linear, rank: int = 16, alpha: int = 32):
        super().__init__()
        self.original = original
        self.original.requires_grad_(False)

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        lora = (x @ self.lora_A @ self.lora_B) * self.scale
        return base + lora


def apply_lora_to_blocks(blocks: nn.ModuleList, rank: int = 16, alpha: int = 32):
    """Apply LoRA to q, k, v, o projections in self_attn and cross_attn of each block."""
    for block in blocks:
        for attn_name in ["self_attn", "cross_attn"]:
            attn = getattr(block, attn_name, None)
            if attn is None:
                continue
            for proj_name in ["q", "k", "v", "o"]:
                original = getattr(attn, proj_name, None)
                if original is not None and isinstance(original, nn.Linear):
                    setattr(attn, proj_name, LoRALinear(original, rank, alpha))


def get_lora_parameters(model: nn.Module) -> list:
    """Collect all LoRA parameters from a model."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A, module.lora_B])
    return params
