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

        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        lora = (x @ self.lora_A @ self.lora_B) * self.scale
        return base + lora


def apply_lora_to_blocks(
    blocks: nn.ModuleList,
    rank: int = 16,
    alpha: int = 32,
    lora_ffn: bool = False,
):
    """Apply LoRA to attention projections and optionally FFN layers in each block."""
    for block in blocks:
        for attn_name in ["self_attn", "cross_attn"]:
            attn = getattr(block, attn_name, None)
            if attn is None:
                continue
            for proj_name in ["q", "k", "v", "o"]:
                original = getattr(attn, proj_name, None)
                if original is not None and isinstance(original, nn.Linear):
                    setattr(attn, proj_name, LoRALinear(original, rank, alpha))

        if lora_ffn:
            ffn = getattr(block, "ffn", None)
            if ffn is not None and isinstance(ffn, nn.Sequential):
                for i, layer in enumerate(ffn):
                    if isinstance(layer, nn.Linear):
                        ffn[i] = LoRALinear(layer, rank, alpha)


def get_lora_parameters(model: nn.Module) -> list:
    """Collect all LoRA parameters from a model."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A, module.lora_B])
    return params
