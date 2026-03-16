"""
Action tokenizer for variable-dimension action vectors.

Adapts the ScalarTokenizer pattern: each action dimension is independently
embedded via a shared MLP + learned dimension positional embedding, then
mean-pooled and projected to the model's cross-attention dimension.
"""

import torch
import torch.nn as nn


class ActionTokenizer(nn.Module):

    def __init__(
        self,
        model_dim: int = 1536,
        hidden_dim: int = 256,
        max_action_dim: int = 32,
        max_action_len: int = 16,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.max_action_dim = max_action_dim

        # Shared scalar-to-hidden projection (like ScalarTokenizer)
        self.scalar_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learned dimension positional encoding
        self.dim_pos_embed = nn.Embedding(max_action_dim, hidden_dim)

        # Project pooled hidden to model dim
        self.proj = nn.Linear(hidden_dim, model_dim)

        # Temporal positional embedding for action timesteps
        self.temporal_pos_embed = nn.Embedding(max_action_len, model_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: [B, T, action_dim]

        Returns:
            [B, T, model_dim] action embeddings for cross-attention concatenation
        """
        B, T, D = actions.shape

        if D > self.max_action_dim:
            raise ValueError(
                f"Action dim {D} exceeds max_action_dim {self.max_action_dim}"
            )

        # Per-dimension embedding: [B, T, D, 1] -> [B, T, D, hidden_dim]
        tokens = self.scalar_embed(actions.unsqueeze(-1))

        # Add dimension positional encoding
        dim_indices = torch.arange(D, device=actions.device)
        tokens = tokens + self.dim_pos_embed(dim_indices)

        # Mean pool across action dimensions -> [B, T, hidden_dim]
        tokens = tokens.mean(dim=2)

        # Project to model dim -> [B, T, model_dim]
        tokens = self.proj(tokens)

        # Add temporal positional encoding
        t_indices = torch.arange(T, device=actions.device)
        tokens = tokens + self.temporal_pos_embed(t_indices)

        return tokens
