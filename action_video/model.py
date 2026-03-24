"""
ActionConditionedVace: wraps VaceWanModel with action conditioning.

Actions are tokenized and concatenated to the text cross-attention context,
reusing the existing cross-attention layers without architecture changes.
"""

import math
import os

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F

import importlib.util
import sys
from pathlib import Path

# Import wan submodules directly from file paths to avoid wan/__init__.py
# which eagerly imports torchvision (may be broken in some environments).
_WAN_DIR = Path(__file__).resolve().parent.parent / "wan"


def _load_module(name: str, filepath: Path):
    """Load a Python module from file without triggering package __init__.py."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Ensure wan.modules namespace exists without running wan/__init__.py
for ns in ["wan", "wan.modules"]:
    if ns not in sys.modules:
        import types
        pkg = types.ModuleType(ns)
        pkg.__path__ = [str(_WAN_DIR if ns == "wan" else _WAN_DIR / "modules")]
        pkg.__package__ = ns
        sys.modules[ns] = pkg

# Load wan.modules.attention first (dependency of model.py)
_attn_mod = _load_module("wan.modules.attention", _WAN_DIR / "modules" / "attention.py")

# On pre-Ampere GPUs (e.g. V100), FlashAttention is not supported.
# Monkey-patch flash_attention -> attention (SDPA fallback) so the model works.
if torch.cuda.is_available():
    _cc = torch.cuda.get_device_capability()
    if _cc[0] < 8:
        import logging as _logging
        _logging.getLogger(__name__).info(
            f"GPU compute capability {_cc[0]}.{_cc[1]} < 8.0: "
            "using PyTorch SDPA fallback instead of FlashAttention"
        )
        _attn_mod.flash_attention = _attn_mod.attention

_wan_model = _load_module("wan.modules.model", _WAN_DIR / "modules" / "model.py")
_wan_vace_model = _load_module("wan.modules.vace_model", _WAN_DIR / "modules" / "vace_model.py")
_wan_vae = _load_module("wan.modules.vae", _WAN_DIR / "modules" / "vae.py")
_wan_t5 = _load_module("wan.modules.t5", _WAN_DIR / "modules" / "t5.py")

sinusoidal_embedding_1d = _wan_model.sinusoidal_embedding_1d
VaceWanModel = _wan_vace_model.VaceWanModel
WanVAE = _wan_vae.WanVAE
T5EncoderModel = _wan_t5.T5EncoderModel

from .action_tokenizer import ActionTokenizer
from .config import ActionVideoConfig
from .lora import apply_lora_to_blocks, get_lora_parameters


class ActionConditionedVace(nn.Module):

    def __init__(self, config: ActionVideoConfig, device: str | torch.device = "cpu", device_id: int | None = None):
        super().__init__()
        self.config = config
        # Backward compat: device_id=0 -> device="cuda:0"
        if device_id is not None:
            device = f"cuda:{device_id}"
        self._init_device = torch.device(device)

        checkpoint_dir = config.checkpoint_dir

        # Load frozen VAE (not an nn.Module submodule — invisible to DDP)
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self._init_device,
        )

        # Load frozen T5 text encoder (not an nn.Module submodule — invisible to DDP)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
        )

        # Load VaceWanModel from pretrained
        self.vace_model = VaceWanModel.from_pretrained(checkpoint_dir)
        self.vace_model.eval().requires_grad_(False)

        # Swap inactive/reactive weight rows in vace_patch_embedding so the
        # pre-trained content-processing weights (channels 0-15, originally
        # inactive) align with the reactive channel where history data now
        # lives, and vice-versa.
        with torch.no_grad():
            w = self.vace_model.vace_patch_embedding.weight  # [out, 96, k_t, k_h, k_w]
            inactive_w = w[:, :16].clone()
            reactive_w = w[:, 16:32].clone()
            w[:, :16] = reactive_w
            w[:, 16:32] = inactive_w

        # Action conditioning modules (trainable)
        self.action_tokenizer = ActionTokenizer(
            model_dim=config.dim,
            hidden_dim=config.action_hidden_dim,
            max_action_dim=config.max_action_dim,
            max_action_len=config.max_action_len,
        )

        # Adapter: zero-initialized so action signal starts as no-op
        self.action_adapter = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim),
        )
        nn.init.zeros_(self.action_adapter[-1].weight)
        nn.init.zeros_(self.action_adapter[-1].bias)

        # Apply LoRA to main blocks
        if config.use_lora:
            apply_lora_to_blocks(
                self.vace_model.blocks,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                lora_ffn=config.lora_ffn,
            )

        # Unfreeze VACE blocks if requested
        if config.train_vace_blocks:
            self.vace_model.vace_blocks.requires_grad_(True)
            self.vace_model.vace_patch_embedding.requires_grad_(True)

        # When device != "cpu", move everything immediately (single-GPU / eval path)
        if self._init_device.type != "cpu":
            self.vace_model.to(self._init_device)
            self.action_tokenizer.to(self._init_device)
            self.action_adapter.to(self._init_device)

        # Store stride/patch for VACE encoding
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size

    @property
    def device(self) -> torch.device:
        """Device of trainable parameters (valid after init or accelerator.prepare)."""
        return self.action_adapter[-1].weight.device

    @property
    def current_device(self) -> torch.device:
        """Alias for device."""
        return self.device

    def setup_frozen_components(self, device: torch.device):
        """Move frozen VAE and T5 to target device. Call after accelerator.prepare()."""
        self.vae.model.to(device)
        self.vae.mean = self.vae.mean.to(device)
        self.vae.std = self.vae.std.to(device)
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]
        self.vae.device = device
        if not self.config.t5_cpu:
            self.text_encoder.model.to(device)

    def get_trainable_parameters(self) -> list:
        """Get all parameters that should be trained."""
        params = []
        # Action tokenizer + adapter
        params.extend(self.action_tokenizer.parameters())
        params.extend(self.action_adapter.parameters())
        # LoRA parameters
        params.extend(get_lora_parameters(self.vace_model))
        # VACE blocks (if unfrozen)
        if self.config.train_vace_blocks:
            for p in self.vace_model.vace_blocks.parameters():
                if p.requires_grad:
                    params.append(p)
            for p in self.vace_model.vace_patch_embedding.parameters():
                if p.requires_grad:
                    params.append(p)
        return params

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> list[torch.Tensor]:
        """Encode text prompts using T5 (on CPU if configured)."""
        device = self.current_device
        if self.config.t5_cpu:
            context = self.text_encoder(texts, torch.device("cpu"))
            return [t.to(device) for t in context]
        else:
            self.text_encoder.model.to(device)
            context = self.text_encoder(texts, device)
            self.text_encoder.model.cpu()
            return context

    @torch.no_grad()
    def vace_encode_frames(self, frames: list[torch.Tensor], masks: list[torch.Tensor]):
        """Encode video frames and masks into VACE context.

        History frames are placed in the reactive channel (channels 16-31)
        as conditioning reference for generation.  Future regions go to the
        inactive channel (zeros, since frames are pre-masked).

        Args:
            frames: list of [3, T, H, W] video tensors (future already zeroed)
            masks: list of [1, T, H, W] binary mask tensors (0=history, 1=future)

        Returns:
            vace_context: list of [96, T_lat, H_lat, W_lat] tensors
        """
        masks_binary = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]

        # Inactive: future regions (zeros, since frames are pre-masked)
        inactive = [f * m for f, m in zip(frames, masks_binary)]
        # Reactive: history frames as conditioning reference
        reactive = [f * (1 - m) for f, m in zip(frames, masks_binary)]

        inactive_latents = self.vae.encode(inactive)
        reactive_latents = self.vae.encode(reactive)

        # Concatenate inactive + reactive -> 32 channels
        frame_latents = [
            torch.cat([u, c], dim=0)
            for u, c in zip(inactive_latents, reactive_latents)
        ]

        # Encode masks to latent resolution -> 64 channels
        mask_latents = self._encode_masks(masks_binary)

        # Concatenate frame_latents (32ch) + mask_latents (64ch) -> 96ch
        vace_context = [
            torch.cat([z, m], dim=0)
            for z, m in zip(frame_latents, mask_latents)
        ]

        return vace_context

    def _encode_masks(self, masks: list[torch.Tensor]) -> list[torch.Tensor]:
        """Downsample binary masks to latent resolution.

        Input: [1, T, H, W] -> Output: [64, T_lat, H_lat, W_lat]
        (64 = vae_stride[1] * vae_stride[2] = 8*8)
        """
        result = []
        for mask in masks:
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // self.vae_stride[0])
            height = 2 * (int(height) // (self.vae_stride[1] * 2))
            width = 2 * (int(width) // (self.vae_stride[2] * 2))

            m = mask[0]  # [T, H, W]
            m = m.view(
                depth, height, self.vae_stride[1],
                width, self.vae_stride[2],
            )
            m = m.permute(2, 4, 0, 1, 3)  # [8, 8, T, H_lat, W_lat]
            m = m.reshape(
                self.vae_stride[1] * self.vae_stride[2],
                depth, height, width,
            )

            m = F.interpolate(
                m.unsqueeze(0),
                size=(new_depth, height, width),
                mode="nearest-exact",
            ).squeeze(0)

            result.append(m)
        return result

    def forward_with_actions(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        vace_context: list[torch.Tensor],
        actions: torch.Tensor,
        context: list[torch.Tensor],
        seq_len: int,
        vace_context_scale: float = 1.0,
    ) -> list[torch.Tensor]:
        """Modified VaceWanModel forward with action token injection.

        Args:
            x: list of noisy latent tensors [C_in, F, H, W]
            t: diffusion timesteps [B]
            vace_context: list of VACE context tensors [96, F_lat, H_lat, W_lat]
            actions: action tensor [B, T_action, action_dim]
            context: list of T5 text embeddings [L, C]
            seq_len: max sequence length
            vace_context_scale: VACE context strength

        Returns:
            list of predicted velocity tensors [C_out, F, H, W]
        """
        model = self.vace_model
        device = model.patch_embedding.weight.device

        if model.freqs.device != device:
            model.freqs = model.freqs.to(device)

        # Patch embedding
        x_emb = [model.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x_emb]
        )
        x_emb = [u.flatten(2).transpose(1, 2) for u in x_emb]
        seq_lens = torch.tensor([u.size(1) for u in x_emb], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x_emb = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x_emb
        ])

        # Time embeddings
        with amp.autocast(dtype=torch.float32):
            e = model.time_embedding(
                sinusoidal_embedding_1d(model.freq_dim, t).float()
            )
            e0 = model.time_projection(e).unflatten(1, (6, model.dim))

        # Text context embedding
        text_ctx = model.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )  # [B, 512, 1536]

        # Action token injection
        action_tokens = self.action_adapter(
            self.action_tokenizer(actions)
        )  # [B, T_action, 1536]

        # Concatenate action tokens with text context
        context_augmented = torch.cat([action_tokens, text_ctx], dim=1)

        # Build kwargs
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=model.freqs,
            context=context_augmented,
            context_lens=None,
        )

        # VACE forward -> skip connections
        hints = model.forward_vace(x_emb, vace_context, seq_len, kwargs)
        kwargs["hints"] = hints
        kwargs["context_scale"] = vace_context_scale

        # Main transformer blocks
        for block in model.blocks:
            if self.config.gradient_checkpointing and self.training:
                x_emb = torch.utils.checkpoint.checkpoint(
                    block, x_emb, use_reentrant=False, **kwargs
                )
            else:
                x_emb = block(x_emb, **kwargs)

        # Head
        x_emb = model.head(x_emb, e)

        # Unpatchify
        x_emb = model.unpatchify(x_emb, grid_sizes)
        return [u.float() for u in x_emb]

    def compute_seq_len(self, latent_shape: tuple) -> int:
        """Compute seq_len from latent shape [C, F, H, W]."""
        _, f, h, w = latent_shape
        return math.ceil(
            (h * w) / (self.patch_size[1] * self.patch_size[2]) * f
        )

    def forward(
        self,
        video: torch.Tensor,
        mask: torch.Tensor,
        actions: torch.Tensor,
        tasks: list[str],
        vace_context_scale: float = 1.0,
    ) -> dict:
        """Forward pass — delegates to training_step. Required for DDP."""
        return self.training_step(video, mask, actions, tasks, vace_context_scale)

    def training_step(
        self,
        video: torch.Tensor,
        mask: torch.Tensor,
        actions: torch.Tensor,
        tasks: list[str],
        vace_context_scale: float = 1.0,
    ) -> dict:
        """Compute flow matching training loss.

        Args:
            video: [B, 3, T, H, W] video frames
            mask: [B, 1, T, H, W] binary mask (0=history, 1=future)
            actions: [B, T_action, action_dim]
            tasks: list of task description strings
            vace_context_scale: VACE context strength

        Returns:
            dict with 'loss'
        """
        B = video.shape[0]
        device = self.current_device

        # Encode text (cached on CPU if t5_cpu)
        text_context = self.encode_text(tasks)

        # Encode video and masks to VACE context (frozen VAE)
        # Zero out future frames so the model cannot look ahead through
        # the reactive channel — matches eval-time behaviour.
        with torch.no_grad():
            masked_video = video * (1 - mask)
            masked_list = [masked_video[i] for i in range(B)]
            masks_list = [mask[i] for i in range(B)]

            vace_ctx = self.vace_encode_frames(masked_list, masks_list)

            # Encode full video (not masked) as target latents
            full_list = [video[i] for i in range(B)]
            target_latents = self.vae.encode(full_list)  # list of [16, F_lat, H_lat, W_lat]

        # Flow matching: sample sigma in [0, 1] and interpolate
        # Wan convention: sigma=0 (timestep=0) is clean, sigma=1 (timestep=1000) is noise
        # x_sigma = (1 - sigma) * target + sigma * noise
        sigma = torch.rand(B, device=device)
        noise = [torch.randn_like(z) for z in target_latents]

        x_t = []
        for i in range(B):
            si = sigma[i].view(1, 1, 1, 1)
            x_t.append((1 - si) * target_latents[i] + si * noise[i])

        # Compute seq_len
        seq_len = self.compute_seq_len(target_latents[0].shape)

        # Scale sigma to timestep: t = sigma * num_train_timesteps
        t_scaled = sigma * self.config.num_train_timesteps

        # Forward pass
        predicted = self.forward_with_actions(
            x=x_t,
            t=t_scaled,
            vace_context=vace_ctx,
            actions=actions,
            context=text_context,
            seq_len=seq_len,
            vace_context_scale=vace_context_scale,
        )

        # Target velocity = noise - target (dx/dsigma for flow matching)
        target_velocity = [n - z for z, n in zip(target_latents, noise)]

        # MSE loss
        loss = sum(
            torch.nn.functional.mse_loss(pred, target)
            for pred, target in zip(predicted, target_velocity)
        ) / B

        return {"loss": loss}
