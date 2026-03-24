import dataclasses
from typing import Optional

import torch


_T5_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclasses.dataclass
class ActionVideoConfig:
    # Paths
    checkpoint_dir: str = "Wan2.1-VACE-1.3B"
    t5_checkpoint: str = "models_t5_umt5-xxl-enc-bf16.pth"
    t5_tokenizer: str = "google/umt5-xxl"
    vae_checkpoint: str = "Wan2.1_VAE.pth"

    # Model dims (from Wan2.1-VACE-1.3B/config.json)
    dim: int = 1536
    num_heads: int = 12
    num_layers: int = 30
    vae_stride: tuple = (4, 8, 8)
    patch_size: tuple = (1, 2, 2)
    num_train_timesteps: int = 1000

    # Action conditioning
    max_action_dim: int = 32
    action_hidden_dim: int = 256
    max_action_len: int = 32

    # VACE image settings
    target_height: int = 256
    target_width: int = 448
    num_frames: int = 13  # 4n+1 for VAE: 1 history + 12 future
    num_history_frames: int = 3
    num_future_frames: int = 10

    # Data (used by eval scripts; training uses per-dataset DatasetConfig)
    repo_id: str = "HuggingFaceVLA/libero"
    image_key: str = "observation.images.image"
    action_key: str = "action"
    state_key: str = "observation.state"
    task_key: str = "task"
    horizon: int = 30
    horizon_skip: int = 5

    # Training
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    num_train_steps: int = 5000
    warmup_steps: int = 200
    gradient_clip: float = 1.0
    amp_dtype: str = "bfloat16"

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_ffn: bool = False
    use_lora: bool = True
    train_vace_blocks: bool = True
    gradient_checkpointing: bool = True

    # Performance (A100)
    use_tf32: bool = True
    compile_model: bool = False
    cudnn_benchmark: bool = True

    # T5
    t5_cpu: bool = False
    t5_dtype: torch.dtype = torch.bfloat16
    text_len: int = 256

    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 250
    checkpoint_save_dir: str = "/scratch/s5649552/Wan2.1/checkpoints"
    exp_name: str = "action_vace_libero"
    project_name: str = "action_vace"
    wandb_enabled: bool = True
    max_checkpoints: int = 3

    # Resume
    resume_checkpoint: Optional[str] = None

    # Workers
    num_workers: int = 8

    def __post_init__(self):
        # Convert lists from YAML to tuples
        if isinstance(self.vae_stride, list):
            self.vae_stride = tuple(self.vae_stride)
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)

    @property
    def param_dtype(self) -> torch.dtype:
        if self.amp_dtype == "bfloat16":
            return torch.bfloat16
        elif self.amp_dtype == "float16":
            return torch.float16
        return torch.float32

    @classmethod
    def from_hydra(cls, cfg) -> "ActionVideoConfig":
        """Construct from nested Hydra DictConfig (model + training groups)."""
        model = cfg.model
        training = cfg.training

        # Resolve t5_dtype from string
        t5_dtype_str = model.get("t5_dtype", "bfloat16")
        t5_dtype = _T5_DTYPE_MAP.get(t5_dtype_str, torch.bfloat16)

        return cls(
            # Model fields
            checkpoint_dir=model.checkpoint_dir,
            t5_checkpoint=model.t5_checkpoint,
            t5_tokenizer=model.t5_tokenizer,
            vae_checkpoint=model.vae_checkpoint,
            dim=model.dim,
            num_heads=model.num_heads,
            num_layers=model.num_layers,
            vae_stride=tuple(model.vae_stride),
            patch_size=tuple(model.patch_size),
            num_train_timesteps=model.num_train_timesteps,
            max_action_dim=model.max_action_dim,
            action_hidden_dim=model.action_hidden_dim,
            max_action_len=model.max_action_len,
            target_height=model.target_height,
            target_width=model.target_width,
            num_frames=model.num_frames,
            num_history_frames=model.num_history_frames,
            num_future_frames=model.num_future_frames,
            t5_cpu=model.t5_cpu,
            t5_dtype=t5_dtype,
            text_len=model.text_len,
            # Training fields
            batch_size=training.batch_size,
            gradient_accumulation_steps=training.gradient_accumulation_steps,
            learning_rate=training.learning_rate,
            weight_decay=training.weight_decay,
            num_train_steps=training.num_train_steps,
            warmup_steps=training.warmup_steps,
            gradient_clip=training.gradient_clip,
            amp_dtype=training.amp_dtype,
            lora_rank=training.lora_rank,
            lora_alpha=training.lora_alpha,
            lora_ffn=training.lora_ffn,
            use_lora=training.use_lora,
            train_vace_blocks=training.train_vace_blocks,
            gradient_checkpointing=training.gradient_checkpointing,
            use_tf32=training.use_tf32,
            compile_model=training.compile_model,
            cudnn_benchmark=training.cudnn_benchmark,
            log_interval=training.log_interval,
            save_interval=training.save_interval,
            wandb_enabled=training.wandb_enabled,
            max_checkpoints=training.max_checkpoints,
            resume_checkpoint=training.get("resume_checkpoint", None),
            num_workers=training.num_workers,
            # Top-level fields
            exp_name=cfg.exp_name,
            project_name=cfg.project_name,
            checkpoint_save_dir=cfg.checkpoint_dir,
            horizon=cfg.horizon,
            horizon_skip=cfg.horizon_skip,
        )
