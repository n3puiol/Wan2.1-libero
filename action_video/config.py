import dataclasses
from typing import Optional

import torch


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

    # Data
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
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_steps: int = 2000
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    amp_dtype: str = "bfloat16"

    # LoRA
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_ffn: bool = True
    use_lora: bool = True
    train_vace_blocks: bool = False
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
    save_interval: int = 100
    checkpoint_save_dir: str = "checkpoints"
    exp_name: str = "action_vace_libero"
    project_name: str = "action_vace"
    wandb_enabled: bool = True
    max_checkpoints: int = 3

    # Resume
    resume_checkpoint: Optional[str] = None

    # Workers
    num_workers: int = 8

    @property
    def param_dtype(self) -> torch.dtype:
        if self.amp_dtype == "bfloat16":
            return torch.bfloat16
        elif self.amp_dtype == "float16":
            return torch.float16
        return torch.float32
