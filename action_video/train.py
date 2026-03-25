"""
Training script for action-conditioned Wan2.1-VACE.

Usage (single-GPU):
    python -m action_video.train

Usage (multi-GPU with Accelerate):
    accelerate launch -m action_video.train

Override config:
    python -m action_video.train --config-name libero training.batch_size=8

Supports multi-GPU DDP via HuggingFace Accelerate, Hydra config management,
and multi-dataset training via ConcatDataset.
"""

import dataclasses
import gc
import io
import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm

from .config import ActionVideoConfig
from .dataset import DatasetConfig, LiberoVideoDataset, collate_fn, create_multi_dataset
from .model import ActionConditionedVace

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def setup_performance(config: ActionVideoConfig):
    """Configure A100-optimized performance settings."""
    if config.use_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if config.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True


class Trainer:

    def __init__(self, config: ActionVideoConfig, train_loader: DataLoader):
        self.config = config

        setup_performance(config)

        # Determine Accelerate mixed precision setting
        if config.amp_dtype == "bfloat16":
            mixed_precision = "bf16"
        elif config.amp_dtype == "float16":
            mixed_precision = "fp16"
        else:
            mixed_precision = "no"

        # Create Accelerator with DDP support
        # find_unused_parameters=True because the model has data-dependent branches
        # (LoRA, optional VACE block training, gradient checkpointing)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs],
        )

        self.device = self.accelerator.device
        self.is_main = self.accelerator.is_main_process

        if self.is_main:
            logger.info("Initializing model on CPU...")

        # Initialize model on CPU — Accelerate handles device placement
        model = ActionConditionedVace(config, device="cpu")

        # Create optimizer for trainable params only
        trainable_params = model.get_trainable_parameters()
        total_trainable = sum(p.numel() for p in trainable_params)
        total_params = sum(p.numel() for p in model.vace_model.parameters())
        if self.is_main:
            logger.info(
                f"Trainable params: {total_trainable:,} / {total_params:,} "
                f"({100 * total_trainable / total_params:.1f}%)"
            )

        optimizer = optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
        )

        # Cosine scheduler with warmup (NOT passed to accelerator.prepare)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(config.num_train_steps - config.warmup_steps, 1),
            eta_min=1e-7,
        )

        # Prepare with Accelerate (handles DDP wrapping, device placement)
        self.model, self.optimizer, self.train_loader = \
            self.accelerator.prepare(model, optimizer, train_loader)

        # Move frozen components (VAE, T5) to GPU AFTER accelerator.prepare()
        # These are not nn.Module submodules, so DDP doesn't touch them
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.setup_frozen_components(self.device)

        if self.is_main:
            logger.info(f"Frozen components moved to {self.device}")

        # torch.compile for the forward pass
        if config.compile_model and torch.cuda.is_available():
            if self.is_main:
                logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Checkpoint dir (only on main process)
        self.save_dir = Path(config.checkpoint_save_dir) / config.exp_name
        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Resume
        if config.resume_checkpoint:
            self.load_checkpoint(config.resume_checkpoint)

    def get_lr(self) -> float:
        if self.step < self.config.warmup_steps:
            return self.config.learning_rate * self.step / max(self.config.warmup_steps, 1)
        return self.optimizer.param_groups[0]["lr"]

    def save_checkpoint(self, step: int, loss: float = None):
        """Save only trainable parameters (main process only)."""
        self.accelerator.wait_for_everyone()
        if not self.is_main:
            return

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        trainable_names = {
            name for name, p in unwrapped_model.named_parameters() if p.requires_grad
        }
        trainable_state = {
            name: param
            for name, param in unwrapped_model.state_dict().items()
            if name in trainable_names
        }

        checkpoint = {
            "step": step,
            "epoch": self.epoch,
            "model_state_dict": trainable_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": dataclasses.asdict(self.config),
        }

        is_best = loss is not None and loss < self.best_loss
        if is_best:
            self.best_loss = loss

        path = self.save_dir / f"checkpoint_{step}.pt"
        buf = io.BytesIO()
        torch.save(checkpoint, buf)
        buf.seek(0)
        with open(path, "wb") as f:
            f.write(buf.getvalue())

        logger.info(f"Saved checkpoint: {path}" + (f" (best: {loss:.6f})" if is_best else ""))

        # Symlink latest
        latest = self.save_dir / "checkpoint_latest.pt"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(path.name)

        # Prune old checkpoints
        if self.config.max_checkpoints > 0:
            ckpts = sorted(
                self.save_dir.glob("checkpoint_[0-9]*.pt"),
                key=lambda p: p.stat().st_mtime,
            )
            best_path = self.save_dir / f"checkpoint_{step}.pt" if is_best else None
            while len(ckpts) > self.config.max_checkpoints:
                oldest = ckpts.pop(0)
                if best_path and oldest.resolve() == best_path.resolve():
                    continue
                oldest.unlink()

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        missing, unexpected = unwrapped_model.load_state_dict(state, strict=False)
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        if self.is_main and missing:
            logger.info(f"Loaded checkpoint from {path} (step {checkpoint['step']}, {len(missing)} missing keys)")

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint["step"]
        self.epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))

    def train(self):
        if self.is_main:
            logger.info(f"Starting training from step {self.step}")
            effective_batch = (
                self.config.batch_size
                * self.config.gradient_accumulation_steps
                * self.accelerator.num_processes
            )
            logger.info(
                f"Effective batch size: {effective_batch} "
                f"(batch_size={self.config.batch_size} x accum={self.config.gradient_accumulation_steps} "
                f"x gpus={self.accelerator.num_processes})"
            )

        # Initialize WandB (main process only)
        wandb_run = None
        if self.config.wandb_enabled and self.is_main:
            try:
                import wandb

                wandb_id_file = self.save_dir / "wandb_id.txt"
                if self.step == 0:
                    wandb_run = wandb.init(
                        project=self.config.project_name,
                        name=self.config.exp_name,
                        config=dataclasses.asdict(self.config),
                    )
                    wandb_id_file.write_text(wandb.run.id)
                elif wandb_id_file.exists():
                    wandb_id = wandb_id_file.read_text().strip()
                    wandb_run = wandb.init(
                        project=self.config.project_name,
                        id=wandb_id,
                        resume="must",
                        config=dataclasses.asdict(self.config),
                    )
                    logger.info(f"Resumed wandb run: {wandb_id}")
                else:
                    wandb_run = wandb.init(
                        project=self.config.project_name,
                        name=self.config.exp_name,
                        config=dataclasses.asdict(self.config),
                    )
                    wandb_id_file.write_text(wandb.run.id)
            except Exception as e:
                logger.warning(f"WandB init failed: {e}. Continuing without logging.")

        pbar = tqdm(
            total=self.config.num_train_steps,
            initial=self.step,
            desc="Training",
            disable=not self.is_main,
        )

        accumulated_metrics = {}
        self.optimizer.zero_grad()

        while self.step < self.config.num_train_steps:
            for batch in self.train_loader:
                # Move batch to device
                video = batch["video"].to(self.device)
                mask = batch["mask"].to(self.device)
                tasks = batch["task"]

                actions = batch["actions"]
                if actions is not None:
                    actions = actions.to(self.device)
                else:
                    B = video.shape[0]
                    actions = torch.zeros(
                        B, self.config.num_future_frames, 7,
                        device=self.device,
                    )

                # Apply warmup LR
                if self.step < self.config.warmup_steps:
                    lr = self.get_lr()
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr

                # Accelerate accumulation context handles grad sync and loss scaling
                with self.accelerator.accumulate(self.model):
                    # Forward pass through DDP (Accelerate handles autocast)
                    loss_dict = self.model(
                        video=video,
                        mask=mask,
                        actions=actions,
                        tasks=tasks,
                    )

                    loss = loss_dict["loss"]

                    # Backward pass (Accelerate handles scaling and gradient sync)
                    self.accelerator.backward(loss)

                    # Only perform optimizer operations on actual gradient sync
                    if self.accelerator.sync_gradients:
                        # Gradient clipping
                        if self.config.gradient_clip > 0:
                            self.accelerator.clip_grad_norm_(
                                self.accelerator.unwrap_model(self.model).get_trainable_parameters(),
                                self.config.gradient_clip,
                            )

                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        if self.step >= self.config.warmup_steps:
                            self.scheduler.step()

                        self.step += 1

                # Accumulate metrics
                loss_val = loss.detach().item()
                if "loss" not in accumulated_metrics:
                    accumulated_metrics["loss"] = []
                accumulated_metrics["loss"].append(loss_val)

                # Step-based operations (only on actual optimizer steps)
                if self.accelerator.sync_gradients:
                    # Logging (main process only)
                    if self.is_main and self.step % self.config.log_interval == 0 and self.step > 0:
                        avg_loss = sum(accumulated_metrics["loss"]) / len(accumulated_metrics["loss"])
                        lr = self.get_lr()
                        pbar.write(f"Step {self.step}: loss={avg_loss:.6f}, lr={lr:.2e}")

                        if wandb_run:
                            import wandb
                            wandb.log({"loss": avg_loss, "lr": lr}, step=self.step)

                        accumulated_metrics = {}

                    # Save checkpoint
                    if self.step % self.config.save_interval == 0 and self.step > 0:
                        avg_loss = sum(accumulated_metrics.get("loss", [loss_val])) / max(
                            len(accumulated_metrics.get("loss", [1])), 1
                        )
                        self.save_checkpoint(self.step, loss=avg_loss)
                        gc.collect()
                        torch.cuda.empty_cache()

                    # Update progress bar
                    if self.is_main:
                        pbar.update(1)

                if self.step >= self.config.num_train_steps:
                    break

            self.epoch += 1

        # Final save
        self.save_checkpoint(self.step, loss=self.best_loss)
        pbar.close()
        if self.is_main:
            logger.info("Training completed!")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    # Resolve original working directory (Hydra changes cwd)
    orig_cwd = hydra.utils.get_original_cwd()
    os.chdir(orig_cwd)

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Build ActionVideoConfig from Hydra config
    config = ActionVideoConfig.from_hydra(cfg)

    # Build per-dataset configs
    dataset_configs = []
    for ds_cfg in cfg.datasets:
        ds = DatasetConfig(
            repo_id=ds_cfg.repo_id,
            image_key=ds_cfg.get("image_key", "observation.images.image"),
            action_key=ds_cfg.get("action_key", "action"),
            state_key=ds_cfg.get("state_key", "observation.state"),
            task_key=ds_cfg.get("task_key", "task"),
        )
        dataset_configs.append(ds)
        logger.info(f"Added dataset: {ds.repo_id}")

    # Create dataset (single or multi via ConcatDataset)
    logger.info("Creating dataset...")
    dataset = create_multi_dataset(config, dataset_configs)

    # Create DataLoader
    # shuffle=False because StreamingLeRobotDataset (IterableDataset) handles its own shuffling
    use_persistent = config.num_workers > 0
    prefetch = 4 if config.num_workers > 0 else None
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=cfg.training.get("pin_memory", True),
        drop_last=cfg.training.get("drop_last", True),
        collate_fn=collate_fn,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch,
    )
    logger.info(f"DataLoader: streaming from {len(dataset_configs)} dataset(s)")

    # Create trainer (Accelerator is initialized inside)
    trainer = Trainer(config, train_loader)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
