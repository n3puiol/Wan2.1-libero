"""
Training script for action-conditioned Wan2.1-VACE on Libero data.

Usage:
    python -m action_video.train [--checkpoint_dir Wan2.1-VACE-1.3B] [--batch_size 1] ...

Supports single-GPU training with gradient accumulation and bfloat16 AMP.
"""

import argparse
import dataclasses
import gc
import io
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ActionVideoConfig
from .dataset import LiberoVideoDataset, collate_fn
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
        logger.info("TF32 enabled for matmul and cuDNN")
    if config.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark enabled")


def parse_args() -> ActionVideoConfig:
    parser = argparse.ArgumentParser(description="Train action-conditioned VACE")

    config = ActionVideoConfig()
    for field in dataclasses.fields(config):
        if field.type == torch.dtype:
            continue
        name = f"--{field.name}"
        if field.type == bool:
            parser.add_argument(name, type=lambda x: x.lower() == "true", default=field.default)
        elif field.type == tuple:
            continue  # skip tuples, use defaults
        else:
            parser.add_argument(name, type=field.type, default=field.default)

    args = parser.parse_args()
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


class Trainer:

    def __init__(self, config: ActionVideoConfig):
        self.config = config
        self.device = torch.device("cuda:0")

        setup_performance(config)

        logger.info("Initializing model...")
        self.model = ActionConditionedVace(config, device_id=0)

        # Move trainable modules to device
        self.model.action_tokenizer.to(self.device)
        self.model.action_adapter.to(self.device)

        # torch.compile for the forward pass (A100 SM80+ supports inductor)
        if config.compile_model:
            logger.info("Compiling model with torch.compile (may take a minute on first step)...")
            self.model.forward_with_actions = torch.compile(
                self.model.forward_with_actions, mode="reduce-overhead"
            )

        # Create optimizer for trainable params only
        trainable_params = self.model.get_trainable_parameters()
        total_trainable = sum(p.numel() for p in trainable_params)
        total_params = sum(p.numel() for p in self.model.vace_model.parameters())
        logger.info(
            f"Trainable params: {total_trainable:,} / {total_params:,} "
            f"({100 * total_trainable / total_params:.1f}%)"
        )

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
        )

        # Cosine scheduler with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(config.num_train_steps - config.warmup_steps, 1),
            eta_min=1e-7,
        )

        # Create dataset and dataloader
        logger.info("Creating dataset...")
        dataset = LiberoVideoDataset(config)
        self.train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=config.num_workers > 0,
        )
        logger.info(f"Dataset: {len(dataset)} samples, {len(self.train_loader)} batches")

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Checkpoint dir
        self.save_dir = Path(config.checkpoint_save_dir) / config.exp_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # WandB
        self.wandb_run = None
        if config.wandb_enabled:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.project_name,
                    name=config.exp_name,
                    config=dataclasses.asdict(config),
                )
            except Exception as e:
                logger.warning(f"WandB init failed: {e}. Continuing without logging.")

        # Resume
        if config.resume_checkpoint:
            self.load_checkpoint(config.resume_checkpoint)

    def get_lr(self) -> float:
        if self.step < self.config.warmup_steps:
            return self.config.learning_rate * self.step / max(self.config.warmup_steps, 1)
        return self.optimizer.param_groups[0]["lr"]

    def save_checkpoint(self, step: int, loss: float = None):
        """Save only trainable parameters."""
        trainable_names = {
            name for name, p in self.model.named_parameters() if p.requires_grad
        }
        trainable_state = {
            name: param
            for name, param in self.model.state_dict().items()
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
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        logger.info(f"Loaded checkpoint from {path} (step {checkpoint['step']}, {len(missing)} missing keys)")

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint["step"]
        self.epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))

    def train(self):
        logger.info(f"Starting training from step {self.step}")
        effective_batch = self.config.batch_size * self.config.gradient_accumulation_steps
        logger.info(f"Effective batch size: {effective_batch}")

        pbar = tqdm(total=self.config.num_train_steps, initial=self.step, desc="Training")
        accumulated_loss = 0.0
        accum_count = 0

        self.optimizer.zero_grad()

        while self.step < self.config.num_train_steps:
            for batch in self.train_loader:
                # Move to device
                video = batch["video"].to(self.device)
                mask = batch["mask"].to(self.device)
                tasks = batch["task"]

                actions = batch["actions"]
                if actions is not None:
                    actions = actions.to(self.device)
                else:
                    # Fallback: zero actions
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

                # Forward pass with AMP
                with torch.amp.autocast("cuda", dtype=self.config.param_dtype):
                    loss_dict = self.model.training_step(
                        video=video,
                        mask=mask,
                        actions=actions,
                        tasks=tasks,
                    )

                loss = loss_dict["loss"] / self.config.gradient_accumulation_steps
                loss.backward()

                accumulated_loss += loss_dict["loss"].detach().item()
                accum_count += 1

                # Optimizer step at accumulation boundary
                if accum_count >= self.config.gradient_accumulation_steps:
                    if self.config.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.get_trainable_parameters(),
                            self.config.gradient_clip,
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.step >= self.config.warmup_steps:
                        self.scheduler.step()

                    self.step += 1
                    avg_loss = accumulated_loss / accum_count
                    accumulated_loss = 0.0
                    accum_count = 0

                    # Logging
                    if self.step % self.config.log_interval == 0:
                        lr = self.get_lr()
                        pbar.write(
                            f"Step {self.step}: loss={avg_loss:.6f}, lr={lr:.2e}"
                        )
                        if self.wandb_run:
                            import wandb
                            wandb.log({"loss": avg_loss, "lr": lr}, step=self.step)

                    # Save checkpoint
                    if self.step % self.config.save_interval == 0 and self.step > 0:
                        self.save_checkpoint(self.step, loss=avg_loss)
                        gc.collect()
                        torch.cuda.empty_cache()

                    pbar.update(1)

                if self.step >= self.config.num_train_steps:
                    break

            self.epoch += 1

        # Final save
        self.save_checkpoint(self.step, loss=self.best_loss)
        pbar.close()
        logger.info("Training completed!")


def main():
    config = parse_args()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
