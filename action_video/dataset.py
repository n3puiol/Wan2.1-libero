"""
LiberoVideoDataset: loads Libero data via LeRobot and prepares it for
VACE-style video prediction with action conditioning.

Each sample contains:
  - video: [3, num_frames, H, W] full clip (history + future)
  - mask: [1, num_frames, H, W] binary mask (0=history, 1=future)
  - actions: [num_future_frames, action_dim] normalized future actions
  - task: str task description
"""

import logging
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from .config import ActionVideoConfig

logger = logging.getLogger(__name__)


class NormStat:
    """Action normalization statistics."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, min_val: np.ndarray, max_val: np.ndarray):
        self.mean = mean
        self.std = std
        self.min = min_val
        self.max = max_val
        self._cache: dict = {}

    def get_tensors(self, dim: int, dtype: torch.dtype, device: torch.device) -> dict:
        key = (dim, dtype, device)
        if key not in self._cache:
            self._cache[key] = {
                "mean": torch.tensor(self.mean[:dim], dtype=dtype, device=device),
                "std": torch.tensor(self.std[:dim], dtype=dtype, device=device),
                "min": torch.tensor(self.min[:dim], dtype=dtype, device=device),
                "max": torch.tensor(self.max[:dim], dtype=dtype, device=device),
            }
        return self._cache[key]


def normalize_action(x: torch.Tensor, stats: NormStat) -> torch.Tensor:
    """Normalize to [-1, 1] via z-score then min-max rescaling."""
    dim = x.shape[-1]
    tensors = stats.get_tensors(dim, x.dtype, x.device)
    mean, std = tensors["mean"], tensors["std"]
    min_val, max_val = tensors["min"], tensors["max"]

    std_safe = std.clamp(min=1e-6)
    x = (x - mean) / std_safe

    z_min = (min_val - mean) / std_safe
    z_max = (max_val - mean) / std_safe
    z_span = (z_max - z_min).clamp(min=1e-6)

    x = 2.0 * (x - z_min) / z_span - 1.0
    return x.clamp(-1.0, 1.0)


class LiberoVideoDataset(torch.utils.data.Dataset):

    def __init__(self, config: ActionVideoConfig):
        self.config = config
        self.num_history = config.num_history_frames
        self.num_future = config.num_future_frames
        self.target_h = config.target_height
        self.target_w = config.target_width

        meta = LeRobotDatasetMetadata(config.repo_id)
        fps = meta.fps

        # Build delta_timestamps
        # History: frames going backwards from current, spaced by horizon_skip
        # Future: frames going forward, spaced by horizon_skip
        skip = config.horizon_skip
        history_ts = [
            -(self.num_history - 1 - i) * skip / fps
            for i in range(self.num_history)
        ]
        future_ts = [
            (i + 1) * skip / fps
            for i in range(self.num_future)
        ]

        delta_timestamps = {
            config.image_key: history_ts + future_ts,
        }
        if config.action_key in meta.features:
            delta_timestamps[config.action_key] = future_ts

        self.dataset = LeRobotDataset(
            config.repo_id,
            delta_timestamps=delta_timestamps,
        )

        # Build action normalization stats
        self.action_stats = None
        stats = self.dataset.meta.stats
        if config.action_key in stats:
            self.action_stats = NormStat(
                mean=np.array(stats[config.action_key]["mean"]),
                std=np.array(stats[config.action_key]["std"]),
                min_val=np.array(stats[config.action_key]["min"]),
                max_val=np.array(stats[config.action_key]["max"]),
            )

        logger.info(
            f"LiberoVideoDataset: {len(self.dataset)} samples, "
            f"fps={fps}, history={self.num_history}, future={self.num_future}"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        for _ in range(10):
            try:
                return self._load_sample(index)
            except Exception as e:
                logger.warning(f"Error at index {index}: {e}. Retrying.")
                index = random.randint(0, len(self) - 1)
        raise RuntimeError(f"Failed to load sample after 10 retries")

    def _load_sample(self, index: int) -> dict:
        item = self.dataset[index]
        cfg = self.config

        # Images: [num_frames, C, H, W] from LeRobot
        images = item[cfg.image_key]  # [T, C, H, W]
        images = images.float()

        # Normalize to [-1, 1] (LeRobot gives [0, 1] after transforms)
        if images.max() > 1.0:
            images = images / 255.0
        images = images * 2.0 - 1.0

        # Resize to target resolution
        T, C, H, W = images.shape
        if H != self.target_h or W != self.target_w:
            images = F.interpolate(
                images, size=(self.target_h, self.target_w),
                mode="bilinear", align_corners=False,
            )

        # Convert to [C, T, H, W] for VACE
        video = images.permute(1, 0, 2, 3)  # [3, T, H, W]

        # Create mask: 0 for history (known), 1 for future (to generate)
        mask = torch.zeros(1, T, self.target_h, self.target_w)
        mask[:, self.num_history:, :, :] = 1.0

        # Actions
        actions = None
        if cfg.action_key in item:
            actions = item[cfg.action_key].float()  # [T_future, action_dim]
            if self.action_stats is not None:
                actions = normalize_action(actions, self.action_stats)

        # Task description
        task = item.get(cfg.task_key, "A robot arm performing a manipulation task.")

        return {
            "video": video,
            "mask": mask,
            "actions": actions,
            "task": task,
        }


def collate_fn(batch: List[dict]) -> Dict[str, torch.Tensor]:
    """Collate batch, padding actions to max dim if needed."""
    videos = torch.stack([b["video"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    tasks = [b["task"] for b in batch]

    # Pad actions to max action_dim in batch
    has_actions = any(b["actions"] is not None for b in batch)
    actions = None
    action_mask = None

    if has_actions:
        max_dim = max(
            b["actions"].shape[-1] for b in batch if b["actions"] is not None
        )
        T = next(
            b["actions"].shape[0] for b in batch if b["actions"] is not None
        )
        padded = []
        masks_a = []
        for b in batch:
            if b["actions"] is not None:
                a = b["actions"]
                pad_size = max_dim - a.shape[-1]
                if pad_size > 0:
                    a = F.pad(a, (0, pad_size), value=0.0)
                    m = torch.cat([
                        torch.ones(T, a.shape[-1] - pad_size, dtype=torch.bool),
                        torch.zeros(T, pad_size, dtype=torch.bool),
                    ], dim=-1)
                else:
                    m = torch.ones_like(a, dtype=torch.bool)
            else:
                a = torch.zeros(T, max_dim)
                m = torch.zeros(T, max_dim, dtype=torch.bool)
            padded.append(a)
            masks_a.append(m)
        actions = torch.stack(padded)
        action_mask = torch.stack(masks_a)

    return {
        "video": videos,
        "mask": masks,
        "actions": actions,
        "action_mask": action_mask,
        "task": tasks,
    }
