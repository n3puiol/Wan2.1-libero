"""
LiberoVideoDataset: loads data via StreamingLeRobotDataset and prepares it for
VACE-style video prediction with action conditioning.

Each sample contains:
  - video: [3, num_frames, H, W] full clip (history + future)
  - mask: [1, num_frames, H, W] binary mask (0=history, 1=future)
  - actions: [num_future_frames, action_dim] normalized future actions
  - task: str task description
"""

import dataclasses
import logging
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

from .config import ActionVideoConfig


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DatasetConfig:
    """Per-dataset configuration for multi-dataset training."""
    repo_id: str
    image_key: str = "observation.images.image"
    action_key: str = "action"
    state_key: str = "observation.state"
    task_key: str = "task"


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


def _process_item(
    item: dict,
    image_key: str,
    action_key: str,
    task_key: str,
    target_h: int,
    target_w: int,
    num_history: int,
    action_stats: NormStat | None,
) -> dict:
    """Shared processing logic for both map and streaming datasets."""
    images = item[image_key].float()

    if images.max() > 1.0:
        images = images / 255.0
    images = images * 2.0 - 1.0

    T, C, H, W = images.shape
    if H != target_h or W != target_w:
        images = F.interpolate(
            images, size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        )

    video = images.permute(1, 0, 2, 3)

    mask = torch.zeros(1, T, target_h, target_w)
    mask[:, num_history:, :, :] = 1.0

    actions = None
    if action_key in item:
        actions = item[action_key].float()
        if action_stats is not None:
            actions = normalize_action(actions, action_stats)

    task = item.get(task_key, "A robot arm performing a manipulation task.")

    return {
        "video": video,
        "mask": mask,
        "actions": actions,
        "task": task,
    }


class MapLiberoVideoDataset(torch.utils.data.Dataset):
    """Map-style dataset for eval/inference (supports __getitem__ and __len__)."""

    def __init__(self, config: ActionVideoConfig, dataset_config: DatasetConfig | None = None):
        self.config = config
        self.num_history = config.num_history_frames
        self.num_future = config.num_future_frames
        self.target_h = config.target_height
        self.target_w = config.target_width

        self.repo_id = dataset_config.repo_id if dataset_config else config.repo_id
        self.image_key = dataset_config.image_key if dataset_config else config.image_key
        self.action_key = dataset_config.action_key if dataset_config else config.action_key
        self.task_key = dataset_config.task_key if dataset_config else config.task_key

        meta = LeRobotDatasetMetadata(self.repo_id)
        fps = meta.fps

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
            self.image_key: history_ts + future_ts,
        }
        if self.action_key in meta.features:
            delta_timestamps[self.action_key] = future_ts

        self.dataset = LeRobotDataset(
            self.repo_id,
            delta_timestamps=delta_timestamps,
        )

        self.action_stats = None
        stats = self.dataset.meta.stats
        if self.action_key in stats:
            self.action_stats = NormStat(
                mean=np.array(stats[self.action_key]["mean"]),
                std=np.array(stats[self.action_key]["std"]),
                min_val=np.array(stats[self.action_key]["min"]),
                max_val=np.array(stats[self.action_key]["max"]),
            )

        logger.info(
            f"MapLiberoVideoDataset({self.repo_id}): {len(self.dataset)} samples, "
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
        return _process_item(item, self.image_key, self.action_key, self.task_key,
                             self.target_h, self.target_w, self.num_history, self.action_stats)


# Backward compat alias — eval scripts import this name
LiberoVideoDataset = MapLiberoVideoDataset


class StreamingLiberoVideoDataset(torch.utils.data.IterableDataset):
    """Streaming dataset for training (no __getitem__, uses StreamingLeRobotDataset)."""

    def __init__(self, config: ActionVideoConfig, dataset_config: DatasetConfig | None = None):
        super().__init__()
        self.config = config
        self.num_history = config.num_history_frames
        self.num_future = config.num_future_frames
        self.target_h = config.target_height
        self.target_w = config.target_width

        # Per-dataset keys (from DatasetConfig if provided, else from ActionVideoConfig)
        self.repo_id = dataset_config.repo_id if dataset_config else config.repo_id
        self.image_key = dataset_config.image_key if dataset_config else config.image_key
        self.action_key = dataset_config.action_key if dataset_config else config.action_key
        self.task_key = dataset_config.task_key if dataset_config else config.task_key

        # PIL -> Tensor transform (StreamingLeRobotDataset returns raw PIL images)
        self._image_transforms = T.ToTensor()

        # Create streaming dataset to read metadata (fps, features, stats)
        self._streaming_dataset = StreamingLeRobotDataset(
            self.repo_id,
            image_transforms=self._image_transforms,
            streaming=True,
        )
        fps = self._streaming_dataset.fps

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
            self.image_key: history_ts + future_ts,
        }
        if self.action_key in self._streaming_dataset.meta.features:
            delta_timestamps[self.action_key] = future_ts

        # Build action normalization stats from metadata
        self.action_stats = None
        stats = self._streaming_dataset.meta.stats
        if self.action_key in stats:
            self.action_stats = NormStat(
                mean=np.array(stats[self.action_key]["mean"]),
                std=np.array(stats[self.action_key]["std"]),
                min_val=np.array(stats[self.action_key]["min"]),
                max_val=np.array(stats[self.action_key]["max"]),
            )

        # Re-create the streaming dataset with delta_timestamps
        self._streaming_dataset = StreamingLeRobotDataset(
            self.repo_id,
            image_transforms=self._image_transforms,
            delta_timestamps=delta_timestamps,
            streaming=True,
        )

        # Monkey-patch _get_delta_frames to handle PIL images in parquet-based datasets.
        # The upstream _get_delta_frames calls torch.stack on raw PIL images when
        # image columns are stored as parquet (not video). We wrap it to convert PIL -> tensor.
        _orig_get_delta_frames = self._streaming_dataset._get_delta_frames
        _to_tensor = self._image_transforms

        def _patched_get_delta_frames(dataset_iterator, item):
            from PIL import Image
            query_result, padding = _orig_get_delta_frames.__wrapped__(
                self._streaming_dataset, dataset_iterator, item
            ) if hasattr(_orig_get_delta_frames, '__wrapped__') else _orig_get_delta_frames(dataset_iterator, item)
            return query_result, padding

        # Deeper patch: override the problematic torch.stack inside _get_delta_frames
        # by converting PIL images to tensors in the frame collection step.
        import types
        from PIL import Image as PILImage

        original_method = self._streaming_dataset._get_delta_frames

        def _safe_get_delta_frames(self_ds, dataset_iterator, item):
            """Patched _get_delta_frames that converts PIL images to tensors before stacking."""
            query_result = {}
            padding = {}

            for key, delta_indices in self_ds.delta_indices.items():
                if key in self_ds.meta.video_keys:
                    continue

                target_frames = []
                is_pad = []
                delta_results = {}

                negative_deltas = sorted([d for d in delta_indices if d < 0], reverse=True)
                positive_deltas = sorted([d for d in delta_indices if d > 0])
                zero_deltas = [d for d in delta_indices if d == 0]

                for delta in zero_deltas:
                    delta_results[delta] = (item[key], False)

                last_successful_frame = item[key]
                lookahead_failed = False

                for delta in negative_deltas:
                    try:
                        past_item = dataset_iterator.peek_backward(-delta)
                        if past_item.get("episode_index") == item.get("episode_index"):
                            delta_results[delta] = (past_item[key], False)
                            last_successful_frame = past_item[key]
                        else:
                            delta_results[delta] = (last_successful_frame, True)
                    except Exception:
                        delta_results[delta] = (last_successful_frame, True)

                last_successful_frame = item[key]
                for delta in positive_deltas:
                    if lookahead_failed:
                        delta_results[delta] = (last_successful_frame, True)
                        continue
                    try:
                        future_item = dataset_iterator.peek_forward(delta)
                        if future_item.get("episode_index") == item.get("episode_index"):
                            delta_results[delta] = (future_item[key], False)
                            last_successful_frame = future_item[key]
                        else:
                            delta_results[delta] = (last_successful_frame, True)
                            lookahead_failed = True
                    except Exception:
                        delta_results[delta] = (last_successful_frame, True)
                        lookahead_failed = True

                for delta in delta_indices:
                    frame, is_padded = delta_results[delta]
                    # Convert PIL images to tensors
                    if isinstance(frame, PILImage.Image):
                        frame = _to_tensor(frame)
                    target_frames.append(frame)
                    is_pad.append(is_padded)

                if target_frames:
                    query_result[key] = torch.stack(target_frames)
                    padding[f"{key}_is_pad"] = torch.BoolTensor(is_pad)

            return query_result, padding

        self._streaming_dataset._get_delta_frames = types.MethodType(
            _safe_get_delta_frames, self._streaming_dataset
        )

        logger.info(
            f"StreamingLiberoVideoDataset({self.repo_id}): streaming, "
            f"fps={fps}, history={self.num_history}, future={self.num_future}"
        )

    def __iter__(self):
        for item in self._streaming_dataset:
            try:
                yield _process_item(item, self.image_key, self.action_key, self.task_key,
                                    self.target_h, self.target_w, self.num_history, self.action_stats)
            except Exception as e:
                logger.warning(f"Error processing item in {self.repo_id}: {e}. Skipping.")


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


def create_multi_dataset(
    config: ActionVideoConfig,
    dataset_configs: list[DatasetConfig],
) -> torch.utils.data.IterableDataset:
    """Create a (possibly chained) streaming dataset from multiple dataset configs."""
    datasets = []
    for ds_cfg in dataset_configs:
        datasets.append(StreamingLiberoVideoDataset(config, dataset_config=ds_cfg))
    if len(datasets) == 1:
        return datasets[0]
    return torch.utils.data.ChainDataset(datasets)
