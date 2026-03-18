"""
Debug script to visualize the mask split into inactive (history) and reactive (future).

Saves per-frame PNGs so you can verify:
  - inactive_tXX.png: should show real image for history frames (0..num_history-1),
    black for future frames.
  - reactive_tXX.png: should be black for history frames, real image for future frames.
  - mask_tXX.png: white = generate (future), black = keep (history).

Usage:
    python -m action_video.debug_mask [--sample_index 0] [--output_dir debug_mask_output]
"""

import argparse
import logging
from pathlib import Path

import torch
from PIL import Image

from .config import ActionVideoConfig
from .dataset import LiberoVideoDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [3, H, W] float tensor in [0, 1] to PIL Image."""
    arr = (tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def main():
    parser = argparse.ArgumentParser(description="Debug mask visualization")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="debug_mask_output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ActionVideoConfig()
    dataset = LiberoVideoDataset(config)
    sample = dataset[args.sample_index]

    video = sample["video"]  # [3, T, H, W] in [-1, 1]
    mask = sample["mask"]    # [1, T, H, W]
    task = sample["task"]
    T = video.shape[1]

    logger.info(f"Task: {task}")
    logger.info(f"Video shape: {list(video.shape)}, Mask shape: {list(mask.shape)}")
    logger.info(f"num_history={config.num_history_frames}, num_future={config.num_future_frames}")

    # Binarize mask (same as model.vace_encode_frames)
    mask_binary = torch.where(mask > 0.5, 1.0, 0.0)

    # Inactive: history visible, future zeroed
    inactive = video * (1 - mask_binary)
    # Reactive: future visible, history zeroed
    reactive = video * mask_binary

    # Rescale from [-1, 1] to [0, 1] for saving
    video_vis = (video.clamp(-1, 1) + 1) / 2
    inactive_vis = (inactive.clamp(-1, 1) + 1) / 2
    reactive_vis = (reactive.clamp(-1, 1) + 1) / 2

    for t in range(T):
        is_history = mask_binary[0, t].mean().item() < 0.5
        label = "HISTORY" if is_history else "FUTURE"
        logger.info(f"  frame {t:2d}: mask_mean={mask_binary[0, t].mean().item():.1f}  [{label}]")

        tensor_to_pil(video_vis[:, t]).save(output_dir / f"original_t{t:02d}.png")
        tensor_to_pil(inactive_vis[:, t]).save(output_dir / f"inactive_t{t:02d}.png")
        tensor_to_pil(reactive_vis[:, t]).save(output_dir / f"reactive_t{t:02d}.png")

        # Save mask as grayscale
        mask_frame = (mask_binary[0, t] * 255).byte().numpy()
        Image.fromarray(mask_frame, mode="L").save(output_dir / f"mask_t{t:02d}.png")

    logger.info(f"Saved {T * 4} images to {output_dir}/")
    logger.info("Expected:")
    logger.info(f"  - inactive_t00..t{config.num_history_frames - 1:02d}: real images (history)")
    logger.info(f"  - inactive_t{config.num_history_frames:02d}..t{T - 1:02d}: black (future zeroed)")
    logger.info(f"  - reactive_t00..t{config.num_history_frames - 1:02d}: black (history zeroed)")
    logger.info(f"  - reactive_t{config.num_history_frames:02d}..t{T - 1:02d}: real images (future)")


if __name__ == "__main__":
    main()
