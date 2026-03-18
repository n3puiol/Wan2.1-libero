"""
Evaluation script for ActionConditionedVace.

Given a trained checkpoint, conditioning image(s), and actions from the dataset,
run diffusion sampling to generate future video frames.

Usage:
    python -m action_video.eval \
        --checkpoint checkpoints/action_vace_libero/checkpoint_latest.pt \
        --num_samples 8 \
        --output_dir eval_outputs
"""

import argparse
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from tqdm import tqdm

from .config import ActionVideoConfig
from .dataset import LiberoVideoDataset, collate_fn
from .model import ActionConditionedVace

# Import schedulers from wan without triggering wan/__init__.py
import importlib.util as _ilu

_WAN_DIR = Path(__file__).resolve().parent.parent / "wan"


def _load_wan_util(name, filename):
    fullname = f"wan.utils.{name}"
    if fullname in sys.modules:
        return sys.modules[fullname]
    # Ensure wan.utils namespace exists
    for ns in ["wan", "wan.utils"]:
        if ns not in sys.modules:
            import types
            pkg = types.ModuleType(ns)
            pkg.__path__ = [str(_WAN_DIR if ns == "wan" else _WAN_DIR / "utils")]
            pkg.__package__ = ns
            sys.modules[ns] = pkg
    spec = _ilu.spec_from_file_location(fullname, _WAN_DIR / "utils" / filename)
    mod = _ilu.module_from_spec(spec)
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)
    return mod


_fm_unipc = _load_wan_util("fm_solvers_unipc", "fm_solvers_unipc.py")
_fm_solvers = _load_wan_util("fm_solvers", "fm_solvers.py")
FlowUniPCMultistepScheduler = _fm_unipc.FlowUniPCMultistepScheduler
FlowDPMSolverMultistepScheduler = _fm_solvers.FlowDPMSolverMultistepScheduler
get_sampling_sigmas = _fm_solvers.get_sampling_sigmas
retrieve_timesteps = _fm_solvers.retrieve_timesteps

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ActionConditionedVace")

    # Model
    parser.add_argument("--checkpoint_dir", type=str, default="Wan2.1-VACE-1.3B")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pt)")

    # Sampling
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--sampling_steps", type=int, default=30)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vace_context_scale", type=float, default=1.0)

    # Output
    parser.add_argument("--output_dir", type=str, default="eval_outputs")
    parser.add_argument("--save_grid", action="store_true",
                        help="Save side-by-side comparison grids")
    parser.add_argument("--fps", type=int, default=8)

    # Data overrides
    parser.add_argument("--sample_indices", type=str, default=None,
                        help="Comma-separated sample indices to evaluate")

    return parser.parse_args()


def load_model(args) -> tuple[ActionConditionedVace, ActionVideoConfig]:
    """Load model with trained checkpoint."""
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_dict = ckpt.get("config", {})

    config = ActionVideoConfig()
    for key, value in config_dict.items():
        if hasattr(config, key) and not isinstance(value, dict):
            try:
                setattr(config, key, value)
            except (TypeError, AttributeError):
                pass

    config.checkpoint_dir = args.checkpoint_dir

    logger.info("Loading model...")
    model = ActionConditionedVace(config, device_id=0)

    # Load trained weights
    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=False
    )
    logger.info(
        f"Loaded checkpoint step {ckpt.get('step', '?')}: "
        f"{len(missing)} missing, {len(unexpected)} unexpected keys"
    )

    model.eval()
    return model, config


@torch.no_grad()
def generate_video(
    model: ActionConditionedVace,
    config: ActionVideoConfig,
    history_frames: torch.Tensor,
    mask: torch.Tensor,
    actions: torch.Tensor,
    task: str,
    sampling_steps: int = 30,
    guide_scale: float = 5.0,
    shift: float = 5.0,
    solver: str = "unipc",
    seed: int = 42,
    vace_context_scale: float = 1.0,
) -> torch.Tensor:
    """Run diffusion sampling to generate future frames.

    Args:
        model: trained ActionConditionedVace
        config: model config
        history_frames: [3, T, H, W] full video (history + placeholder future)
        mask: [1, T, H, W] binary mask (0=history, 1=generate)
        actions: [T_action, action_dim] future actions
        task: task description string
        sampling_steps: number of diffusion steps
        guide_scale: classifier-free guidance scale
        shift: noise schedule shift
        solver: "unipc" or "dpm++"
        seed: random seed
        vace_context_scale: VACE context strength

    Returns:
        generated: [3, T, H, W] generated video tensor in [-1, 1]
    """
    device = model.device

    # Zero out future frames so the model cannot look ahead.
    # reactive channel will encode zeros for the masked region.
    history_frames = history_frames * (1 - mask)

    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    # Encode text
    context = model.encode_text([task])
    context_null = model.encode_text([""])

    # Encode VACE context (history frames + masks)
    frames_list = [history_frames.to(device)]
    masks_list = [mask.to(device)]
    vace_ctx = model.vace_encode_frames(frames_list, masks_list)

    # Compute target latent shape from VACE context
    # vace_ctx is [96, F_lat, H_lat, W_lat]; target is [16, F_lat, H_lat, W_lat]
    target_shape = list(vace_ctx[0].shape)
    target_shape[0] = target_shape[0] // 6  # 96 -> 16 (VAE latent channels)

    # Initial noise
    noise = torch.randn(
        *target_shape, dtype=torch.float32, device=device, generator=seed_g
    )
    latents = [noise]

    # Sequence length
    seq_len = math.ceil(
        (target_shape[2] * target_shape[3])
        / (config.patch_size[1] * config.patch_size[2])
        * target_shape[1]
    )

    # Actions
    actions_batch = actions.unsqueeze(0).to(device)

    # Setup scheduler
    if solver == "unipc":
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
        timesteps = scheduler.timesteps
    elif solver == "dpm++":
        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
        timesteps, _ = retrieve_timesteps(
            scheduler, device=device, sigmas=sampling_sigmas
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Diffusion sampling loop
    with amp.autocast(dtype=config.param_dtype):
        for t in tqdm(timesteps, desc="Sampling", leave=False):
            timestep = torch.tensor([t], device=device)

            # Conditional prediction
            pred_cond = model.forward_with_actions(
                x=latents,
                t=timestep,
                vace_context=vace_ctx,
                actions=actions_batch,
                context=context,
                seq_len=seq_len,
                vace_context_scale=vace_context_scale,
            )[0]

            # Unconditional prediction (zero actions, null text)
            zero_actions = torch.zeros_like(actions_batch)
            pred_uncond = model.forward_with_actions(
                x=latents,
                t=timestep,
                vace_context=vace_ctx,
                actions=zero_actions,
                context=context_null,
                seq_len=seq_len,
                vace_context_scale=vace_context_scale,
            )[0]

            # CFG
            noise_pred = pred_uncond + guide_scale * (pred_cond - pred_uncond)

            # Scheduler step
            temp_x0 = scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latents = [temp_x0.squeeze(0)]

    # Decode latents to video
    videos = model.vae.decode(latents)
    return videos[0]  # [3, T, H, W]


def _tensor_to_pil(tensor: torch.Tensor):
    """Convert a [3, H, W] float tensor in [0,1] to a PIL Image."""
    from PIL import Image
    arr = (tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def save_video_as_frames(video: torch.Tensor, output_dir: Path, prefix: str):
    """Save video tensor as individual PNG frames."""
    frames_dir = output_dir / f"{prefix}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # video: [3, T, H, W] in [-1, 1]
    video = (video.clamp(-1, 1) + 1) / 2  # to [0, 1]
    T = video.shape[1]
    for t in range(T):
        _tensor_to_pil(video[:, t]).save(frames_dir / f"frame_{t:04d}.png")


def save_video_as_mp4(video: torch.Tensor, path: Path, fps: int = 8):
    """Save video tensor as mp4 using imageio."""
    try:
        import imageio.v3 as iio
    except ImportError:
        logger.warning("imageio not available, falling back to frame saving")
        save_video_as_frames(video, path.parent, path.stem)
        return

    # video: [3, T, H, W] in [-1, 1] -> [T, H, W, 3] uint8
    video = (video.clamp(-1, 1) + 1) / 2 * 255
    video = video.permute(1, 2, 3, 0).cpu().to(torch.uint8).numpy()
    iio.imwrite(str(path), video, fps=fps)


def save_comparison_grid(
    gt_video: torch.Tensor,
    gen_video: torch.Tensor,
    mask: torch.Tensor,
    output_path: Path,
):
    """Save a side-by-side comparison image: GT vs generated, with mask overlay."""
    from PIL import Image

    # Both [3, T, H, W] in [-1, 1]
    gt = (gt_video.clamp(-1, 1) + 1) / 2
    gen = (gen_video.clamp(-1, 1) + 1) / 2

    T = min(gt.shape[1], gen.shape[1])
    num_cols = min(T, 8)
    indices = torch.linspace(0, T - 1, num_cols).long()

    gt_frames = [gt[:, i] for i in indices]   # list of [3, H, W]
    gen_frames = [gen[:, i] for i in indices]

    # Build grid: top row GT, bottom row generated
    _, H, W = gt_frames[0].shape
    pad = 2
    grid_w = num_cols * W + (num_cols + 1) * pad
    grid_h = 2 * H + 3 * pad
    grid = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))

    for col, (gt_f, gen_f) in enumerate(zip(gt_frames, gen_frames)):
        x = pad + col * (W + pad)
        grid.paste(_tensor_to_pil(gt_f), (x, pad))
        grid.paste(_tensor_to_pil(gen_f), (x, 2 * pad + H))

    grid.save(output_path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config = load_model(args)

    # Create dataset
    logger.info("Loading dataset...")
    dataset = LiberoVideoDataset(config)

    # Select sample indices
    if args.sample_indices:
        indices = [int(x) for x in args.sample_indices.split(",")]
    else:
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    logger.info(f"Evaluating {len(indices)} samples: {indices}")

    for i, idx in enumerate(indices):
        logger.info(f"Sample {i + 1}/{len(indices)} (dataset index {idx})")

        sample = dataset[idx]
        video = sample["video"]    # [3, T, H, W]
        mask = sample["mask"]      # [1, T, H, W]
        actions = sample["actions"]  # [T_future, action_dim]
        task = sample["task"]

        if actions is None:
            actions = torch.zeros(config.num_future_frames, 7)

        # Generate
        gen_video = generate_video(
            model=model,
            config=config,
            history_frames=video,
            mask=mask,
            actions=actions,
            task=task,
            sampling_steps=args.sampling_steps,
            guide_scale=args.guide_scale,
            shift=args.shift,
            solver=args.solver,
            seed=args.seed + i,
            vace_context_scale=args.vace_context_scale,
        )

        # Save generated video
        prefix = f"sample_{idx:06d}"
        save_video_as_mp4(gen_video, output_dir / f"{prefix}_gen.mp4", fps=args.fps)

        # Save ground truth video
        save_video_as_mp4(video, output_dir / f"{prefix}_gt.mp4", fps=args.fps)

        # Save comparison grid
        if args.save_grid:
            save_comparison_grid(
                video, gen_video, mask,
                output_dir / f"{prefix}_comparison.png",
            )

        # Save metadata
        meta_path = output_dir / f"{prefix}_meta.txt"
        meta_path.write_text(
            f"dataset_index: {idx}\n"
            f"task: {task}\n"
            f"actions_shape: {list(actions.shape)}\n"
            f"seed: {args.seed + i}\n"
            f"sampling_steps: {args.sampling_steps}\n"
            f"guide_scale: {args.guide_scale}\n"
            f"solver: {args.solver}\n"
        )

        logger.info(f"  Saved to {output_dir / prefix}*")

        torch.cuda.empty_cache()

    logger.info(f"Done. Results in {output_dir}")


if __name__ == "__main__":
    main()
