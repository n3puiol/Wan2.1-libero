"""
Compute flow matching eval loss on a held-out LeRobot dataset.

Uses the same loss as training (MSE on predicted velocity) but evaluated
on `n3puiol/libero_cosmos_policy_lerobot` to measure generalisation.

Usage:
    python -m action_video.eval_loss \
        --checkpoint checkpoints/action_vace_libero/checkpoint_latest.pt \
        --num_samples 50 \
        --num_noise_samples 5
"""

import argparse
import dataclasses
import json
import logging
import random
from pathlib import Path

import torch
import torch.nn.functional as F
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

EVAL_REPO_ID = "n3puiol/libero_cosmos_policy_lerobot"


def parse_args():
    parser = argparse.ArgumentParser(description="Compute eval loss")

    # Model
    parser.add_argument("--checkpoint_dir", type=str, default="Wan2.1-VACE-1.3B")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pt)")

    # Eval
    parser.add_argument("--eval_repo_id", type=str, default=EVAL_REPO_ID,
                        help="HuggingFace repo id for eval dataset")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of dataset samples to evaluate")
    parser.add_argument("--num_noise_samples", type=int, default=5,
                        help="Noise draws per sample (averaged for stable estimate)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_dir", type=str, default="eval_outputs")

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
def compute_flow_matching_loss(
    model: ActionConditionedVace,
    video: torch.Tensor,
    mask: torch.Tensor,
    actions: torch.Tensor,
    tasks: list[str],
    num_noise_samples: int = 5,
) -> dict:
    """Compute flow matching loss averaged over multiple noise draws.

    Args:
        model: trained ActionConditionedVace in eval mode
        video: [B, 3, T, H, W]
        mask: [B, 1, T, H, W]
        actions: [B, T_action, action_dim]
        tasks: list of task strings
        num_noise_samples: number of (sigma, noise) draws to average over

    Returns:
        dict with 'loss' (scalar), 'losses_per_draw' (list)
    """
    device = model.device
    B = video.shape[0]

    # Encode text once
    text_context = model.encode_text(tasks)

    # Encode VACE context (with future zeroed out)
    masked_video = video * (1 - mask)
    masked_list = [masked_video[i].to(device) for i in range(B)]
    masks_list = [mask[i].to(device) for i in range(B)]
    vace_ctx = model.vace_encode_frames(masked_list, masks_list)

    # Encode full video as target latents
    full_list = [video[i].to(device) for i in range(B)]
    target_latents = model.vae.encode(full_list)

    seq_len = model.compute_seq_len(target_latents[0].shape)

    losses = []
    for _ in range(num_noise_samples):
        sigma = torch.rand(B, device=device)
        noise = [torch.randn_like(z) for z in target_latents]

        x_t = []
        for i in range(B):
            si = sigma[i].view(1, 1, 1, 1)
            x_t.append((1 - si) * target_latents[i] + si * noise[i])

        t_scaled = sigma * model.config.num_train_timesteps

        with torch.amp.autocast("cuda", dtype=model.config.param_dtype):
            predicted = model.forward_with_actions(
                x=x_t,
                t=t_scaled,
                vace_context=vace_ctx,
                actions=actions.to(device),
                context=text_context,
                seq_len=seq_len,
            )

        target_velocity = [n - z for z, n in zip(target_latents, noise)]

        loss = sum(
            F.mse_loss(pred.float(), tgt.float())
            for pred, tgt in zip(predicted, target_velocity)
        ) / B

        losses.append(loss.item())

    return {
        "loss": sum(losses) / len(losses),
        "losses_per_draw": losses,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model, config = load_model(args)

    # Create eval dataset with the held-out repo
    eval_config = dataclasses.replace(config, repo_id=args.eval_repo_id)
    logger.info(f"Loading eval dataset: {args.eval_repo_id}")
    eval_dataset = LiberoVideoDataset(eval_config)

    # Select sample indices
    rng = random.Random(args.seed)
    num_samples = min(args.num_samples, len(eval_dataset))
    indices = rng.sample(range(len(eval_dataset)), num_samples)

    logger.info(f"Evaluating {num_samples} samples with {args.num_noise_samples} noise draws each")

    all_losses = []
    per_sample = []

    for i, idx in enumerate(tqdm(indices, desc="Eval loss")):
        sample = eval_dataset[idx]
        video = sample["video"].unsqueeze(0)     # [1, 3, T, H, W]
        mask = sample["mask"].unsqueeze(0)        # [1, 1, T, H, W]
        task = sample["task"]

        actions = sample["actions"]
        if actions is None:
            actions = torch.zeros(config.num_future_frames, 7)
        actions = actions.unsqueeze(0)            # [1, T_action, action_dim]

        result = compute_flow_matching_loss(
            model, video, mask, actions, [task],
            num_noise_samples=args.num_noise_samples,
        )

        all_losses.append(result["loss"])
        per_sample.append({
            "index": idx,
            "task": task,
            "loss": result["loss"],
            "losses_per_draw": result["losses_per_draw"],
        })

        if (i + 1) % 10 == 0:
            running_mean = sum(all_losses) / len(all_losses)
            logger.info(f"  [{i+1}/{num_samples}] running eval loss: {running_mean:.6f}")

        torch.cuda.empty_cache()

    mean_loss = sum(all_losses) / len(all_losses)
    std_loss = (sum((l - mean_loss) ** 2 for l in all_losses) / len(all_losses)) ** 0.5

    logger.info(f"Eval loss: {mean_loss:.6f} +/- {std_loss:.6f} ({num_samples} samples)")

    # Save results
    results = {
        "eval_repo_id": args.eval_repo_id,
        "checkpoint": args.checkpoint,
        "num_samples": num_samples,
        "num_noise_samples": args.num_noise_samples,
        "seed": args.seed,
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "per_sample": per_sample,
    }
    results_path = output_dir / "eval_loss.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
