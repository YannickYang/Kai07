"""Training script for pi0.7 High-Level Policy (subtask text generation).

Trains the HLP to predict subtask instructions given observations.
Uses PaliGemma backbone initialized from pi0.5 base weights.

Usage:
    python scripts/train_high_level_policy.py \
        --data-dir /path/to/fold_clothes_dataset \
        --subtask-annotations /path/to/subtask_annotations.json \
        --task-prompt "Flatten and fold the cloth" \
        --base-weights /path/to/pi05_base/params

    # Resume training:
    python scripts/train_high_level_policy.py \
        --data-dir /path/to/fold_clothes_dataset \
        --subtask-annotations /path/to/subtask_annotations.json \
        --task-prompt "Flatten and fold the cloth" \
        --checkpoint-dir /path/to/hlp_checkpoints \
        --resume
"""

import argparse
import json
import logging
import os
import random
import time

import cv2
import numpy as np
import safetensors.torch
import sentencepiece
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class SubtaskDataset(Dataset):
    """Dataset for high-level policy training.

    Each sample is a subtask boundary:
        Input: observation images + task prompt + previous subtask history
        Target: current subtask text

    Loads frames from LeRobot v2.1 video format.
    """

    def __init__(
        self,
        data_dir: str,
        subtask_annotations: dict,
        task_prompt: str,
        cameras: list[str],
        image_size: int = 448,
        max_prompt_len: int = 128,
        max_target_len: int = 32,
        tokenizer_path: str | None = None,
    ):
        self.data_dir = data_dir
        self.task_prompt = task_prompt
        self.cameras = cameras
        self.image_size = image_size
        self.max_prompt_len = max_prompt_len
        self.max_target_len = max_target_len

        # Load tokenizer (PaliGemma SentencePiece)
        if tokenizer_path is None:
            tokenizer_path = os.path.join(
                os.path.dirname(__file__), "..", "src", "openpi", "assets", "paligemma_tokenizer.model"
            )
        self.tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

        # Build training samples: one per subtask boundary
        self.samples = []
        for ep_key, subtasks in subtask_annotations.items():
            ep_idx = int(ep_key)
            for st_idx, subtask in enumerate(subtasks):
                # Previous subtask history
                history = [subtasks[i]["text"] for i in range(st_idx)]
                self.samples.append({
                    "episode_idx": ep_idx,
                    "frame_idx": subtask["start"],
                    "history": history,
                    "target_text": subtask["text"],
                })

        logger.info(f"Created {len(self.samples)} training samples from {len(subtask_annotations)} episodes")

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, episode_idx: int, frame_idx: int, camera: str) -> np.ndarray | None:
        """Load a single frame from video."""
        chunk_idx = episode_idx // 1000
        video_path = os.path.join(
            self.data_dir, "videos", f"chunk-{chunk_idx:03d}", camera, f"episode_{episode_idx:06d}.mp4"
        )
        if not os.path.exists(video_path):
            return None

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            return frame
        return None

    def _tokenize(self, text: str, max_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize text and pad/truncate to max_len."""
        token_ids = self.tokenizer.encode(text)
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]

        mask = [True] * len(token_ids) + [False] * (max_len - len(token_ids))
        token_ids = token_ids + [0] * (max_len - len(token_ids))

        return np.array(token_ids, dtype=np.int64), np.array(mask, dtype=bool)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load observation images
        images = {}
        for camera in self.cameras:
            frame = self._load_frame(sample["episode_idx"], sample["frame_idx"], camera)
            if frame is not None:
                # Normalize to [0, 1] and convert to CHW
                img = frame.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                images[camera] = img
            else:
                images[camera] = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)

        # Build prompt
        history_str = ""
        if sample["history"]:
            history_str = " Previous: " + "; ".join(sample["history"]) + "."

        prompt = f"Task: {self.task_prompt}.{history_str} Next subtask:"
        target = " " + sample["target_text"]

        # Tokenize
        prompt_tokens, prompt_mask = self._tokenize(prompt, self.max_prompt_len)
        target_tokens, target_mask = self._tokenize(target, self.max_target_len)

        return {
            "images": images,
            "prompt_tokens": prompt_tokens,
            "prompt_mask": prompt_mask,
            "target_tokens": target_tokens,
            "target_mask": target_mask,
        }


def collate_fn(batch):
    """Custom collate for SubtaskDataset."""
    cameras = list(batch[0]["images"].keys())
    images = {cam: torch.tensor(np.stack([b["images"][cam] for b in batch])) for cam in cameras}
    return {
        "images": images,
        "prompt_tokens": torch.tensor(np.stack([b["prompt_tokens"] for b in batch])),
        "prompt_mask": torch.tensor(np.stack([b["prompt_mask"] for b in batch])),
        "target_tokens": torch.tensor(np.stack([b["target_tokens"] for b in batch])),
        "target_mask": torch.tensor(np.stack([b["target_mask"] for b in batch])),
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load annotations
    with open(args.subtask_annotations) as f:
        subtask_annotations = json.load(f)
    logger.info(f"Loaded annotations for {len(subtask_annotations)} episodes")

    # Create dataset
    dataset = SubtaskDataset(
        data_dir=args.data_dir,
        subtask_annotations=subtask_annotations,
        task_prompt=args.task_prompt,
        cameras=args.cameras.split(","),
        image_size=args.image_size,
        max_prompt_len=args.max_prompt_len,
        max_target_len=args.max_target_len,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Create model
    from types import SimpleNamespace

    model_config = SimpleNamespace(
        paligemma_variant="gemma_2b",
        hlp_freeze_vision=args.freeze_vision,
        hlp_max_generate_len=64,
    )

    from openpi.models_pytorch.high_level_policy import HighLevelPolicy

    model = HighLevelPolicy(model_config).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Load base weights (from pi0.5 PaliGemma)
    if args.base_weights:
        logger.info(f"Loading base weights from {args.base_weights}")
        model_path = os.path.join(args.base_weights, "model.safetensors")
        if os.path.exists(model_path):
            safetensors.torch.load_model(model, model_path, strict=False)
            logger.info("Loaded base VLM weights (strict=False)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LR schedule: linear warmup + cosine decay
    def lr_schedule(step):
        if step < args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.num_steps - args.warmup_steps)
        return args.lr * 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))

    # Resume
    global_step = 0
    if args.resume and args.checkpoint_dir:
        ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            global_step = ckpt["step"]
            logger.info(f"Resumed from step {global_step}")

    # Checkpoint directory
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    model.train()
    start_time = time.time()
    losses = []

    logger.info(f"Training for {args.num_steps} steps, batch_size={args.batch_size}")

    while global_step < args.num_steps:
        for batch in dataloader:
            if global_step >= args.num_steps:
                break

            # Move to device
            images_list = [batch["images"][cam].to(device) for cam in batch["images"]]
            image_masks = [torch.ones(args.batch_size, dtype=torch.bool, device=device)] * len(images_list)
            prompt_tokens = batch["prompt_tokens"].to(device)
            prompt_mask = batch["prompt_mask"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            target_mask = batch["target_mask"].to(device)

            # Update LR
            for pg in optimizer.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward
            loss = model(
                images_list, image_masks,
                prompt_tokens, prompt_mask,
                target_tokens, target_mask,
            )

            # Backward
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())

            # Logging
            if global_step % args.log_interval == 0 and global_step > 0:
                avg_loss = sum(losses) / len(losses)
                elapsed = time.time() - start_time
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={lr:.2e} "
                    f"grad_norm={grad_norm:.2f} time={elapsed:.1f}s"
                )
                losses = []
                start_time = time.time()

            # Checkpoint
            if args.checkpoint_dir and global_step % args.save_interval == 0 and global_step > 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                }
                torch.save(ckpt, os.path.join(args.checkpoint_dir, "latest.pt"))
                safetensors.torch.save_model(
                    model, os.path.join(args.checkpoint_dir, f"model_{global_step}.safetensors")
                )
                logger.info(f"Saved checkpoint at step {global_step}")

            global_step += 1

    logger.info(f"Training complete. Final step: {global_step}")


def main():
    parser = argparse.ArgumentParser(description="Train pi0.7 High-Level Policy")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to LeRobot dataset")
    parser.add_argument("--subtask-annotations", type=str, required=True, help="Path to subtask_annotations.json")
    parser.add_argument("--task-prompt", type=str, required=True, help="Task instruction text")
    parser.add_argument("--base-weights", type=str, default=None, help="Path to pi0.5 base weights")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/hlp", help="Checkpoint directory")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")

    parser.add_argument("--cameras", type=str, default="observation.images.head,observation.images.left,observation.images.right")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--max-prompt-len", type=int, default=128)
    parser.add_argument("--max-target-len", type=int, default=32)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-vision", action="store_true", default=True)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
