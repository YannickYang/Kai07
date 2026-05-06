#!/usr/bin/env python3
"""Fine-tune BAGEL-7B-MoT as a world model for pi0.7.

Trains BAGEL to predict subgoal images given current observation + subtask text,
using flow matching (velocity prediction) loss on VAE latents.

Usage:
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        scripts/finetune_bagel_world_model.py \
        --model_path /path/to/BAGEL-7B-MoT \
        --data_dir /path/to/dataset \
        --output_dir checkpoints/bagel_wm \
        --total_steps 5000
"""

import argparse
import functools
import gc
import logging
import os
import shutil
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper,
)
from torch.utils.data import DataLoader

BAGEL_DIR = os.environ.get("BAGEL_DIR", os.path.join(os.path.dirname(__file__), "../../Bagel"))
if os.path.isdir(BAGEL_DIR):
    sys.path.insert(0, BAGEL_DIR)

from data.data_utils import (
    add_special_tokens, patchify, prepare_attention_mask_per_sample,
    get_flattened_position_ids_extrapolate,
)
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper,
    fsdp_ema_setup, fsdp_ema_update,
)
from train.train_utils import get_latest_ckpt

import importlib.util
_rds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "robot_edit_dataset.py")
_rds_spec = importlib.util.spec_from_file_location("robot_edit_dataset", _rds_path)
_rds_mod = importlib.util.module_from_spec(_rds_spec)
_rds_spec.loader.exec_module(_rds_mod)
RobotEditDataset = _rds_mod.RobotEditDataset

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BAGEL as world model for pi0.7")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_latent_size", type=int, default=32)
    parser.add_argument("--latent_patch_size", type=int, default=2)
    parser.add_argument("--vit_max_num_patch_per_side", type=int, default=70)
    parser.add_argument("--vit_patch_size", type=int, default=14)
    parser.add_argument("--connector_act", type=str, default="gelu_pytorch_tanh")
    parser.add_argument("--timestep_shift", type=float, default=1.0)

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--subtask_annotations", type=str, default=None)
    parser.add_argument("--cameras", type=str, default="head,left,right")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--output_dir", type=str, default="checkpoints/bagel_wm")
    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--freeze_vit", action="store_true", default=True)
    parser.add_argument("--no_freeze_vit", dest="freeze_vit", action="store_false")
    parser.add_argument("--freeze_llm", action="store_true", default=False)

    parser.add_argument("--text_cond_dropout", type=float, default=0.1)
    parser.add_argument("--vae_cond_dropout", type=float, default=0.3)
    parser.add_argument("--vit_cond_dropout", type=float, default=0.3)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="bagel_world_model")

    return parser.parse_args()


def build_model(args, device):
    model_path = args.model_path

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 - 1
    vit_config.rope = True

    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        latent_patch_size=args.latent_patch_size,
        max_latent_size=args.max_latent_size,
        vit_max_num_patch_per_side=args.vit_max_num_patch_per_side,
        connector_act=args.connector_act,
        interpolate_pos=False,
        timestep_shift=args.timestep_shift,
    )
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    for param in vae_model.parameters():
        param.requires_grad = False
    if args.freeze_vit:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False
    if args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params/1e9:.2f}B total, {trainable_params/1e9:.2f}B trainable")

    return model, vae_model, vae_config, tokenizer, new_token_ids, config


def collate_fn(batch):
    return batch


def pack_robot_edit_samples(samples, tokenizer, new_token_ids, args, vae_downsample):
    """Pack robot editing samples into BAGEL's native edit format.

    Follows UnifiedEditIterableDataset exactly:
      1) vae_source (loss=0, cfg=1) — obs latent as clean conditioning
      2) vit_source (cfg=1)         — obs ViT features
      3) text      (cfg=1)          — edit instruction
      4) vae_target (loss=1)        — target latent, flow matching loss

    CFG dropout randomly drops conditioning tokens during training.
    """
    import random as _random
    import torchvision.transforms as T

    vit_patch_size = args.vit_patch_size
    latent_patch_size = args.latent_patch_size
    vae_image_downsample = latent_patch_size * vae_downsample
    max_latent_size = args.max_latent_size
    max_num_patch_per_side = args.vit_max_num_patch_per_side

    bos_token_id = new_token_ids['bos_token_id']
    eos_token_id = new_token_ids['eos_token_id']
    start_of_image = new_token_ids['start_of_image']
    end_of_image = new_token_ids['end_of_image']

    vit_size = (args.image_size // vit_patch_size) * vit_patch_size
    vae_size = (args.image_size // vae_image_downsample) * vae_image_downsample

    vit_transform = T.Compose([
        T.Resize((vit_size, vit_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    vae_transform = T.Compose([
        T.Resize((vae_size, vae_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    curr = 0
    sample_lens_list = []
    packed_position_ids = []
    packed_text_ids = []
    packed_text_indexes = []
    packed_vit_tokens_list = []
    vit_token_seqlens = []
    packed_vit_position_ids_list = []
    packed_vit_token_indexes = []
    vae_image_tensors = []
    vae_latent_shapes = []
    packed_latent_position_ids_list = []
    packed_vae_token_indexes = []
    packed_timesteps = []
    mse_loss_indexes = []
    nested_attention_masks = []

    def _add_vae_image(img_tensor, loss, enable_cfg):
        """Add a VAE image segment. Returns True if actually added (not dropped by CFG)."""
        nonlocal curr
        if enable_cfg and _random.random() < args.vae_cond_dropout:
            return False  # dropped — caller handles rope_id

        split_len = 0
        # <vision_start>
        packed_text_ids.append(start_of_image)
        packed_text_indexes.append(curr)
        curr += 1
        split_len += 1

        vae_image_tensors.append(img_tensor)
        H, W = img_tensor.shape[1], img_tensor.shape[2]
        h = H // vae_image_downsample
        w = W // vae_image_downsample
        vae_latent_shapes.append((h, w))
        packed_latent_position_ids_list.append(
            get_flattened_position_ids_extrapolate(H, W, vae_image_downsample, max_latent_size)
        )
        num_tokens = h * w
        packed_vae_token_indexes.extend(range(curr, curr + num_tokens))

        if loss:
            mse_loss_indexes.extend(range(curr, curr + num_tokens))
            timestep = np.random.randn()
        else:
            timestep = float('-inf')  # sigmoid(-inf)=0 → clean latent, no noise
        packed_timesteps.extend([timestep] * num_tokens)

        curr += num_tokens
        split_len += num_tokens

        # <vision_end>
        packed_text_ids.append(end_of_image)
        packed_text_indexes.append(curr)
        curr += 1
        split_len += 1

        return split_len

    def _add_vit_image(img_tensor, enable_cfg):
        """Add a ViT image segment. Returns split_len or False if dropped."""
        nonlocal curr
        if enable_cfg and _random.random() < args.vit_cond_dropout:
            return False

        split_len = 0
        packed_text_ids.append(start_of_image)
        packed_text_indexes.append(curr)
        curr += 1
        split_len += 1

        vit_tokens = patchify(img_tensor, vit_patch_size)
        num_tokens = vit_tokens.shape[0]
        packed_vit_token_indexes.extend(range(curr, curr + num_tokens))
        packed_vit_tokens_list.append(vit_tokens)
        vit_token_seqlens.append(num_tokens)
        packed_vit_position_ids_list.append(
            get_flattened_position_ids_extrapolate(
                img_tensor.shape[1], img_tensor.shape[2],
                vit_patch_size, max_num_patch_per_side
            )
        )
        curr += num_tokens
        split_len += num_tokens

        packed_text_ids.append(end_of_image)
        packed_text_indexes.append(curr)
        curr += 1
        split_len += 1

        return split_len

    def _add_text(text_ids_encoded, enable_cfg):
        """Add a text segment. Returns split_len or False if dropped."""
        nonlocal curr
        if enable_cfg and _random.random() < args.text_cond_dropout:
            return False

        shifted = [bos_token_id] + text_ids_encoded
        packed_text_ids.extend(shifted)
        packed_text_indexes.extend(range(curr, curr + len(shifted)))
        curr += len(shifted)

        packed_text_ids.append(eos_token_id)
        packed_text_indexes.append(curr)
        curr += 1

        return len(shifted) + 1

    for s in samples:
        if s is None:
            continue

        obs_img = s["obs_image"]
        target_img = s["target_image"]
        text = s["text"]

        prompt = f"Edit the image to show the result after: {text}"
        text_ids = tokenizer.encode(prompt)
        obs_vit = vit_transform(obs_img)
        obs_vae = vae_transform(obs_img)
        target_vae = vae_transform(target_img)

        split_lens = []
        attn_modes = []
        curr_rope_id = 0
        sample_start = curr

        # --- 1) Source observation as VAE conditioning (loss=0, full attn) ---
        vae_src_len = _add_vae_image(obs_vae, loss=False, enable_cfg=True)
        if vae_src_len:
            split_lens.append(vae_src_len)
            attn_modes.append("full")
            packed_position_ids.extend([curr_rope_id] * vae_src_len)
            curr_rope_id += 1  # loss=0 → increment
        else:
            curr_rope_id += 1  # CFG dropped but still advance rope_id

        # --- 2) Source observation as ViT (full attn) ---
        vit_len = _add_vit_image(obs_vit, enable_cfg=True)
        if vit_len:
            split_lens.append(vit_len)
            attn_modes.append("full")
            packed_position_ids.extend([curr_rope_id] * vit_len)
            curr_rope_id += 1
        else:
            curr_rope_id += 1  # CFG dropped

        # --- 3) Text instruction (causal attn) ---
        text_len = _add_text(text_ids, enable_cfg=True)
        if text_len:
            split_lens.append(text_len)
            attn_modes.append("causal")
            packed_position_ids.extend(range(curr_rope_id, curr_rope_id + text_len))
            curr_rope_id += text_len
        # text CFG drop: rope_id does NOT advance (matches original)

        # --- 4) Target image as VAE (loss=1, noise attn) ---
        vae_tgt_len = _add_vae_image(target_vae, loss=True, enable_cfg=False)
        split_lens.append(vae_tgt_len)
        attn_modes.append("noise")
        packed_position_ids.extend([curr_rope_id] * vae_tgt_len)
        # loss=1 → do NOT increment curr_rope_id (matches original)

        sample_len = curr - sample_start
        sample_lens_list.append(sample_len)
        nested_attention_masks.append(
            prepare_attention_mask_per_sample(split_lens, attn_modes)
        )

    if not vae_image_tensors:
        return None

    image_sizes = [t.shape for t in vae_image_tensors]
    max_size = [max(s) for s in zip(*image_sizes)]
    padded_images = torch.zeros(len(vae_image_tensors), *max_size)
    for i, t in enumerate(vae_image_tensors):
        padded_images[i, :, :t.shape[1], :t.shape[2]] = t

    data = {
        "sequence_length": curr,
        "sample_lens": sample_lens_list,
        "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
        "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
        "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
        "nested_attention_masks": nested_attention_masks,
        "packed_vit_tokens": torch.cat(packed_vit_tokens_list, dim=0) if packed_vit_tokens_list else None,
        "packed_vit_position_ids": torch.cat(packed_vit_position_ids_list, dim=0) if packed_vit_position_ids_list else None,
        "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long) if packed_vit_token_indexes else None,
        "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int) if vit_token_seqlens else None,
        "padded_images": padded_images,
        "patchified_vae_latent_shapes": vae_latent_shapes,
        "packed_latent_position_ids": torch.cat(packed_latent_position_ids_list, dim=0),
        "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
        "packed_timesteps": torch.tensor(packed_timesteps, dtype=torch.float),
        "mse_loss_indexes": torch.tensor(mse_loss_indexes, dtype=torch.long),
    }
    return data


def move_data_to_device(data, device):
    data["packed_text_ids"] = data["packed_text_ids"].to(device)
    data["packed_text_indexes"] = data["packed_text_indexes"].to(device)
    data["packed_position_ids"] = data["packed_position_ids"].to(device)
    data["nested_attention_masks"] = [m.to(device) for m in data["nested_attention_masks"]]
    if data["packed_vit_tokens"] is not None:
        data["packed_vit_tokens"] = data["packed_vit_tokens"].to(device)
        data["packed_vit_position_ids"] = data["packed_vit_position_ids"].to(device)
        data["packed_vit_token_indexes"] = data["packed_vit_token_indexes"].to(device)
        data["vit_token_seqlens"] = data["vit_token_seqlens"].to(device)
    data["padded_images"] = data["padded_images"].to(device)
    data["packed_latent_position_ids"] = data["packed_latent_position_ids"].to(device)
    data["packed_vae_token_indexes"] = data["packed_vae_token_indexes"].to(device)
    data["packed_timesteps"] = data["packed_timesteps"].to(device)
    data["mse_loss_indexes"] = data["mse_loss_indexes"].to(device)
    return data


def main():
    args = parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if rank == 0:
        logger.info(f"Fine-tuning BAGEL world model")
        logger.info(f"  Model: {args.model_path}")
        logger.info(f"  Data: {args.data_dir}")
        logger.info(f"  Output: {args.output_dir}")
        logger.info(f"  GPUs: {world_size}")
        logger.info(f"  Steps: {args.total_steps}")
        logger.info(f"  LR: {args.lr}")

    if args.wandb and rank == 0:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    import time as _t
    _t0 = _t.time()
    logger.info("Building model...")
    model, vae_model, vae_config, tokenizer, new_token_ids, config = build_model(args, device)
    vae_downsample = vae_config.downsample
    logger.info(f"  build_model done in {_t.time()-_t0:.1f}s")

    fsdp_config = FSDPConfig(
        sharding_strategy="FULL_SHARD",
        backward_prefetch="BACKWARD_PRE",
        cpu_offload=False,
        num_replicate=1,
    )

    resume_from = args.model_path
    if args.auto_resume:
        latest = get_latest_ckpt(args.output_dir)
        if latest:
            resume_from = latest
            logger.info(f"Resuming from {resume_from}")

    model, _ = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model=None, resume_from_ema=True
    )
    _t1 = _t.time()
    logger.info("Deepcopy model -> ema_model...")
    ema_model = deepcopy(model).cpu()
    model.cpu()
    logger.info(f"  deepcopy done in {_t.time()-_t1:.1f}s")

    _t2 = _t.time()
    logger.info("Setting up FSDP...")
    ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    fsdp_model = fsdp_wrapper(model, fsdp_config)
    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=grad_checkpoint_check_fn,
    )
    logger.info(f"  FSDP setup done in {_t.time()-_t2:.1f}s")

    vae_model.to(device).eval()
    fsdp_model.train()

    # Dataset
    subtask_path = args.subtask_annotations or os.path.join(args.data_dir, "subtask_annotations.json")
    dataset = RobotEditDataset(
        data_dir=args.data_dir,
        subtask_annotations_path=subtask_path,
        cameras=args.cameras.split(","),
        image_size=args.image_size,
        rank=rank,
        world_size=world_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer + scheduler
    from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-15,
        weight_decay=0.0,
    )
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
        min_lr=args.min_lr,
    )

    train_step = 0
    if args.auto_resume and resume_from != args.model_path:
        optimizer, scheduler, train_step, _ = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config
        )

    # Training loop
    logger.info(f"Starting training from step {train_step}...")
    optimizer.zero_grad()
    micro_step = 0
    start_time = time.time()

    while train_step < args.total_steps:
        for batch_raw in dataloader:
            if train_step >= args.total_steps:
                break

            batch_raw = [s for s in batch_raw if s is not None]
            data = None
            if batch_raw:
                data = pack_robot_edit_samples(
                    batch_raw, tokenizer, new_token_ids, args, vae_downsample
                )

            # All ranks must agree to skip; otherwise the collectives below desync.
            has_data = torch.tensor([int(data is not None)], device=device)
            dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
            if has_data.item() == 0:
                continue

            data = move_data_to_device(data, device)

            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                # Encode images through frozen VAE (both source conditioning + target)
                with torch.no_grad():
                    data["padded_latent"] = vae_model.encode(data.pop("padded_images"))

                # Build forward kwargs — only include ViT if not all CFG-dropped
                fwd_kwargs = dict(
                    sequence_length=data["sequence_length"],
                    packed_text_ids=data["packed_text_ids"],
                    packed_text_indexes=data["packed_text_indexes"],
                    sample_lens=data["sample_lens"],
                    packed_position_ids=data["packed_position_ids"],
                    nested_attention_masks=data["nested_attention_masks"],
                    padded_latent=data["padded_latent"],
                    patchified_vae_latent_shapes=data["patchified_vae_latent_shapes"],
                    packed_latent_position_ids=data["packed_latent_position_ids"],
                    packed_vae_token_indexes=data["packed_vae_token_indexes"],
                    packed_timesteps=data["packed_timesteps"],
                    mse_loss_indexes=data["mse_loss_indexes"],
                )
                has_vit = data["packed_vit_tokens"] is not None
                any_has_vit_t = torch.tensor([int(has_vit)], device=device)
                dist.all_reduce(any_has_vit_t, op=dist.ReduceOp.MAX)
                any_rank_has_vit = any_has_vit_t.item() > 0

                skip_und = False
                if has_vit:
                    fwd_kwargs.update(
                        packed_vit_tokens=data["packed_vit_tokens"],
                        packed_vit_token_indexes=data["packed_vit_token_indexes"],
                        packed_vit_position_ids=data["packed_vit_position_ids"],
                        vit_token_seqlens=data["vit_token_seqlens"],
                    )
                elif any_rank_has_vit:
                    # Other ranks have ViT tokens but this one doesn't.
                    # Provide a dummy 1-token ViT input so the FSDP-wrapped
                    # ViT encoder runs on every rank, keeping collectives in sync.
                    dummy_idx = fwd_kwargs["sequence_length"]
                    fwd_kwargs["sequence_length"] += 1
                    fwd_kwargs["sample_lens"] = list(fwd_kwargs["sample_lens"])
                    fwd_kwargs["sample_lens"][-1] += 1
                    last_mask = fwd_kwargs["nested_attention_masks"][-1]
                    n = last_mask.shape[0]
                    new_mask = last_mask.new_full((n + 1, n + 1), float("-inf"))
                    new_mask[:n, :n] = last_mask
                    fwd_kwargs["nested_attention_masks"][-1] = new_mask
                    fwd_kwargs["packed_position_ids"] = torch.cat([
                        fwd_kwargs["packed_position_ids"],
                        torch.zeros(1, device=device, dtype=torch.long),
                    ])
                    vit_dim = 3 * args.vit_patch_size ** 2
                    fwd_kwargs.update(
                        packed_vit_tokens=torch.zeros(1, vit_dim, device=device, dtype=torch.bfloat16),
                        packed_vit_token_indexes=torch.tensor([dummy_idx], device=device, dtype=torch.long),
                        packed_vit_position_ids=torch.tensor([0], device=device, dtype=torch.long),
                        vit_token_seqlens=torch.tensor([1], device=device, dtype=torch.int),
                    )
                else:
                    # ALL ranks dropped all ViT — safe to skip uniformly
                    fsdp_model._fsdp_wrapped_module.config.visual_und = False
                    skip_und = True

                try:
                    loss_dict = fsdp_model(**fwd_kwargs)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"CUDA OOM at step {train_step}: {e}")
                        torch.cuda.empty_cache()
                    raise e
                finally:
                    if skip_und:
                        fsdp_model._fsdp_wrapped_module.config.visual_und = True

            # MSE loss (flow matching velocity prediction)
            mse = loss_dict["mse"]
            total_mse_tokens = torch.tensor(len(data["mse_loss_indexes"]), device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            loss = mse.mean(dim=-1).sum() * world_size / total_mse_tokens

            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                grad_norm = fsdp_model.clip_grad_norm_(args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                fsdp_ema_update(ema_model, fsdp_model, decay=args.ema_decay)
                optimizer.zero_grad()
                train_step += 1

                if rank == 0 and train_step % args.log_every == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = train_step / elapsed if elapsed > 0 else 0
                    lr_now = scheduler.get_last_lr()[0]
                    mse_val = loss.item() * args.gradient_accumulation_steps
                    logger.info(
                        f"step={train_step}/{args.total_steps} "
                        f"mse={mse_val:.4f} "
                        f"grad_norm={grad_norm:.3f} lr={lr_now:.2e} "
                        f"speed={steps_per_sec:.2f} steps/s"
                    )
                    if args.wandb:
                        import wandb
                        wandb.log({
                            "mse": mse_val,
                            "grad_norm": grad_norm,
                            "lr": lr_now,
                            "step": train_step,
                        })

                if train_step % args.save_every == 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    FSDPCheckpoint.fsdp_save_ckpt(
                        args.output_dir, train_step, fsdp_model, ema_model,
                        optimizer, scheduler, None, logger, fsdp_config,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    if rank == 0:
                        logger.info(f"Saved checkpoint at step {train_step}")

    # Final save
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    FSDPCheckpoint.fsdp_save_ckpt(
        args.output_dir, train_step, fsdp_model, ema_model,
        optimizer, scheduler, None, logger, fsdp_config,
    )

    if rank == 0:
        save_dir = os.path.join(args.output_dir, f"{train_step:07d}")
        export_dir = os.path.join(args.output_dir, "final")
        os.makedirs(export_dir, exist_ok=True)
        for f in ["config.json", "llm_config.json", "vit_config.json",
                  "ae.safetensors", "tokenizer.json", "tokenizer_config.json",
                  "vocab.json", "merges.txt", "preprocessor_config.json",
                  "generation_config.json"]:
            src = os.path.join(args.model_path, f)
            if os.path.exists(src):
                shutil.copy2(src, export_dir)
        ema_src = os.path.join(save_dir, "ema.safetensors")
        if os.path.exists(ema_src):
            shutil.copy2(ema_src, export_dir)
        logger.info(f"Exported final model to {export_dir}")

    dist.destroy_process_group()
    if rank == 0 and args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
