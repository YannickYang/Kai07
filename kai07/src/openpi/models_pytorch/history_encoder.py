"""MEM (Memory-Efficient Multi-frame) History Encoder for pi0.7.

Encodes multi-frame, multi-view history observations into a compact token sequence.
Following pi0.7 paper:
  - Each history frame is encoded through the shared SigLIP encoder
  - Spatial compression: 1024 tokens/frame → 64 tokens/frame via 2-layer MLP
  - Temporal: concatenate all compressed frame tokens
  - History stride: 1 second (30 frames at 30fps)
  - Random global drop of entire history with probability 0.3 during training
"""

import math
import random

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SpatialCompressor(nn.Module):
    """Compress per-frame image tokens from N_patch to N_compressed tokens.

    Uses a 2-layer MLP that maps each frame's patch tokens to a smaller set of tokens.
    Input: [B, N_patch, D] -> Output: [B, N_compressed, D]
    """

    def __init__(self, input_tokens: int, output_tokens: int, hidden_dim: int):
        super().__init__()
        self.output_tokens = output_tokens
        # Project to compressed space: [N_patch, D] -> [output_tokens, D]
        # We use a linear layer that maps the token dimension
        self.proj_down = nn.Linear(input_tokens, output_tokens)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compress spatial tokens.

        Args:
            x: [B, N_patch, D] - patch tokens from SigLIP

        Returns:
            [B, N_compressed, D] - compressed tokens
        """
        # Transpose to [B, D, N_patch], compress tokens, transpose back
        x = x.transpose(1, 2)  # [B, D, N_patch]
        x = self.proj_down(x)  # [B, D, N_compressed]
        x = x.transpose(1, 2)  # [B, N_compressed, D]
        x = self.norm(x)
        x = x + self.mlp(x)  # Residual
        return x


class MEMHistoryEncoder(nn.Module):
    """Memory-Efficient Multi-frame History Encoder.

    Encodes a sequence of historical observations (multi-view, multi-frame)
    into a compact token sequence for the VLA backbone.

    Architecture:
        1. Each history frame goes through shared SigLIP encoder → N_patch tokens per view
        2. Spatial compression: N_patch → N_compressed tokens per view per frame
        3. All compressed tokens concatenated: [N_frames * N_views * N_compressed, D]

    Training-time augmentation:
        - Random global drop of entire history (prob 0.3)
        - The image encoder is shared with the current observation encoder (SigLIP)
    """

    def __init__(
        self,
        siglip_dim: int = 1152,  # SigLIP So400m output dim
        backbone_dim: int = 2048,  # Gemma 2B hidden dim
        num_patch_tokens: int = 256,  # SigLIP 224x224/14 = 16x16 = 256 patches
        num_compressed_tokens: int = 64,  # Compressed tokens per frame per view
        max_history_frames: int = 6,
        num_views: int = 3,
        history_drop_prob: float = 0.3,
    ):
        super().__init__()
        self.siglip_dim = siglip_dim
        self.backbone_dim = backbone_dim
        self.num_patch_tokens = num_patch_tokens
        self.num_compressed_tokens = num_compressed_tokens
        self.max_history_frames = max_history_frames
        self.num_views = num_views
        self.history_drop_prob = history_drop_prob

        # Project from SigLIP dim to backbone dim (if different)
        self.input_proj = nn.Linear(siglip_dim, backbone_dim) if siglip_dim != backbone_dim else nn.Identity()

        # Spatial compressor (shared across frames and views)
        self.spatial_compressor = SpatialCompressor(
            input_tokens=num_patch_tokens,
            output_tokens=num_compressed_tokens,
            hidden_dim=backbone_dim,
        )

        # Temporal position embedding
        self.temporal_pos_embed = nn.Embedding(max_history_frames, backbone_dim)

        # View embedding
        self.view_embed = nn.Embedding(num_views, backbone_dim)

    def forward(
        self,
        history_image_tokens: list[Tensor],
        history_masks: list[Tensor] | None = None,
        train: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Encode history observations into compressed tokens.

        Args:
            history_image_tokens: List of [B, N_patch, siglip_dim] tensors,
                one per (frame, view). Ordered as:
                [frame0_view0, frame0_view1, frame0_view2, frame1_view0, ...]
            history_masks: List of [B] boolean masks per (frame, view).
                True = valid, False = padded/missing.
            train: Whether in training mode.

        Returns:
            tokens: [B, N_total_compressed, backbone_dim] - compressed history tokens
            masks: [B, N_total_compressed] - boolean mask (True = valid)
        """
        if not history_image_tokens:
            batch_size = 1
            device = torch.device("cpu")
            empty_tokens = torch.zeros(batch_size, 0, self.backbone_dim, device=device)
            empty_masks = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
            return empty_tokens, empty_masks

        batch_size = history_image_tokens[0].shape[0]
        device = history_image_tokens[0].device

        # Training: random global drop of entire history
        if train and random.random() < self.history_drop_prob:
            total_tokens = self.max_history_frames * self.num_views * self.num_compressed_tokens
            empty_tokens = torch.zeros(batch_size, total_tokens, self.backbone_dim, device=device)
            empty_masks = torch.zeros(batch_size, total_tokens, dtype=torch.bool, device=device)
            return empty_tokens, empty_masks

        num_entries = len(history_image_tokens)
        num_frames = num_entries // self.num_views

        all_compressed = []
        all_masks = []

        for frame_idx in range(num_frames):
            for view_idx in range(self.num_views):
                entry_idx = frame_idx * self.num_views + view_idx
                if entry_idx >= num_entries:
                    break

                # [B, N_patch, siglip_dim]
                img_tokens = history_image_tokens[entry_idx]

                # Project to backbone dim
                img_tokens = self.input_proj(img_tokens)  # [B, N_patch, backbone_dim]

                # Spatial compression
                compressed = self.spatial_compressor(img_tokens)  # [B, N_compressed, backbone_dim]

                # Add temporal position embedding
                t_embed = self.temporal_pos_embed(
                    torch.tensor(frame_idx, device=device)
                )  # [backbone_dim]
                compressed = compressed + t_embed.unsqueeze(0).unsqueeze(0)

                # Add view embedding
                v_embed = self.view_embed(
                    torch.tensor(view_idx, device=device)
                )  # [backbone_dim]
                compressed = compressed + v_embed.unsqueeze(0).unsqueeze(0)

                all_compressed.append(compressed)

                # Mask
                if history_masks is not None and entry_idx < len(history_masks):
                    mask = history_masks[entry_idx]  # [B]
                    # Expand mask to all compressed tokens
                    mask = mask[:, None].expand(batch_size, self.num_compressed_tokens)
                else:
                    mask = torch.ones(batch_size, self.num_compressed_tokens, dtype=torch.bool, device=device)
                all_masks.append(mask)

        if not all_compressed:
            total_tokens = self.max_history_frames * self.num_views * self.num_compressed_tokens
            empty_tokens = torch.zeros(batch_size, total_tokens, self.backbone_dim, device=device)
            empty_masks = torch.zeros(batch_size, total_tokens, dtype=torch.bool, device=device)
            return empty_tokens, empty_masks

        # Concatenate all compressed tokens
        tokens = torch.cat(all_compressed, dim=1)  # [B, N_total, backbone_dim]
        masks = torch.cat(all_masks, dim=1)  # [B, N_total]

        return tokens, masks
