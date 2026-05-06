"""PI0.7 PyTorch model implementation.

Extends PI0Pytorch (pi0.5) with:
  - MEM history encoder for multi-frame temporal compression
  - Linear state projection (replaces discrete state tokens)
  - 860M action expert (via config)
  - Subgoal image support
  - Observation delay simulation for training
  - Classifier-Free Guidance (CFG) for inference
"""

import logging
import math
import random

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.history_encoder import MEMHistoryEncoder
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.pi0_pytorch import (
    PI0Pytorch,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    sample_beta,
)

logger = logging.getLogger("openpi")


# Block IDs for pi0.7 block-causal attention
BLOCK_HISTORY_OBS = 0
BLOCK_CURRENT_OBS = 1
BLOCK_SUBGOAL = 2
BLOCK_TEXT = 3
BLOCK_STATE = 4
BLOCK_ACTION = 5


def make_block_causal_masks(
    block_ids: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    """Create pi0.7 block-causal attention mask.

    Attention rules:
        history_obs: bidirectional within block
        current_obs: attend to history_obs + bidirectional within
        subgoal:     attend to history_obs + current_obs + bidirectional within
        text:        attend to history_obs + current_obs + subgoal + causal within
        state:       attend to all above + bidirectional within
        action:      attend to all above + bidirectional within

    This is a strictly lower-triangular pattern at the block level,
    except text which uses causal (token-level) within its block.

    Args:
        block_ids: [B, N] integer tensor, each value is one of BLOCK_* constants
        pad_mask: [B, N] boolean tensor, True = valid token

    Returns:
        att_mask: [B, 1, N, N] float tensor with 0.0 (attend) and -inf (block)
    """
    B, N = block_ids.shape
    device = block_ids.device

    # [B, N, 1] x [B, 1, N] -> broadcast comparison
    q_block = block_ids[:, :, None]  # [B, N, 1] - query blocks
    k_block = block_ids[:, None, :]  # [B, 1, N] - key blocks

    # Rule: query can attend to key if key_block <= query_block
    # (lower-triangular at block level)
    block_mask = k_block <= q_block  # [B, N, N]

    # Special case: text block uses causal attention within itself
    # Within the text block, only attend to positions <= current position
    is_text_q = (q_block == BLOCK_TEXT)  # [B, N, 1]
    is_text_k = (k_block == BLOCK_TEXT)  # [B, 1, N]
    text_self = is_text_q & is_text_k  # [B, N, N]

    # Position indices for causal masking within text block
    pos = torch.arange(N, device=device)
    causal_mask = pos[None, :, None] >= pos[None, None, :]  # [1, N, N]

    # Apply causal within text block, bidirectional otherwise within same block
    block_mask = torch.where(text_self, block_mask & causal_mask, block_mask)

    # Apply padding mask
    pad_2d = pad_mask[:, None, :] & pad_mask[:, :, None]  # [B, N, N]
    block_mask = block_mask & pad_2d

    # Convert to 4D float mask: [B, 1, N, N]
    att_mask = block_mask[:, None, :, :]
    att_mask = torch.where(att_mask, 0.0, -2.3819763e38)

    return att_mask


class PI07Pytorch(PI0Pytorch):
    """PI0.7 model with MEM history encoder and enhanced features.

    Key differences from PI0Pytorch (pi0.5):
    1. MEM history encoder: compresses multi-frame history into tokens
    2. Linear state projection: state → backbone dim (no discrete tokenization)
    3. State tokens for both current and history states
    4. Subgoal image support in prefix
    5. Observation delay simulation during training
    6. CFG support during inference
    """

    def __init__(self, config):
        # Initialize parent (PI0Pytorch) - handles backbone, action expert, etc.
        super().__init__(config)

        self.pi07 = True
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        paligemma_config = _gemma.get_config(config.paligemma_variant)

        # MEM History Encoder
        # SigLIP So400m/14: 1152-dim, 256 patches for 224x224, or 1024 patches for 448x448
        siglip_dim = 1152
        if config.image_resolution == 448:
            num_patch_tokens = 1024  # 448/14 = 32, 32x32 = 1024
        else:
            num_patch_tokens = 256  # 224/14 = 16, 16x16 = 256

        self.history_encoder = MEMHistoryEncoder(
            siglip_dim=siglip_dim,
            backbone_dim=paligemma_config.width,
            num_patch_tokens=num_patch_tokens,
            num_compressed_tokens=64,
            max_history_frames=config.max_history_frames,
            num_views=3,  # head, left, right
            history_drop_prob=0.3,
        )

        # Linear state projection: state_dim → backbone_dim
        # This replaces the discrete state tokenization in pi0.5
        self.state_linear_proj = nn.Linear(config.action_dim, paligemma_config.width)

        # History state projection (same dimensions, separate weights)
        self.history_state_proj = nn.Linear(config.action_dim, paligemma_config.width)

        # Override action projections for the new expert width
        # (parent __init__ created these with width from config, but we need to be explicit)
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        # pi0.7 uses adaRMSNorm (like pi0.5)
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # Max observation delay for training (implemented in data_loader.ObsDelayDataset)
        # The dataset loads observation from frame t-d, action from frame t (d ∈ [0, max_obs_delay])
        self.max_obs_delay = getattr(config, "max_obs_delay", 12)

        # CFG beta for inference
        self.cfg_beta = getattr(config, "cfg_beta", 1.7)

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        history_image_tokens=None,
        history_masks=None,
        subgoal_images=None,
        subgoal_masks=None,
        train=True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed all prefix tokens for pi0.7.

        Token order:
            [history_obs | current_obs | subgoal | text]

        Returns:
            embs: [B, N_prefix, D]
            pad_masks: [B, N_prefix]
            att_masks: [B, N_prefix] (legacy, kept for compatibility)
            block_ids: [B, N_prefix] integer block IDs for block-causal attention
        """
        embs = []
        pad_masks = []
        att_masks = []
        block_ids_list = []

        # 1. History observations (MEM encoded)
        if history_image_tokens is not None and len(history_image_tokens) > 0:
            history_embs, history_pad = self.history_encoder(
                history_image_tokens, history_masks, train=train
            )
            if history_embs.shape[1] > 0:
                embs.append(history_embs)
                pad_masks.append(history_pad)
                att_masks += [0] * history_embs.shape[1]
                block_ids_list += [BLOCK_HISTORY_OBS] * history_embs.shape[1]

        # 2. Current observation images (through SigLIP)
        for img, img_mask in zip(images, img_masks, strict=True):
            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs
            block_ids_list += [BLOCK_CURRENT_OBS] * num_img_embs

        # 3. Subgoal images (through SigLIP, if available)
        if subgoal_images is not None:
            for sg_idx, (sg_img, sg_mask) in enumerate(
                zip(subgoal_images, subgoal_masks or [None] * len(subgoal_images), strict=False)
            ):
                def subgoal_embed_func(img):
                    return self.paligemma_with_expert.embed_image(img)

                sg_emb = self._apply_checkpoint(subgoal_embed_func, sg_img)
                bsize, num_sg_embs = sg_emb.shape[:2]

                embs.append(sg_emb)
                if sg_mask is not None:
                    pad_masks.append(sg_mask[:, None].expand(bsize, num_sg_embs))
                else:
                    pad_masks.append(torch.ones(bsize, num_sg_embs, dtype=torch.bool, device=sg_img.device))
                att_masks += [0] * num_sg_embs
                block_ids_list += [BLOCK_SUBGOAL] * num_sg_embs

        # 4. Language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
        block_ids_list += [BLOCK_TEXT] * num_lang_embs

        # Concatenate all prefix tokens
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, att_masks.shape[0])

        # Block IDs
        block_ids = torch.tensor(block_ids_list, dtype=torch.long, device=pad_masks.device)
        block_ids = block_ids[None, :].expand(bsize, block_ids.shape[0])

        return embs, pad_masks, att_masks, block_ids

    def embed_suffix(self, state, noisy_actions, timestep, history_states=None):
        """Embed state and actions for the action expert.

        Token order in suffix:
            [state | action_tokens]

        Returns:
            embs, pad_masks, att_masks, adarms_cond, block_ids
        """
        embs = []
        pad_masks = []
        att_masks = []
        block_ids_list = []
        bsize = noisy_actions.shape[0]
        device = noisy_actions.device

        # 1. History state tokens (if available)
        if history_states is not None and history_states.shape[1] > 0:
            def hist_state_func(hs):
                return self.history_state_proj(hs)

            hist_state_embs = self._apply_checkpoint(hist_state_func, history_states.float())
            embs.append(hist_state_embs)
            n_hist = hist_state_embs.shape[1]
            pad_masks.append(torch.ones(bsize, n_hist, dtype=torch.bool, device=device))
            att_masks += [1] + [0] * (n_hist - 1)
            block_ids_list += [BLOCK_STATE] * n_hist

        # 2. Current state token (linear projection)
        def state_func(s):
            return self.state_linear_proj(s)

        state_emb = self._apply_checkpoint(state_func, state.float())
        embs.append(state_emb[:, None, :])
        pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))

        if history_states is None or history_states.shape[1] == 0:
            att_masks += [1]
        else:
            att_masks += [0]
        block_ids_list += [BLOCK_STATE]

        # 3. Action tokens with timestep conditioning
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        # adaRMSNorm conditioning via time MLP
        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        adarms_cond = time_emb
        action_time_emb = action_emb

        embs.append(action_time_emb)
        action_time_dim = action_time_emb.shape[1]
        pad_masks.append(torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device))

        att_masks += [1] + [0] * (self.config.action_horizon - 1)
        block_ids_list += [BLOCK_ACTION] * action_time_dim

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=device)
        att_masks = att_masks[None, :].expand(bsize, att_masks.shape[0])

        block_ids = torch.tensor(block_ids_list, dtype=torch.long, device=device)
        block_ids = block_ids[None, :].expand(bsize, block_ids.shape[0])

        return embs, pad_masks, att_masks, adarms_cond, block_ids

    def _compute_vlm_lm_loss(self, prefix_embs, prefix_pad_masks, prefix_block_ids, lang_tokens, lang_masks):
        """Compute language modeling loss on text tokens for Knowledge Insulation.

        Runs a separate VLM-only forward pass (no action expert) to compute
        next-token prediction loss, preserving VLM language understanding.

        Args:
            prefix_embs: [B, N_prefix, D] prefix embeddings (with gradients)
            prefix_pad_masks: [B, N_prefix] padding mask
            prefix_block_ids: [B, N_prefix] block IDs
            lang_tokens: [B, T] tokenized text
            lang_masks: [B, T] text padding mask

        Returns:
            Scalar VLM language modeling loss
        """
        # Build prefix-only block-causal attention mask
        prefix_att_4d = make_block_causal_masks(prefix_block_ids, prefix_pad_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Run VLM backbone (GemmaModel) to get hidden states
        vlm_backbone = self.paligemma_with_expert.paligemma.language_model.model
        outputs = vlm_backbone(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_att_4d,
            position_ids=prefix_position_ids,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state  # [B, N_prefix, D]

        # Text tokens are at the end of the prefix
        num_text_tokens = lang_tokens.shape[1]
        text_hidden = hidden_states[:, -num_text_tokens:, :]  # [B, T, D]

        # Apply LM head
        lm_head = self.paligemma_with_expert.paligemma.language_model.lm_head
        text_hidden_cast = text_hidden.to(dtype=lm_head.weight.dtype)
        logits = lm_head(text_hidden_cast)  # [B, T, vocab_size]

        # Next-token prediction: predict token[i+1] from position[i]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = lang_tokens[:, 1:].contiguous()
        shift_mask = lang_masks[:, 1:].contiguous().float()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_mask.shape)

        vlm_loss = (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1.0)
        return vlm_loss

    def forward(self, observation, actions, noise=None, time=None) -> tuple[Tensor, Tensor]:
        """Training forward pass with Knowledge Insulation (KI).

        KI prevents action loss from degrading VLM capabilities:
        - Action loss: computed with detached prefix (no VLM gradient)
        - VLM loss: language modeling on text tokens (preserves VLM)

        Returns:
            (action_loss, vlm_loss) tuple.
            action_loss: [B, action_horizon, action_dim] per-element MSE
            vlm_loss: scalar cross-entropy for VLM preservation
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=True
        )

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embed prefix WITH gradients (needed for VLM loss)
        prefix_embs, prefix_pad_masks, prefix_att_masks, prefix_block_ids = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, train=True
        )

        # Embed suffix (state + noisy actions)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond, suffix_block_ids = self.embed_suffix(
            state, x_t, time
        )

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # === ACTION LOSS (KI: detach prefix to stop VLM gradients from action loss) ===
        prefix_embs_detached = prefix_embs.detach()

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        block_ids = torch.cat([prefix_block_ids, suffix_block_ids], dim=1)

        # Use block-causal attention mask
        att_2d_masks_4d = make_block_causal_masks(block_ids, pad_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs_detached, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        action_loss = F.mse_loss(u_t, v_t, reduction="none")

        # === VLM LOSS (language modeling, gradients flow through VLM) ===
        vlm_loss = self._compute_vlm_lm_loss(
            prefix_embs, prefix_pad_masks, prefix_block_ids, lang_tokens, lang_masks
        )

        return action_loss, vlm_loss

    @torch.no_grad()
    def sample_actions(
        self, device, observation, noise=None, num_steps=10,
        cfg_beta=None, uncond_observation=None,
    ) -> Tensor:
        """Inference with block-causal attention and Classifier-Free Guidance.

        When cfg_beta > 1.0 and uncond_observation is provided, uses CFG:
            v = v_uncond + beta * (v_cond - v_uncond)

        Args:
            device: Target device
            observation: Conditioned observation (with metadata in prompt)
            noise: Optional initial noise
            num_steps: Number of denoising steps
            cfg_beta: CFG strength (1.0 = no guidance)
            uncond_observation: Unconditioned observation (metadata stripped from prompt)
        """
        beta = cfg_beta if cfg_beta is not None else self.cfg_beta
        use_cfg = beta > 1.0 and uncond_observation is not None

        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False
        )

        # === Conditioned prefix cache ===
        prefix_embs, prefix_pad_masks, _, prefix_block_ids = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, train=False
        )
        prefix_att_4d = make_block_causal_masks(prefix_block_ids, prefix_pad_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # === Unconditioned prefix cache (for CFG) ===
        uncond_past_kv = None
        uncond_prefix_block_ids = None
        uncond_prefix_pad_masks = None
        if use_cfg:
            u_imgs, u_img_masks, u_lang_tok, u_lang_masks, _ = self._preprocess_observation(
                uncond_observation, train=False
            )
            u_prefix_embs, u_prefix_pad_masks, _, u_prefix_block_ids = self.embed_prefix(
                u_imgs, u_img_masks, u_lang_tok, u_lang_masks, train=False
            )
            u_prefix_att_4d = make_block_causal_masks(u_prefix_block_ids, u_prefix_pad_masks)
            u_prefix_pos_ids = torch.cumsum(u_prefix_pad_masks, dim=1) - 1

            _, uncond_past_kv = self.paligemma_with_expert.forward(
                attention_mask=u_prefix_att_4d,
                position_ids=u_prefix_pos_ids,
                past_key_values=None,
                inputs_embeds=[u_prefix_embs, None],
                use_cache=True,
            )
            uncond_prefix_block_ids = u_prefix_block_ids
            uncond_prefix_pad_masks = u_prefix_pad_masks

        # === Denoising loop ===
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            # Conditioned velocity
            v_cond = self.denoise_step(
                state, prefix_pad_masks, past_key_values, x_t, expanded_time,
                prefix_block_ids=prefix_block_ids,
            )

            if use_cfg:
                # Unconditioned velocity
                v_uncond = self.denoise_step(
                    state, uncond_prefix_pad_masks, uncond_past_kv, x_t, expanded_time,
                    prefix_block_ids=uncond_prefix_block_ids,
                )
                # CFG combination: v = v_uncond + beta * (v_cond - v_uncond)
                v_t = v_uncond + beta * (v_cond - v_uncond)
            else:
                v_t = v_cond

            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        prefix_block_ids=None,
    ):
        """Apply one denoising step with block-causal attention."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond, suffix_block_ids = self.embed_suffix(
            state, x_t, timestep
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        # For the cached forward pass, we need cross-attention masks from suffix to prefix
        # and self-attention masks within suffix

        # Cross-attention: suffix queries attending to prefix keys
        # Use block-causal logic: each suffix block can attend to prefix blocks
        # with lower or equal block ID
        if prefix_block_ids is not None:
            # [B, suffix_len, prefix_len] cross-attention mask
            s_block = suffix_block_ids[:, :, None]  # [B, suffix_len, 1]
            p_block = prefix_block_ids[:, None, :]  # [B, 1, prefix_len]
            cross_mask = (p_block <= s_block) & prefix_pad_masks[:, None, :]
        else:
            cross_mask = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        # Self-attention within suffix using block-causal
        suffix_self_mask = make_block_causal_masks(suffix_block_ids, suffix_pad_masks)
        suffix_self_mask_2d = (suffix_self_mask[:, 0, :, :] == 0.0)  # [B, suffix_len, suffix_len]

        # Combine cross + self into full attention mask
        full_mask = torch.cat([cross_mask, suffix_self_mask_2d], dim=2)  # [B, suffix_len, prefix_len+suffix_len]

        # Position IDs
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_4d = full_mask[:, None, :, :]
        full_att_4d = torch.where(full_att_4d, 0.0, -2.3819763e38)

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
