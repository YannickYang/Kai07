"""High-Level Policy for pi0.7 subtask text generation.

Generates subtask instructions given:
  - Current multi-view observations (via SigLIP)
  - Task instruction
  - Optional history of previous subtasks

Uses the PaliGemma backbone (SigLIP + Gemma 2B), trained with
cross-entropy loss on subtask text prediction.

Architecture:
    Input: [image_tokens | prompt_tokens | target_tokens]
    Attention: bidirectional on prefix (images + prompt), causal on target
    Loss: cross-entropy on target subtask tokens
"""

import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.models.auto import CONFIG_MAPPING

import openpi.models.gemma as _gemma

logger = logging.getLogger("openpi")


class HighLevelPolicy(nn.Module):
    """High-level policy for subtask text generation.

    Reuses PaliGemma backbone (SigLIP vision + Gemma language model).
    No action expert is needed - only the VLM generates text.

    Training:
        Teacher-forced cross-entropy on subtask token prediction.
        Initialized from pi0.5 base VLM weights.

    Inference:
        Autoregressive subtask text generation with KV-cache.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        vlm_config = _gemma.get_config(getattr(config, "paligemma_variant", "gemma_2b"))

        # Create PaliGemma model (VLM only)
        from transformers import PaliGemmaForConditionalGeneration

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.vocab_size = 257152

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)

        # Optionally freeze vision tower
        self.freeze_vision = getattr(config, "hlp_freeze_vision", True)
        if self.freeze_vision:
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False

        self.max_generate_len = getattr(config, "hlp_max_generate_len", 64)
        self.hidden_dim = vlm_config.width

    def embed_images(self, images: list[Tensor]) -> Tensor:
        """Encode observation images through SigLIP.

        Args:
            images: List of [B, C, H, W] image tensors (one per camera view)

        Returns:
            [B, N_img_total, D] concatenated image embeddings
        """
        all_embs = []
        for img in images:
            emb = self.paligemma.model.get_image_features(img)
            all_embs.append(emb)
        return torch.cat(all_embs, dim=1)

    def forward(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        prompt_tokens: Tensor,
        prompt_masks: Tensor,
        target_tokens: Tensor,
        target_masks: Tensor,
    ) -> Tensor:
        """Training forward: teacher-forced cross-entropy on subtask tokens.

        Args:
            images: List of [B, C, H, W] image tensors per camera
            image_masks: List of [B] boolean masks (which cameras valid)
            prompt_tokens: [B, T_prompt] tokenized prompt (task + history)
            prompt_masks: [B, T_prompt] prompt padding mask
            target_tokens: [B, T_target] tokenized target subtask
            target_masks: [B, T_target] target padding mask

        Returns:
            Scalar cross-entropy loss
        """
        # Encode images
        img_embs = self.embed_images(images)
        B, N_img, D = img_embs.shape

        # Embed text tokens
        lm = self.paligemma.language_model
        prompt_embs = lm.embed_tokens(prompt_tokens) * math.sqrt(D)
        target_embs = lm.embed_tokens(target_tokens) * math.sqrt(D)

        # Concatenate: [images | prompt | target]
        all_embs = torch.cat([img_embs, prompt_embs, target_embs], dim=1)
        T_prompt = prompt_tokens.shape[1]
        T_target = target_tokens.shape[1]
        N_prefix = N_img + T_prompt
        N_total = all_embs.shape[1]

        # Padding mask
        img_pad = torch.ones(B, N_img, dtype=torch.bool, device=all_embs.device)
        pad_mask = torch.cat([img_pad, prompt_masks, target_masks], dim=1)

        # Attention: bidirectional prefix, causal target
        ar_mask = torch.zeros(B, N_total, dtype=torch.long, device=all_embs.device)
        ar_mask[:, N_prefix:] = 1  # Target portion is causal

        cumsum = torch.cumsum(ar_mask, dim=1)
        att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d = pad_mask[:, None, :] & pad_mask[:, :, None]
        att_2d = att_2d & pad_2d
        att_mask_4d = torch.where(att_2d[:, None, :, :], 0.0, -2.3819763e38)

        position_ids = torch.cumsum(pad_mask.long(), dim=1) - 1

        # Forward through VLM backbone (GemmaModel)
        outputs = lm.model(
            inputs_embeds=all_embs.to(dtype=lm.model.embed_tokens.weight.dtype),
            attention_mask=att_mask_4d,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state

        # Predict target tokens: use positions [N_prefix-1 : N_total-1] to predict [N_prefix : N_total]
        pred_hidden = hidden_states[:, N_prefix - 1 : N_total - 1, :]
        logits = lm.lm_head(pred_hidden.to(dtype=lm.lm_head.weight.dtype))

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.contiguous().view(-1, logits.size(-1)),
            target_tokens.contiguous().view(-1),
            reduction="none",
        ).view(target_masks.shape)

        loss = (loss * target_masks.float()).sum() / target_masks.float().sum().clamp(min=1.0)
        return loss

    @torch.no_grad()
    def generate_subtask(
        self,
        images: list[Tensor],
        prompt_tokens: Tensor,
        prompt_masks: Tensor,
        max_new_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int = 1,
    ) -> Tensor:
        """Generate subtask text autoregressively.

        Args:
            images: List of [B, C, H, W] observation images
            prompt_tokens: [B, T] tokenized prompt
            prompt_masks: [B, T] prompt mask
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            eos_token_id: End of sequence token ID

        Returns:
            Generated token IDs [B, T_generated]
        """
        max_new = max_new_tokens or self.max_generate_len
        B = prompt_tokens.shape[0]
        device = prompt_tokens.device
        lm = self.paligemma.language_model
        D = self.hidden_dim

        # Encode images + prompt as prefix
        img_embs = self.embed_images(images)
        N_img = img_embs.shape[1]

        prompt_embs = lm.embed_tokens(prompt_tokens) * math.sqrt(D)
        prefix_embs = torch.cat([img_embs, prompt_embs], dim=1)
        N_prefix = prefix_embs.shape[1]

        prefix_pad = torch.cat([
            torch.ones(B, N_img, dtype=torch.bool, device=device),
            prompt_masks,
        ], dim=1)

        # Bidirectional attention for prefix
        prefix_pad_2d = prefix_pad[:, None, :] & prefix_pad[:, :, None]
        prefix_att_4d = torch.where(prefix_pad_2d[:, None, :, :], 0.0, -2.3819763e38)
        prefix_pos_ids = torch.cumsum(prefix_pad.long(), dim=1) - 1

        # Cache prefix KV
        model_dtype = lm.model.embed_tokens.weight.dtype
        outputs = lm.model(
            inputs_embeds=prefix_embs.to(dtype=model_dtype),
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos_ids,
            use_cache=True,
        )
        past_kv = outputs.past_key_values
        hidden = outputs.last_hidden_state

        # Initialize generation from last prefix hidden state
        next_logits = lm.lm_head(hidden[:, -1:, :].to(dtype=lm.lm_head.weight.dtype))
        next_pos = prefix_pos_ids[:, -1:] + 1

        generated = []
        for _ in range(max_new):
            # Sample
            scaled_logits = next_logits[:, -1, :] / max(temperature, 1e-6)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(scaled_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove] = -float("inf")
                scaled_logits = torch.zeros_like(scaled_logits).scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token)

            if (next_token == eos_token_id).all():
                break

            # Forward one step with cache
            next_emb = lm.embed_tokens(next_token) * math.sqrt(D)
            next_emb = next_emb.to(dtype=model_dtype)

            # Attend to all cached positions
            cache_len = past_kv[0][0].shape[2]
            step_att = torch.zeros(B, 1, 1, cache_len + 1, device=device)

            next_pos = next_pos + 1
            outputs = lm.model(
                inputs_embeds=next_emb,
                attention_mask=step_att,
                position_ids=next_pos,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = outputs.past_key_values
            next_logits = lm.lm_head(
                outputs.last_hidden_state.to(dtype=lm.lm_head.weight.dtype)
            )

        if generated:
            return torch.cat(generated, dim=1)
        return torch.zeros(B, 0, dtype=torch.long, device=device)
