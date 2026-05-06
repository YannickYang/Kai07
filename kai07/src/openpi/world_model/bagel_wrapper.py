"""BAGEL world model wrapper for pi0.7 subgoal image generation.

Wraps ByteDance-Seed/BAGEL-7B-MoT to generate subgoal images given:
  - Current observation images (multi-view)
  - Subtask text description

Loosely coupled design:
  - Training: offline precompute subgoal images via scripts/precompute_subgoals.py
  - Inference: runs on separate GPU(s) asynchronously via Pi07Policy
"""

import logging
import os
import sys

import torch
from PIL import Image

logger = logging.getLogger("openpi")


class BagelWorldModel:
    """Wrapper around BAGEL for subgoal image generation.

    Usage:
        model = BagelWorldModel.from_pretrained(
            model_path="/path/to/BAGEL-7B-MoT",
            bagel_repo="/path/to/Bagel",        # Bagel source code directory
        )
        subgoals = model.generate_subgoals(
            obs_images={"head": pil_img, "left": pil_img},
            subtask_text="Fold the cloth",
        )
    """

    # Default generation hyperparameters (image editing mode)
    DEFAULT_HYPER = dict(
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=30,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    )

    def __init__(self, inferencer, output_size: tuple[int, int] = (448, 448)):
        self.inferencer = inferencer
        self.output_size = output_size

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        bagel_repo: str | None = None,
        device_map: dict | str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        output_size: tuple[int, int] = (448, 448),
    ) -> "BagelWorldModel":
        """Load BAGEL model.

        Args:
            model_path: Path to BAGEL-7B-MoT weights directory
            bagel_repo: Path to Bagel source code (default: auto-detect from project layout)
            device_map: Device placement for model parallelism
            dtype: Model precision
            output_size: Subgoal image output resolution
        """
        # Resolve Bagel source directory
        if bagel_repo is None:
            bagel_repo = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "Bagel")
            )
        if not os.path.isdir(bagel_repo):
            raise FileNotFoundError(
                f"Bagel source not found at {bagel_repo}. "
                "Set bagel_repo= to the directory containing inferencer.py"
            )
        if bagel_repo not in sys.path:
            sys.path.insert(0, bagel_repo)

        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from data.data_utils import add_special_tokens
        from data.transforms import ImageTransform
        from inferencer import InterleaveInferencer
        from modeling.autoencoder import load_ae
        from modeling.bagel import (
            Bagel,
            BagelConfig,
            Qwen2Config,
            Qwen2ForCausalLM,
            SiglipVisionConfig,
            SiglipVisionModel,
        )
        from modeling.qwen2 import Qwen2Tokenizer

        logger.info(f"Loading BAGEL from {model_path}")

        # Configs
        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1

        vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )

        # Init empty -> load weights
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, "ema.safetensors"),
            device_map=device_map,
            dtype=dtype,
            force_hooks=True,
        ).eval()

        inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=ImageTransform(1024, 512, 16),
            vit_transform=ImageTransform(980, 224, 14),
            new_token_ids=new_token_ids,
        )

        logger.info("BAGEL loaded")
        return cls(inferencer, output_size=output_size)

    def generate_subgoals(
        self,
        obs_images: dict[str, Image.Image],
        subtask_text: str,
        **kwargs,
    ) -> dict[str, Image.Image]:
        """Generate subgoal images for each camera view.

        Args:
            obs_images: {"head": PIL.Image, "left": PIL.Image, ...}
            subtask_text: e.g. "Fold the cloth"
            **kwargs: Override default generation hyperparameters

        Returns:
            Subgoal images per camera view, resized to self.output_size
        """
        hyper = {**self.DEFAULT_HYPER, **kwargs}
        prompt = f"Show the expected result after the robot completes: {subtask_text}"
        subgoals = {}

        for view, obs_img in obs_images.items():
            try:
                result = self.inferencer(
                    image=obs_img,
                    text=prompt,
                    image_shapes=self.output_size,
                    **hyper,
                )
                img = result.get("image")
                subgoals[view] = img.resize(self.output_size) if img else obs_img.resize(self.output_size)
            except Exception as e:
                logger.warning(f"BAGEL failed for view {view}: {e}")
                subgoals[view] = obs_img.resize(self.output_size)

        return subgoals
