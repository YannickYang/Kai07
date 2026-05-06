"""Pi0.7 asynchronous inference policy.

Three-thread architecture:
  Thread 1 (VLA): Runs the VLA model to produce action chunks
  Thread 2 (World Model): Generates subgoal images with BAGEL
  Thread 3 (High-Level Policy): Generates subtask text

The VLA thread runs at high frequency (every action chunk), while
the world model and HLP threads run asynchronously and update their
outputs when ready.

Usage:
    policy = Pi07Policy(
        vla_model=vla_model,
        world_model=world_model,      # optional
        high_level_policy=hlp_model,   # optional
        action_smoother=ActionSmoother(...),
    )
    policy.start()

    while not done:
        obs = get_observation()
        action = policy.step(obs)
        robot.execute(action)

    policy.stop()
"""

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from openpi.policies.action_smoother import ActionSmoother

logger = logging.getLogger("openpi")


@dataclass
class SharedState:
    """Thread-safe shared state between inference threads."""

    # Current observation (updated by main thread)
    current_obs: object = None
    obs_lock: threading.Lock = field(default_factory=threading.Lock)

    # Current subtask text (updated by HLP thread)
    subtask_text: str = ""
    subtask_lock: threading.Lock = field(default_factory=threading.Lock)

    # Current subgoal images (updated by world model thread)
    subgoal_images: dict | None = None
    subgoal_lock: threading.Lock = field(default_factory=threading.Lock)

    # Task instruction (set once at start)
    task_instruction: str = ""

    # Subtask history (append-only, protected by subtask_lock)
    subtask_history: list[str] = field(default_factory=list)

    # Control flags
    running: bool = False
    new_subtask_requested: bool = False
    new_subgoal_requested: bool = False


class Pi07Policy:
    """Asynchronous pi0.7 inference policy.

    Coordinates three inference components:
    1. VLA model: action chunk generation (synchronous, high frequency)
    2. World model (BAGEL): subgoal image generation (async, lower frequency)
    3. High-level policy: subtask text generation (async, lowest frequency)

    The VLA runs synchronously on each step, while the world model and HLP
    run in background threads. Their results are used by the VLA when available.
    """

    def __init__(
        self,
        vla_model,
        world_model=None,
        high_level_policy=None,
        action_smoother: ActionSmoother | None = None,
        vla_device: str = "cuda:0",
        wm_device: str = "cuda:4",
        hlp_device: str = "cuda:6",
        action_chunk_interval: int = 50,
        subgoal_refresh_interval: float = 4.0,
        subtask_refresh_interval: float = 8.0,
        cfg_beta: float = 1.7,
        num_denoising_steps: int = 10,
    ):
        self.vla = vla_model
        self.world_model = world_model
        self.hlp = high_level_policy
        self.smoother = action_smoother or ActionSmoother()
        self.vla_device = torch.device(vla_device)
        self.wm_device = torch.device(wm_device) if world_model else None
        self.hlp_device = torch.device(hlp_device) if high_level_policy else None
        self.action_chunk_interval = action_chunk_interval
        self.subgoal_refresh_interval = subgoal_refresh_interval
        self.subtask_refresh_interval = subtask_refresh_interval
        self.cfg_beta = cfg_beta
        self.num_denoising_steps = num_denoising_steps

        self.state = SharedState()
        self._wm_thread: threading.Thread | None = None
        self._hlp_thread: threading.Thread | None = None
        self._last_subgoal_time = 0.0
        self._last_subtask_time = 0.0
        self._step_count = 0

    def start(self, task_instruction: str):
        """Start the inference pipeline.

        Args:
            task_instruction: High-level task description
        """
        self.state.task_instruction = task_instruction
        self.state.subtask_text = task_instruction  # Default subtask = full task
        self.state.running = True
        self.smoother.reset()
        self._step_count = 0

        # Start background threads
        if self.world_model is not None:
            self._wm_thread = threading.Thread(target=self._world_model_loop, daemon=True)
            self._wm_thread.start()
            logger.info("World model thread started")

        if self.hlp is not None:
            self._hlp_thread = threading.Thread(target=self._hlp_loop, daemon=True)
            self._hlp_thread.start()
            logger.info("HLP thread started")

        logger.info(f"Pi07Policy started with task: {task_instruction}")

    def stop(self):
        """Stop the inference pipeline."""
        self.state.running = False
        if self._wm_thread is not None:
            self._wm_thread.join(timeout=5.0)
        if self._hlp_thread is not None:
            self._hlp_thread.join(timeout=5.0)
        logger.info("Pi07Policy stopped")

    def step(self, observation) -> np.ndarray | None:
        """Run one inference step.

        Updates shared observation, requests async updates if needed,
        and returns the next action from the buffer (or generates a new chunk).

        Args:
            observation: Current robot observation

        Returns:
            Action array [action_dim] or None if no action available
        """
        # Update shared observation
        with self.state.obs_lock:
            self.state.current_obs = observation

        # Check if we need a new action chunk
        if self.smoother.buffer_empty or self._step_count % self.action_chunk_interval == 0:
            self._generate_action_chunk(observation)

        # Request async updates based on timing
        now = time.time()
        if now - self._last_subgoal_time > self.subgoal_refresh_interval:
            self.state.new_subgoal_requested = True

        if now - self._last_subtask_time > self.subtask_refresh_interval:
            self.state.new_subtask_requested = True

        self._step_count += 1

        # Return next action from buffer
        return self.smoother.get_action()

    def _generate_action_chunk(self, observation):
        """Generate a new action chunk using the VLA model."""
        # Read current subtask and subgoal images
        with self.state.subtask_lock:
            subtask = self.state.subtask_text

        with self.state.subgoal_lock:
            subgoal_images = self.state.subgoal_images

        # Build observation with subtask in prompt
        # The observation should have the subtask injected into the text prompt
        # This depends on the specific observation format; here we set it as metadata
        if hasattr(observation, "prompt"):
            observation.prompt = f"Task: {self.state.task_instruction}. Subtask: {subtask}."

        # Run VLA
        self.vla.eval()
        with torch.no_grad():
            actions = self.vla.sample_actions(
                self.vla_device,
                observation,
                num_steps=self.num_denoising_steps,
                cfg_beta=self.cfg_beta,
            )

        # Convert to numpy and update smoother
        actions_np = actions.cpu().numpy()[0]  # [action_horizon, action_dim]
        self.smoother.update(actions_np)

    def _world_model_loop(self):
        """Background thread: generate subgoal images with BAGEL."""
        while self.state.running:
            if not self.state.new_subgoal_requested:
                time.sleep(0.1)
                continue

            self.state.new_subgoal_requested = False

            try:
                # Get current observation
                with self.state.obs_lock:
                    obs = self.state.current_obs
                with self.state.subtask_lock:
                    subtask = self.state.subtask_text

                if obs is None:
                    continue

                # Extract observation images for world model
                obs_images = {}
                if hasattr(obs, "images"):
                    for view_name, img_tensor in obs.images.items():
                        from PIL import Image
                        if isinstance(img_tensor, torch.Tensor):
                            img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
                            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                            obs_images[view_name] = Image.fromarray(img_np)

                if not obs_images:
                    continue

                # Generate subgoal images
                subgoals = self.world_model.generate_subgoals(
                    obs_images=obs_images,
                    subtask_text=subtask,
                )

                # Update shared state
                with self.state.subgoal_lock:
                    self.state.subgoal_images = subgoals

                self._last_subgoal_time = time.time()
                logger.debug(f"Generated subgoal images for: {subtask}")

            except Exception as e:
                logger.warning(f"World model error: {e}")
                time.sleep(1.0)

    def _hlp_loop(self):
        """Background thread: generate subtask text with HLP."""
        while self.state.running:
            if not self.state.new_subtask_requested:
                time.sleep(0.1)
                continue

            self.state.new_subtask_requested = False

            try:
                with self.state.obs_lock:
                    obs = self.state.current_obs

                if obs is None:
                    continue

                # Extract images for HLP
                images = []
                if hasattr(obs, "images"):
                    for img_tensor in obs.images.values():
                        if isinstance(img_tensor, torch.Tensor):
                            images.append(img_tensor[:1].to(self.hlp_device))

                if not images:
                    continue

                # Build prompt with history
                with self.state.subtask_lock:
                    history = list(self.state.subtask_history)

                history_str = ""
                if history:
                    history_str = " Previous: " + "; ".join(history[-3:]) + "."

                prompt = f"Task: {self.state.task_instruction}.{history_str} Next subtask:"

                # Tokenize and generate
                # Note: actual tokenization depends on the tokenizer setup
                # This is a placeholder for the generation call
                generated_tokens = self.hlp.generate_subtask(
                    images=images,
                    prompt_tokens=self._tokenize_prompt(prompt),
                    prompt_masks=self._create_prompt_mask(prompt),
                )

                # Decode tokens to text
                new_subtask = self._decode_tokens(generated_tokens[0])

                if new_subtask:
                    with self.state.subtask_lock:
                        old_subtask = self.state.subtask_text
                        self.state.subtask_text = new_subtask
                        if old_subtask != new_subtask:
                            self.state.subtask_history.append(old_subtask)
                            # Also request new subgoal for the new subtask
                            self.state.new_subgoal_requested = True

                    self._last_subtask_time = time.time()
                    logger.info(f"New subtask: {new_subtask}")

            except Exception as e:
                logger.warning(f"HLP error: {e}")
                time.sleep(1.0)

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize a text prompt. Placeholder - uses model tokenizer."""
        # In practice, use the SentencePiece tokenizer
        # For now, return a dummy tensor
        return torch.zeros(1, 128, dtype=torch.long, device=self.hlp_device)

    def _create_prompt_mask(self, prompt: str) -> torch.Tensor:
        """Create prompt mask. Placeholder."""
        return torch.ones(1, 128, dtype=torch.bool, device=self.hlp_device)

    def _decode_tokens(self, tokens: torch.Tensor) -> str:
        """Decode token IDs to text. Placeholder - uses model tokenizer."""
        # In practice, use the SentencePiece tokenizer
        return ""
