"""Chunk-wise action smoothing for pi0.7 inference.

Manages an action buffer with temporal smoothing between overlapping
action chunks from consecutive VLA predictions. Handles inference
latency by discarding stale actions.
"""

import collections
import logging
import time

import numpy as np

logger = logging.getLogger("openpi")


class ActionSmoother:
    """Smooths action chunks using linear interpolation in overlap regions.

    When a new action chunk arrives, it is blended with the remaining
    actions from the previous chunk using a linear ramp. This reduces
    jitter at chunk boundaries.

    Args:
        action_horizon: Number of actions per chunk
        action_dim: Dimensionality of each action
        overlap: Number of overlapping steps to blend (0 = no smoothing)
        control_freq: Robot control frequency in Hz
    """

    def __init__(
        self,
        action_horizon: int = 50,
        action_dim: int = 14,
        overlap: int = 10,
        control_freq: float = 30.0,
    ):
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.overlap = min(overlap, action_horizon - 1)
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq

        # Action buffer: deque of (action, timestamp) pairs
        self._buffer: collections.deque = collections.deque()
        self._last_chunk_time: float | None = None

    def reset(self):
        """Reset the buffer."""
        self._buffer.clear()
        self._last_chunk_time = None

    def update(self, new_chunk: np.ndarray, timestamp: float | None = None):
        """Incorporate a new action chunk into the buffer.

        Args:
            new_chunk: [action_horizon, action_dim] array of new actions
            timestamp: When this chunk was computed (for latency compensation)
        """
        ts = timestamp or time.time()

        if len(self._buffer) == 0 or self.overlap == 0:
            # No blending needed - just replace the buffer
            self._buffer.clear()
            for i in range(new_chunk.shape[0]):
                self._buffer.append((new_chunk[i], ts + i * self.dt))
            self._last_chunk_time = ts
            return

        # Blend with remaining buffer actions in the overlap region
        remaining = list(self._buffer)
        self._buffer.clear()

        n_remaining = len(remaining)
        n_overlap = min(self.overlap, n_remaining, new_chunk.shape[0])

        # Non-overlapping old actions (already committed)
        for i in range(n_remaining - n_overlap):
            self._buffer.append(remaining[i])

        # Blended overlap region: linear interpolation
        for i in range(n_overlap):
            old_idx = n_remaining - n_overlap + i
            new_idx = i
            # Ramp from 0 (old) to 1 (new) across the overlap
            alpha = (i + 1) / (n_overlap + 1)
            blended = (1 - alpha) * remaining[old_idx][0] + alpha * new_chunk[new_idx]
            action_ts = ts + new_idx * self.dt
            self._buffer.append((blended, action_ts))

        # New non-overlapping actions
        for i in range(n_overlap, new_chunk.shape[0]):
            self._buffer.append((new_chunk[i], ts + i * self.dt))

        self._last_chunk_time = ts

    def get_action(self, discard_stale: bool = True) -> np.ndarray | None:
        """Get the next action to execute.

        Args:
            discard_stale: If True, skip actions whose timestamp has passed

        Returns:
            Action array [action_dim] or None if buffer is empty
        """
        now = time.time()

        while len(self._buffer) > 0:
            action, ts = self._buffer[0]
            if discard_stale and ts < now - self.dt:
                # This action is stale, skip it
                self._buffer.popleft()
                continue
            self._buffer.popleft()
            return action

        return None

    def peek_actions(self, n: int = 1) -> list[np.ndarray]:
        """Peek at the next n actions without consuming them."""
        result = []
        for i, (action, _) in enumerate(self._buffer):
            if i >= n:
                break
            result.append(action)
        return result

    @property
    def buffer_size(self) -> int:
        """Number of actions remaining in the buffer."""
        return len(self._buffer)

    @property
    def buffer_empty(self) -> bool:
        return len(self._buffer) == 0
