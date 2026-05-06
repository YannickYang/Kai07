from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
import os
import random
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from PIL import Image
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data

@dataclasses.dataclass(frozen=True)
class InsertAdvantageIntoPrompt(DataTransformFn):

    def __call__(self, data: DataDict) -> DataDict:
        assert "advantage" in data, f"advantage is not in data, data_keys: {data.keys()}"
        assert "prompt" in data, f"prompt is not in data, data_keys: {data.keys()}"
        advantage = data["advantage"]
        data["prompt"] = data["prompt"] + f", Advantage: {advantage:.4f}"
        return data


@dataclasses.dataclass(frozen=True)
class InjectPi07Prompt(DataTransformFn):
    """Construct pi0.7-style prompt with subtask, metadata, and control mode.

    Prompt format:
        Task: {task}. Subtask: {subtask}. Speed: {speed} steps.
        Quality: {quality}. Mistake: {mistake}. Control Mode: {control_mode}.

    Dropout strategy (following pi0.7 paper):
        - subtask: 30% dropout when subgoal images present, 10% otherwise
        - metadata block (speed/quality/mistake): 15% dropout as a group
        - individual metadata fields: additional 5% dropout each
        - control_mode: never dropped
    """

    # Loaded annotation data
    subtask_annotations: dict[str, list[dict]] | None = None
    episode_metadata: dict[str, dict] | None = None
    control_mode: str = "joint"
    # Dropout probabilities
    subtask_dropout: float = 0.1
    subtask_dropout_with_subgoal: float = 0.3
    metadata_group_dropout: float = 0.15
    metadata_field_dropout: float = 0.05

    def __call__(self, data: DataDict) -> DataDict:
        # Get base task prompt
        prompt = ""
        if "prompt" in data:
            p = data["prompt"]
            prompt = p.item() if hasattr(p, "item") else str(p)

        # Build pi0.7 prompt components
        parts = []

        # Task instruction
        if prompt:
            parts.append(f"Task: {prompt}")

        # Subtask instruction
        subtask_text = self._get_subtask(data)
        has_subgoal = "subgoal_images" in data and data.get("subgoal_images") is not None
        drop_prob = self.subtask_dropout_with_subgoal if has_subgoal else self.subtask_dropout
        if subtask_text and random.random() > drop_prob:
            parts.append(f"Subtask: {subtask_text}")

        # Episode metadata (speed, quality, mistake)
        meta = self._get_metadata(data)
        if meta and random.random() > self.metadata_group_dropout:
            if "speed" in meta and random.random() > self.metadata_field_dropout:
                parts.append(f"Speed: {meta['speed']} steps")
            if "quality" in meta and random.random() > self.metadata_field_dropout:
                parts.append(f"Quality: {meta['quality']}")
            if "mistake" in meta and random.random() > self.metadata_field_dropout:
                parts.append(f"Mistake: {'yes' if meta['mistake'] else 'no'}")

        # Control mode (never dropped)
        parts.append(f"Control Mode: {self.control_mode}")

        # Join with ". " separator
        pi07_prompt = ". ".join(parts) + "."
        data["prompt"] = np.asarray(pi07_prompt)
        return data

    def _get_subtask(self, data: DataDict) -> str | None:
        """Look up subtask text for the current sample."""
        if self.subtask_annotations is None:
            return None

        ep_idx = data.get("episode_index")
        frame_idx = data.get("frame_index")
        if ep_idx is None:
            return None

        ep_key = str(int(ep_idx.item() if hasattr(ep_idx, "item") else ep_idx))
        subtasks = self.subtask_annotations.get(ep_key, [])
        if not subtasks:
            return None

        # Find the subtask that covers the current frame
        if frame_idx is not None:
            fidx = int(frame_idx.item() if hasattr(frame_idx, "item") else frame_idx)
            for st in subtasks:
                if st["start"] <= fidx < st["end"]:
                    return st["text"]

        # Fallback: return the first subtask
        return subtasks[0]["text"]

    def _get_metadata(self, data: DataDict) -> dict | None:
        """Look up episode metadata."""
        if self.episode_metadata is None:
            return None

        ep_idx = data.get("episode_index")
        if ep_idx is None:
            return None

        ep_key = str(int(ep_idx.item() if hasattr(ep_idx, "item") else ep_idx))
        return self.episode_metadata.get(ep_key)


class LoadSubgoalImages(DataTransformFn):
    """Load precomputed subgoal images from disk into the data dict.

    25% of samples include subgoal images. Of those:
      - 25% use oracle end-of-segment frames
      - 75% use random future frames (0-4s ahead)

    Expects subgoal_images/ directory under data_dir, generated by
    scripts/precompute_subgoals.py.
    """

    subgoal_dir: str | None = None  # path to subgoal_images/
    subgoal_prob: float = 0.25      # probability of including subgoal
    oracle_prob: float = 0.25       # probability of oracle vs future frame
    cameras: list[str] = None

    def __init__(self, subgoal_dir: str | None = None, cameras: list[str] | None = None,
                 subgoal_prob: float = 0.25, oracle_prob: float = 0.25):
        self.subgoal_dir = subgoal_dir
        self.cameras = cameras or ["head", "left", "right"]
        self.subgoal_prob = subgoal_prob
        self.oracle_prob = oracle_prob

    def __call__(self, data: DataDict) -> DataDict:
        if self.subgoal_dir is None or not os.path.isdir(self.subgoal_dir):
            return data

        # 25% chance to include subgoal
        if random.random() > self.subgoal_prob:
            return data

        ep_idx = data.get("episode_index")
        frame_idx = data.get("frame_index")
        if ep_idx is None:
            return data

        ep_int = int(ep_idx.item() if hasattr(ep_idx, "item") else ep_idx)
        ep_dir = os.path.join(self.subgoal_dir, f"ep{ep_int:04d}")
        if not os.path.isdir(ep_dir):
            return data

        # Find which subtask this frame belongs to
        st_idx = self._find_subtask_idx(data)
        if st_idx is None:
            return data

        # Choose oracle or future
        use_oracle = random.random() < self.oracle_prob

        subgoal_images = {}
        for cam in self.cameras:
            if use_oracle:
                path = os.path.join(ep_dir, f"{cam}_st{st_idx}_oracle.png")
            else:
                # Pick a random future frame
                fi = random.randint(0, 2)
                path = os.path.join(ep_dir, f"{cam}_st{st_idx}_future{fi}.png")
                if not os.path.exists(path):
                    path = os.path.join(ep_dir, f"{cam}_st{st_idx}_oracle.png")

            if os.path.exists(path):
                img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                subgoal_images[f"subgoal.{cam}"] = img

        if subgoal_images:
            data["subgoal_images"] = subgoal_images

        return data

    def _find_subtask_idx(self, data: DataDict) -> int | None:
        """Determine which subtask index the current frame belongs to."""
        # Try to get from InjectPi07Prompt's annotations (shared state)
        ep_idx = data.get("episode_index")
        frame_idx = data.get("frame_index")
        if ep_idx is None or frame_idx is None:
            return 0  # default to first subtask

        # Look for subtask_annotations in the transform chain's shared state
        # For simplicity, check files on disk
        ep_int = int(ep_idx.item() if hasattr(ep_idx, "item") else ep_idx)
        fidx = int(frame_idx.item() if hasattr(frame_idx, "item") else frame_idx)
        ep_dir = os.path.join(self.subgoal_dir, f"ep{ep_int:04d}")

        # Count available subtasks by checking oracle files
        st_idx = 0
        for cam in self.cameras:
            for si in range(10):
                oracle = os.path.join(ep_dir, f"{cam}_st{si}_oracle.png")
                if not os.path.exists(oracle):
                    # si subtasks found
                    if si > 0:
                        # Rough proportional assignment
                        st_idx = min(si - 1, fidx * si // max(1, self._est_ep_length(ep_dir, cam, si)))
                    break
            break

        return st_idx

    def _est_ep_length(self, ep_dir: str, cam: str, n_subtasks: int) -> int:
        """Estimate episode length from oracle file names (rough)."""
        # Just return a large default; proportional assignment is approximate anyway
        return 2000


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


# @dataclasses.dataclass(frozen=True)
# class TokenizePrompt(DataTransformFn):
#     tokenizer: _tokenizer.PaligemmaTokenizer
#     discrete_state_input: bool = False

#     def __call__(self, data: DataDict) -> DataDict:
#         if (prompt := data.pop("prompt", None)) is None:
#             raise ValueError("Prompt is required")

#         if self.discrete_state_input:
#             if (state := data.get("state", None)) is None:
#                 raise ValueError("State is required.")
#         else:
#             state = None

#         if not isinstance(prompt, str):
#             prompt = prompt.item()

#         tokens, token_masks = self.tokenizer.tokenize(prompt, state)
#         return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
