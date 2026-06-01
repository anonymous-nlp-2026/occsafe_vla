"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights

import json
import logging
import time as _time

logger = logging.getLogger(__name__)

@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    occ_data_dir: Optional[str] = None
    occ_resolution: int = 64
    depth_data_dir: Optional[str] = None
    depth_resolution: int = 64

    def __post_init__(self):
        self._occ_index = None
        self._occ_cache = {}
        self._occ_warn_tasks = set()
        if self.occ_data_dir:
            index_path = os.path.join(self.occ_data_dir, "occ_index.json")
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    self._occ_index = json.load(f)
                logger.info("Loaded occ index: %d tasks from %s", len(self._occ_index), index_path)
                self._preload_occ_data()
            else:
                logger.warning("occ_index.json not found at %s, run build_occ_index.py first", index_path)

        # Depth data loading
        self._depth_index = None
        self._depth_cache = {}
        self._depth_warn_tasks = set()
        if self.depth_data_dir:
            depth_index_path = os.path.join(self.depth_data_dir, "depth_index.json")
            if os.path.exists(depth_index_path):
                with open(depth_index_path, "r") as f:
                    self._depth_index = json.load(f)
                logger.info("Loaded depth index: %d tasks from %s", len(self._depth_index), depth_index_path)
                self._preload_depth_data()
            else:
                logger.warning("depth_index.json not found at %s", depth_index_path)

    def _preload_occ_data(self):
        """Preload all BEV occupancy labels into memory.

        Per-sample HDF5 open/close + gzip decompression is the bottleneck (~170% overhead).
        This reads everything once at startup into:
          _occ_cache[task_name][demo_key] = np.ndarray(T, 64, 64), dtype=bool
        Memory: ~490 MB for 10 tasks x 50 demos x ~240 steps x 64x64 bool.
        """
        import h5py

        t0 = _time.time()
        total_files = 0
        total_samples = 0
        total_bytes = 0
        skipped_files = 0

        for task_name, task_info in self._occ_index.items():
            occ_path = os.path.join(self.occ_data_dir, task_info["file"])
            if not os.path.exists(occ_path):
                skipped_files += 1
                continue

            try:
                task_cache = {}
                with h5py.File(occ_path, "r") as f:
                    data_group = f["data"]
                    for demo_key in data_group.keys():
                        bev_path = f"{demo_key}/bev_binary"
                        if bev_path in data_group:
                            arr = data_group[bev_path][:]
                            task_cache[demo_key] = arr
                            total_samples += len(arr)
                            total_bytes += arr.nbytes

                if task_cache:
                    self._occ_cache[task_name] = task_cache
                    total_files += 1
            except Exception as e:
                logger.warning("Failed to preload occ data for %s: %s", task_name, e)

        elapsed = _time.time() - t0
        logger.info(
            "OCC preload complete: %d tasks, %d samples, %.1f MB in %.1fs (skipped %d)",
            total_files, total_samples, total_bytes / 1e6, elapsed, skipped_files,
        )

    def _preload_depth_data(self):
        """Preload all depth labels into memory."""
        import h5py

        t0 = _time.time()
        total_files = 0
        total_samples = 0

        for task_name, task_info in self._depth_index.items():
            depth_path = os.path.join(self.depth_data_dir, task_info["depth_file"])
            if not os.path.exists(depth_path):
                continue

            try:
                task_cache = {}
                with h5py.File(depth_path, "r") as f:
                    data_group = f["data"]
                    for demo_key in data_group.keys():
                        if f"{demo_key}/depth" in data_group:
                            arr = data_group[f"{demo_key}/depth"][:]
                            task_cache[demo_key] = arr
                            total_samples += len(arr)

                if task_cache:
                    self._depth_cache[task_name] = task_cache
                    total_files += 1
            except Exception as e:
                logger.warning("Failed to preload depth data for %s: %s", task_name, e)

        elapsed = _time.time() - t0
        logger.info("Depth preload complete: %d tasks, %d samples in %.1fs", total_files, total_samples, elapsed)

    def _get_occ_label(self, task_name: str, demo_idx: int, step_idx: int) -> Optional[torch.Tensor]:
        """Get BEV occupancy label for a specific (task, demo, step)."""
        if not self._occ_cache:
            return None

        if task_name not in self._occ_cache:
            if task_name not in self._occ_warn_tasks:
                logger.warning("No occ data for task: %s", task_name)
                self._occ_warn_tasks.add(task_name)
            return None

        demo_key = f"demo_{demo_idx}"
        task_cache = self._occ_cache[task_name]
        if demo_key not in task_cache:
            return None

        bev_data = task_cache[demo_key]
        if step_idx >= len(bev_data):
            step_idx = len(bev_data) - 1

        label = torch.from_numpy(bev_data[step_idx].astype(np.float32))
        return label

    def _get_depth_label(self, task_name: str, demo_idx: int, step_idx: int) -> Optional[torch.Tensor]:
        """Get depth label for a specific (task, demo, step)."""
        if not self._depth_cache:
            return None

        if task_name not in self._depth_cache:
            if task_name not in self._depth_warn_tasks:
                logger.warning("No depth data for task: %s", task_name)
                self._depth_warn_tasks.add(task_name)
            return None

        demo_key = f"demo_{demo_idx}"
        task_cache = self._depth_cache[task_name]
        if demo_key not in task_cache:
            return None

        depth_data = task_cache[demo_key]
        if step_idx >= len(depth_data):
            step_idx = len(depth_data) - 1

        label = torch.from_numpy(depth_data[step_idx].astype(np.float32))
        return label

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Transforms a single RLDS batch into the format expected by OpenVLA."""
        dataset_name = rlds_batch.get("dataset_name", "")
        action = np.array(rlds_batch["action"], dtype=np.float32)

        # Get images
        imgs = [Image.fromarray(rlds_batch["observation"]["image_primary"])]
        if "image_wrist" in rlds_batch["observation"] and self.use_wrist_image:
            imgs.append(Image.fromarray(rlds_batch["observation"]["image_wrist"]))

        # Get language instruction
        if "natural_language_instruction" in rlds_batch["task"]:
            lang = rlds_batch["task"]["natural_language_instruction"].decode() if isinstance(
                rlds_batch["task"]["natural_language_instruction"], bytes
            ) else rlds_batch["task"]["natural_language_instruction"]
        else:
            lang = ""

        # Build prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang.lower()}?"},
            {"from": "gpt", "value": self.action_tokenizer(action[:ACTION_DIM])},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # Image transform
        pixel_values = self.image_transform(imgs[0])

        # Labels: only predict action tokens
        labels[: -(ACTION_DIM + 1)] = IGNORE_INDEX

        # Build actions array for chunked prediction
        actions = action.reshape(-1, ACTION_DIM)[:NUM_ACTIONS_CHUNK]
        if len(actions) < NUM_ACTIONS_CHUNK:
            actions = np.pad(actions, ((0, NUM_ACTIONS_CHUNK - len(actions)), (0, 0)))

        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            actions=actions,
            dataset_name=dataset_name,
        )

        # Add wrist image if available
        if len(imgs) > 1:
            return_dict["pixel_values_wrist"] = self.image_transform(imgs[1])

        # Add proprio if enabled
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            return_dict["proprio"] = np.array(rlds_batch["observation"]["proprio"], dtype=np.float32)

        # Add OCC labels
        if self.occ_data_dir and self._occ_cache:
            task_name = rlds_batch.get("task_name", "")
            demo_idx = int(rlds_batch.get("episode_id", 0))
            step_idx = int(rlds_batch.get("step_id", 0))
            occ_label = self._get_occ_label(task_name, demo_idx, step_idx)
            if occ_label is not None:
                return_dict["occ_labels"] = occ_label
            else:
                return_dict["occ_labels"] = torch.zeros(self.occ_resolution, self.occ_resolution)

        # Add depth labels
        if self.depth_data_dir and self._depth_cache:
            task_name = rlds_batch.get("task_name", "")
            demo_idx = int(rlds_batch.get("episode_id", 0))
            step_idx = int(rlds_batch.get("step_id", 0))
            depth_label = self._get_depth_label(task_name, demo_idx, step_idx)
            if depth_label is not None:
                return_dict["depth_labels"] = depth_label
            else:
                return_dict["depth_labels"] = torch.zeros(self.depth_resolution, self.depth_resolution)

        return return_dict


class RLDSDataset(IterableDataset):
    """Wraps an RLDS TFDS Pipeline as a PyTorch IterableDataset."""

    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        self.data_root_dir = data_root_dir
        self.data_mix = data_mix
        self.batch_transform = batch_transform
        self.dataset_length = 0

        # Configure RLDS dataset
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            mixture_spec = [(self.data_mix, 1.0)]

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            mixture_spec,
            data_root_dir,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )

        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=NUM_ACTIONS_CHUNK,
                future_action_window_size=0,
                skip_unlabeled=True,
                goal_relabeling_strategy=None,
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                image_augment_kwargs=dict(
                    random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=["random_resized_crop", "random_brightness", "random_contrast",
                                   "random_saturation", "random_hue"],
                ) if image_aug else {},
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            train=train,
        )

        self.dataset = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        if len(rlds_config["dataset_kwargs_list"]) == 1:
            return make_single_dataset(
                rlds_config["dataset_kwargs_list"][0],
                train=rlds_config["train"],
                traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
                frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
            )
        return make_interleaved_dataset(
            rlds_config["dataset_kwargs_list"],
            rlds_config["sample_weights"],
            train=rlds_config["train"],
            shuffle_buffer_size=rlds_config["shuffle_buffer_size"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        return_dict["occ_labels"] = torch.zeros(64, 64)
        return_dict["depth_labels"] = torch.zeros(64, 64)
        return return_dict
