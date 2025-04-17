# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
from typing import Any, Dict, List, Tuple

from overrides import overrides
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.file_utils import load_jsonl_line_by_line
from utils.prompt_utils import get_prompt

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        max_instances: int | None = None,
        mode: str = "train",
        enable_tqdm: bool = False,
        generation_length: int = 128,
        stats_keys: List[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if max_instances is None:
            max_instances = sys.maxsize

        if stats_keys is None:
            stats_keys = ["total_instances", "input_too_long"]

        # mode is one of ["train", "test"]
        assert mode in ["train", "test"]

        self.max_instances = max_instances
        self.mode = mode
        self.enable_tqdm = enable_tqdm
        self.generation_length = generation_length

        # use to report dataset statistics
        self.stats = dict()
        for key in stats_keys:
            self.stats[key] = 0

        self.instances = self.read(file_path)

    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError("the base class should not be used directly")

    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError("the base class should not be used directly")

    def read(self, file_path: str) -> List[Dict[str, Any]]:
        logger.info(f"Reading dataset files at {file_path}")

        all_yield_instances = []

        # load the jsonl dataset
        json_examples = load_jsonl_line_by_line(file_path, self.max_instances)
        iters = tqdm(json_examples) if self.enable_tqdm else json_examples
        for exp in iters:
            if self.mode == "train":
                example_dict = self.get_train_instance(exp)
            elif self.mode == "test":
                example_dict = self.get_test_instance(exp)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            # note that the returned example_dict might be a list of dicts
            all_yield_instances.extend(example_dict)

        logger.info(f"loaded {len(all_yield_instances)} instances")

        self.stats["total_instances"] = len(all_yield_instances)
        self.report_statistics()

        return all_yield_instances

    def report_statistics(self):
        total = self.stats["total_instances"]

        dataset_stats = "-" * 30 + "\nDataset statistics:\n"
        for key, value in self.stats.items():
            if key == "total_instances":
                continue
            dataset_stats += f"{key}: {value/total:.1%} \n"
        dataset_stats += "-" * 30
        logger.info(dataset_stats)

    def __getitem__(self, idx: int):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def truncate(self, max_instances):
        truncated_instances = self.instances[max_instances:]
        self.instances = self.instances[:max_instances]
        return truncated_instances

    def extend(self, instances):
        self.instances.extend(instances)


class TSPromptingDataset(BaseDataset):
    def __init__(
        self,
        mode: str = "test",
        teacher_instruction: str = "teacher_hints",
        student_instruction: str = "student",
        cot_instruction: str = "assistant_cot",
        first_role: str = "student",
        additional_prompt_func_args: Dict[str, Any] | None = None,
        **kwargs,
    ):

        if additional_prompt_func_args is None:
            additional_prompt_func_args = {}

        assert mode == "test", f"{self.__class__.__name__} only supports test mode"

        self.additional_prompt_args = additional_prompt_func_args
        self.teacher_instruction = get_prompt(
            teacher_instruction, **self.j2_prompt_args
        )
        self.student_instruction = get_prompt(
            student_instruction, **self.j2_prompt_args
        )
        self.cot_instruction = get_prompt(cot_instruction, **self.j2_prompt_args)
        self.first_role = first_role

        super().__init__(mode=mode, **kwargs)

    @property
    def j2_prompt_args(self) -> Dict[str, Any]:
        raise ValueError(
            "The `j2_prompt_args` must be overridden in the child classes!"
        )

    def get_prompt_for_example(self, example: Dict[str, Any]) -> str:
        """with the instruction, connect the components of the example, and then connect the examples"""
        # promptify the current example
        prompt, _ = self.promptify_example(example, **self.additional_prompt_args)

        return prompt

    def promptify_example(
        self, example: Dict[str, Any], add_code: bool = True, **kwargs
    ) -> Tuple[str, str]:
        """given an example json dict, return the input and output."""
        raise NotImplementedError(
            "promptify_example must be implemented by the subclass"
        )

    def get_metadata(self, example: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Child class should override this method!")

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise ValueError(f"{self.__class__.__name__} does not support training")

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        metadata = self.get_metadata(example)
        assert "init_prompt" in metadata, "`init_prompt` must be in `metadata`"

        return [
            {
                "init_prompt": metadata["init_prompt"],
                "init_role": self.first_role,
                "teacher_inst": self.teacher_instruction,
                "student_inst": self.student_instruction,
                "cot_inst": self.cot_instruction,
                "metadata": metadata,
            }
        ]
