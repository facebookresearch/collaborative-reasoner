# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from overrides import overrides

from lightning_modules.datasets import BaseDataModule
from lightning_modules.datasets.base_dataset import TSPromptingDataset


class GPQATSPromptingDataset(TSPromptingDataset):
    @property
    @overrides
    def j2_prompt_args(self) -> Dict[str, Any]:
        return {
            "task_name": "PhD-level biology, physics, and chemistry",
            "task_specific_inst": 'To give a final answer, do it in the format of "The correct answer is (insert answer here)", such as "The correct answer is (B)"',
        }

    @overrides
    def get_metadata(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # process examples of GPQA
        metadata = example.copy()

        init_student_prompt = (
            f"I'm trying to solve this problem: \"{example['Question']}\"\n"
            + "And the choices are:\n"
            + "\n".join(
                [
                    f"(A) {example['choices'][0]}",
                    f"(B) {example['choices'][1]}",
                    f"(C) {example['choices'][2]}",
                    f"(D) {example['choices'][3]}",
                ]
            )
        )

        assert self.first_role == "student", "Only students are supported!"
        metadata["init_prompt"] = init_student_prompt

        return metadata


class GPQATSDataModule(BaseDataModule):
    @overrides
    def setup(self, stage: str | None = None):
        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        assert stage in ["validate"]

        if self.val_data is None:
            val_data = GPQATSPromptingDataset(mode="test", **self.val_set_init_args)
            self.val_data = val_data
