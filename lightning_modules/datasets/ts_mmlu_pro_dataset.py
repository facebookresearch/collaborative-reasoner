# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from overrides import overrides

from lightning_modules.datasets import BaseDataModule
from lightning_modules.datasets.base_dataset import TSPromptingDataset


class MMLUProTSPromptingDataset(TSPromptingDataset):
    @property
    @overrides
    def j2_prompt_args(self) -> Dict[str, Any]:
        return {
            "task_name": "reasoning-focused question answering",
            "task_specific_inst": 'To give a final answer, do it in the format of "The correct answer is (insert answer here)", such as "The correct answer is (B)"',
        }

    @overrides
    def get_metadata(self, example: Dict[str, Any]) -> Dict[str, Any]:
        metadata = example.copy()
        metadata["choices"] = {
            chr(65 + i): option for i, option in enumerate(metadata["options"])
        }

        init_student_prompt = (
            f"I'm trying to solve this problem: \"{metadata['question']}\"\n"
            + "And the choices are:\n"
            + "\n".join(
                [
                    f"({letter}) {option}"
                    for letter, option in metadata["choices"].items()
                ]
            )
        )

        assert self.first_role == "student", "Only students are supported!"
        metadata["init_prompt"] = init_student_prompt

        return metadata


class MMLUProTSDataModule(BaseDataModule):
    dataset_cls = MMLUProTSPromptingDataset
