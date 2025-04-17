# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict

from overrides import overrides

from lightning_modules.datasets import BaseDataModule
from lightning_modules.datasets.base_dataset import TSPromptingDataset


class ExploreTOMTSPromptingDataset(TSPromptingDataset):
    def __init__(
        self,
        mode: str = "test",
        additional_prompt_func_args: Dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            mode=mode, additional_prompt_func_args=additional_prompt_func_args, **kwargs
        )

    @property
    @overrides
    def j2_prompt_args(self) -> Dict[str, Any]:
        return {
            "task_name": "theory-of-mind reasoning",
            "task_specific_inst": 'Put your final answer to the question at the end as "Short Answer: {answer}"',
        }

    @overrides
    def get_metadata(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # process examples of ExploreToM
        metadata = example.copy()

        init_student_prompt = f"I'm trying to answer this question based on the following story:\nStory: {example['story_structure']}\nQuestion: {example['question']}"

        init_teacher_prompt = f"I would like you to try to answer a question based on the following story: \nStory: {example['story_structure']}\nQuestion: {example['question']}"

        assert self.first_role in [
            "student",
            "teacher",
        ], "Only student and teacher roles are supported!"

        if self.first_role == "student":
            metadata["init_prompt"] = init_student_prompt
        elif self.first_role == "teacher":
            metadata["init_prompt"] = init_teacher_prompt

        return metadata


class ExploreTOMTSDataModule(BaseDataModule):

    @overrides
    def setup(self, stage: str | None = None):  # type: ignore
        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        assert stage in ["validate"]

        if self.val_data is None:
            val_data = ExploreTOMTSPromptingDataset(
                mode="test", **self.val_set_init_args
            )
            self.val_data = val_data
