# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from io import StringIO
from typing import Any, Dict, List

from overrides import overrides

from evaluators.base_evaluator import (
    CorrectnessEvaluator,
    EvalResult,
    PromptingEvaluator,
)
from utils.chat_utils import ChatTurn, generate_next_turn
from utils.eval_utils import behavior_quality, social_behavior_metrics


class ExtractionPromptingEvaluator(PromptingEvaluator):

    immediate_eval: bool = True

    sys_prompt = """\
You are an assistant that is helping an user to identify the intention of certain \
responses in a conversation. More specifically, you will help extracting which answer \
the response is submiting as the final answer, or say "not sure yet" if it seems like \
there is no explict answer included in the response.
"""

    @overrides
    @staticmethod
    def support_role(role: str) -> bool:
        return role in ["teacher", "student"]

    @overrides
    def get_evaluator_metrics(self) -> List[str]:
        parent_metrics = super().get_evaluator_metrics()

        return parent_metrics + [
            "agreement",
            "agreement_correctness",
            "valid_answer_rate",
            "persuade_rate",
            "correct_persuade_rate",
            "incorrect_persuade_rate",
            "unsure_persuade_rate",
            "persuasion_quality",
            "assert_rate",
            "correct_assert_rate",
            "incorrect_assert_rate",
            "unsure_assert_rate",
        ]

    @overrides
    async def _async_eval(
        self,
        turn: ChatTurn,
        metadata: Dict[str, Any] | None = None,
        conv_id: str | None = None,
    ) -> EvalResult:
        rater_role = await self.rater_factory.get_chat_role(
            sys_prompt=self.sys_prompt, uuid=conv_id
        )
        rating_prompt = self.promptify_eval_response(
            turn["content"], turn["role"], metadata
        )
        prompt_turn = ChatTurn(role=rater_role.name, content=rating_prompt, info={})
        generate_turn = await generate_next_turn(
            [prompt_turn],
            rater_role,
            self.request_timeout,
            max_tokens=self.max_tokens,
            use_matrix=self.use_matrix,
            endpoint_cache=self.rater_factory.endpoint_cache,
        )

        if generate_turn["info"]["status_ok"]:
            extracted_answer = self.normalize_answer(generate_turn["content"])
            score = self.is_valid_answer(extracted_answer)
        else:
            extracted_answer = "Error: extraction network error"
            score = 0.0

        # put the extracted answer back in the turn so that correctness evaluators know
        # to eval the extracted answer and not the whole turn instead
        turn["info"]["extracted_answer"] = extracted_answer

        return EvalResult(
            score=score,
            info={
                "prompt": prompt_turn["content"],
                "response": generate_turn["content"],
                "extracted_answer": extracted_answer,
                "agreement": False,  # by default, will get changed when stop_conv is called
                "status_ok": generate_turn["info"]["status_ok"],
            },
        )

    @overrides
    def _get_conv_metrics(
        self, conv: List[ChatTurn], metadata: Dict[str, Any] | None = None
    ) -> Dict[str, float | int]:
        # get the common metrics and pop the unused ones
        metric_dict = super()._get_conv_metrics(conv, metadata)

        # check what percentage of extracted answers are valid
        total = 0
        valid_answer = 0
        agreement = False
        agreement_correct = False
        for i, turn in enumerate(conv):
            if (
                i == 0  # first turn doesn't apply
                or "annotation" not in turn["info"]
                or turn["info"]["annotation"][self.name] is None
            ):
                continue

            total += 1
            valid_answer += turn["info"]["annotation"][self.name]["score"]

            if turn["info"]["annotation"][self.name]["info"]["agreement"]:
                agreement = True

                # check if this answer is also correct
                for rater_name in turn["info"]["annotation"].keys():
                    if (
                        CorrectnessEvaluator.is_correctness_evaluator(rater_name)
                        and turn["info"]["annotation"][rater_name]["score"] == 1.0
                    ):
                        agreement_correct = True

        answer_valid_rate = (valid_answer / total) if total > 0 else 0.0
        metric_dict.update(
            {
                "valid_answer_rate": answer_valid_rate,
                "agreement": agreement,
                "agreement_correctness": agreement_correct,
            }
        )

        social_metrics = self.compute_social_metrics(conv, not_sure_score=0.5)
        metric_dict.update(social_metrics)

        return metric_dict

    @overrides
    def stop_conv(self, conv: List[ChatTurn]) -> bool:
        belief_dict = {}

        for turn in conv[::-1]:
            try:
                if turn["info"]["annotation"][self.name]["score"] == 0.0:
                    belief_dict[turn["role"]] = None
                else:
                    belief_dict[turn["role"]] = turn["info"]["annotation"][self.name][
                        "info"
                    ]["extracted_answer"]
            except KeyError:
                belief_dict[turn["role"]] = None

            all_roles = list(belief_dict.keys())
            if (
                len(all_roles) > 1
                and all(self.is_valid_answer(belief_dict[role]) for role in all_roles)
                and self.answer_equals(
                    belief_dict[all_roles[0]], belief_dict[all_roles[1]]
                )
            ):
                # set both turns to have the agreement so we know which ones agree
                # turn["info"]["annotation"][self.name]["info"]["agreement"] = True
                conv[-1]["info"]["annotation"][self.name]["info"]["agreement"] = True
                return True

        return False

    @staticmethod
    def compute_social_metrics(
        conv_list: List[ChatTurn], not_sure_score: float
    ) -> Dict:
        assertive_turn_scores, persuasive_turn_scores, persuasion_quality_scores = (
            social_behavior_metrics(conv_list, not_sure_score)
        )

        # persuasiveness rates and quality over turns for this conversation
        persuade_rate = len(persuasive_turn_scores) / (len(conv_list) - 1)
        persuasion_quality_scores = (
            sum(persuasion_quality_scores) / len(persuasion_quality_scores)
            if len(persuasion_quality_scores) > 0
            else 0.0
        )
        correct_persuade_rate, incorrect_persuade_rate, unsure_persuade_rate = (
            behavior_quality(persuasive_turn_scores, not_sure_score)
        )

        # assertiveness rates over turns for this conversation
        assert_rate = len(assertive_turn_scores) / (len(conv_list) - 1)
        correct_assert_rate, incorrect_assert_rate, unsure_assert_rate = (
            behavior_quality(assertive_turn_scores, not_sure_score)
        )

        return {
            "persuade_rate": persuade_rate,
            "correct_persuade_rate": correct_persuade_rate,
            "incorrect_persuade_rate": incorrect_persuade_rate,
            "unsure_persuade_rate": unsure_persuade_rate,
            "persuasion_quality": persuasion_quality_scores,
            "assert_rate": assert_rate,
            "correct_assert_rate": correct_assert_rate,
            "incorrect_assert_rate": incorrect_assert_rate,
            "unsure_assert_rate": unsure_assert_rate,
        }

    def answer_equals(self, ans1: str, ans2: str) -> bool:
        return ans1 == ans2

    def promptify_eval_response(
        self,
        response: str,
        role: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError("Base class should not not called!")

    def normalize_answer(self, response: str) -> str:
        raise NotImplementedError("Base class should not not called!")

    def is_valid_answer(self, answer: str | None) -> bool:
        raise NotImplementedError("Base class should not not called!")


class MATHExtractionPromptingEvaluator(ExtractionPromptingEvaluator):
    @overrides
    def promptify_eval_response(
        self,
        response: str,
        role: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        assert metadata is not None and "problem" in metadata

        # build the prompt
        with StringIO(self.sys_prompt) as prompt_builder:
            prompt_builder.write(
                f"**This is the original question:**\n{metadata['problem']}\n\n"
                "**This is the response you need to extract answer from:**\n\n"
            )
            prompt_builder.write(response)
            prompt_builder.write(
                '\n\n**Extract the answer in $\\boxed{}$ format, or say "not sure yet".**'
            )

            return prompt_builder.getvalue()

    @overrides
    def normalize_answer(self, response: str) -> str:
        if "not sure yet" in response.lower():
            return "not sure yet"

        pattern = r"\$\\boxed\{.*\}\$"
        matches = re.findall(pattern, response)

        if len(matches) > 0:
            return f"The answer is {matches[0]}."

        return f"Error: {response}"

    @overrides
    def is_valid_answer(self, answer: str | None) -> bool:
        if answer is None:
            return False

        if answer.startswith("Error"):
            return False

        if answer == "not sure yet":
            return False

        return True


class MMLUProExtractionPromptingEvaluator(MATHExtractionPromptingEvaluator):
    @overrides
    def promptify_eval_response(
        self,
        response: str,
        role: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        assert metadata is not None and "question" in metadata

        options_str = "\n".join(
            [f"({letter}) {option}" for letter, option in metadata["choices"].items()]
        )

        # build the prompt
        with StringIO(self.sys_prompt) as prompt_builder:
            prompt_builder.write(
                f"**This is the original question:**\n{metadata['question']}\n\n"
                + f"\n\n**And the options are:**\n{options_str}\n\n"
                + "**This is the response you need to extract answer from:**\n\n"
            )
            prompt_builder.write(response)
            prompt_builder.write(
                '\n\n**Extract the answer in ([A-Z]) format, or say "not sure yet".**'
            )

            return prompt_builder.getvalue()

    @overrides
    def normalize_answer(self, response: str) -> str:
        if "not sure yet" in response.lower():
            return "not sure yet"

        pattern = r"\([A-Z]\)"
        matches = re.findall(pattern, response)

        if len(matches) > 0:
            return f"The answer is {matches[0]}."

        return f"Error: {response}"


class GPQAExtractionPromptingEvaluator(MMLUProExtractionPromptingEvaluator):
    @overrides
    def promptify_eval_response(
        self,
        response: str,
        role: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        assert metadata is not None and "Question" in metadata

        options_str = "\n".join(
            [
                f"(A) {metadata['choices'][0]}",
                f"(B) {metadata['choices'][1]}",
                f"(C) {metadata['choices'][2]}",
                f"(D) {metadata['choices'][3]}",
            ]
        )

        # build the prompt
        with StringIO(self.sys_prompt) as prompt_builder:
            prompt_builder.write(
                f"**This is the original question:**\n{metadata['Question']}\n\n"
                + f"\n\n**And the options are:**\n{options_str}\n\n"
                + "**This is the response you need to extract answer from:**\n\n"
            )
            prompt_builder.write(response)
            prompt_builder.write(
                '\n\n**Extract the answer in ([A-Z]) format, or say "not sure yet".**'
            )

            return prompt_builder.getvalue()


class ExploreTOMExtractionPromptingEvaluator(ExtractionPromptingEvaluator):
    @overrides
    def promptify_eval_response(
        self,
        response: str,
        role: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        assert metadata is not None and "question" in metadata

        # build the prompt
        with StringIO(self.sys_prompt) as prompt_builder:
            prompt_builder.write(
                f"**This is the original story and question:**\nStory: {metadata['story_structure']}\nQuestion: {metadata['question']}\n\n"
                "**This is the response you need to extract an answer from:**\n\n"
            )
            prompt_builder.write(response)
            prompt_builder.write(
                '\n\n**Extract the final answer which should be less than five words, or say "not sure yet".**'
            )

            return prompt_builder.getvalue()

    @overrides
    def normalize_answer(self, response: str) -> str:
        if "not sure yet" in response.lower():
            return "not sure yet"

        pattern = (
            r"(?:Short Answer: (?:The final answer is )?|The final answer is )?(.*)"
        )
        matches = re.findall(pattern, response)

        if len(matches) > 0:
            return f"{matches[0].lower().replace('the', '').rstrip('.').strip()}"

        return f'Error: no answer found in "{response}"'

    @overrides
    def is_valid_answer(self, answer: str | None) -> bool:
        if answer is None:
            return False

        if answer.startswith("Error"):
            return False

        if answer == "not sure yet":
            return False

        return True


class HiTOMExtractionPromptingEvaluator(ExploreTOMExtractionPromptingEvaluator):
    @overrides
    def normalize_answer(self, response: str) -> str:
        if "not sure yet" in response.lower():
            return "not sure yet"

        pattern = (
            r"(?:Short Answer: (?:The final answer is )?|The final answer is )?(.*)"
        )
        matches = re.findall(pattern, response)

        if len(matches) > 0:
            return f"{matches[0].lower().replace('the', '').rstrip('.').strip().replace(' ', '_')}"

        return f'Error: no answer found in "{response}"'
