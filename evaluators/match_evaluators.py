# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from overrides import overrides

from evaluators.base_evaluator import CorrectnessEvaluator, EvalResult
from libs.multiagent_ft.grader import grade_answer
from libs.multiagent_ft.eval_math import parse_answer
from utils.chat_utils import ChatTurn


class MatchEvaluator(CorrectnessEvaluator):
    support_async: bool = False
    immediate_eval: bool = True

    @overrides
    @staticmethod
    def support_role(role: str) -> bool:
        return role in ["student", "teacher"]

    def __init__(self):
        super().__init__()

    @overrides
    def _eval(
        self,
        turn: ChatTurn,
        metadata: Dict[str, Any] | None = None,
        conv_id: str | None = None,
    ) -> EvalResult:
        assert (
            metadata is not None
        ), f"{self.__class__.__name__} eval must have metadata"

        if "extracted_answer" in turn["info"]:
            # if the extracted_answer is presented, meaning the extraction is done,
            # evalute that instead
            eval_dict = self.em_match(turn["info"]["extracted_answer"], metadata)
        else:
            eval_dict = self.em_match(turn["content"], metadata)
        eval_dict["status_ok"] = True

        return EvalResult(score=eval_dict["metrics"]["em"], info=eval_dict)

    def em_match(
        self, extracted_answer: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError


class MATHMatchEvaluator(MatchEvaluator):

    @overrides
    def em_match(
        self, extracted_answer: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        # get the content in the box
        if not (raw_answer := parse_answer(extracted_answer)):
            raw_answer = extracted_answer.lstrip(r"The answer is $\boxed{").rstrip(
                r"}$."
            )
        return {
            "metrics": {
                "em": float(
                    grade_answer(
                        raw_answer,
                        metadata["answer"],
                    )
                )
            }
        }


def extract_letter_answer(long_str: str) -> str:
    return "".join(
        [x for x in long_str.lstrip("The answer is ").strip() if x.isupper()]
    )


class GPQAMatchEvaluator(MatchEvaluator):
    @overrides
    def em_match(
        self, extracted_answer: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        letter_answer = extract_letter_answer(extracted_answer)
        return {"metrics": {"em": letter_answer == chr(ord("A") + metadata["label"])}}


class MMLUProMatchEvaluator(MatchEvaluator):
    @overrides
    def em_match(
        self, extracted_answer: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        letter_answer = extract_letter_answer(extracted_answer)
        return {"metrics": {"em": letter_answer == metadata["answer"]}}


class ExploreTOMMatchEvaluator(MatchEvaluator):

    @overrides
    def em_match(
        self, extracted_answer: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        match = 0
        target = metadata["expected_answer"]
        prediction = extracted_answer

        if "does not know" in target and " not " in prediction:
            match = 1
        elif "knows about" in target[:11] and " not " not in prediction:
            match = 1
        elif target == "yes" and (
            prediction.split(",")[0] == "yes" or prediction.split(".")[0] == "yes"
        ):
            match = 1
        elif target == "no" and (
            prediction.split(",")[0] == "no" or prediction.split(".")[0] == "no"
        ):
            match = 1
        elif prediction in target or target in prediction:
            match = 1

        return {"metrics": {"em": match}}
