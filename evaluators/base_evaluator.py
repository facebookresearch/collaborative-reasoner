# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
from typing import Any, Dict, List, Optional, TypedDict, final

from overrides import overrides

from utils.chat_utils import ChatRoleFactory, ChatTurn, generate_next_turn
from utils.eval_utils import agreement_eval, n_turns_to_success, rater_err_out
from utils.http_utils import DEFAULT_REQUEST_TIMEOUT


class EvalResult(TypedDict):
    score: float
    info: Dict[str, Any]


class Evaluator:
    """
    Evaluates the quality of a generated response from the conversational LLM agent.

    For each turn, the model will call `eval()` or `async_eval()` for this turn,
        depending on whether `support_async`. Then the actual `_eval()` or
        `_async_eval()` functions in the overriden child class will get called and eval
        the turns.

    At the end of each conversation, `get_conv_metrics()` will be called to calcuate
        the conversation-wise metrics for each metrics that's provided by the rater.
    """

    # This attribute will be used by the Trainer to check if an evaluator supports
    # evaluation in async mode.
    support_async: bool = False
    # whether the eval on this turn should be done readily after each turn
    immediate_eval: bool = False

    def __init__(self):
        self.name = self.__class__.__name__

    def stop_conv(self, conv: List[ChatTurn]) -> bool:
        """Whether this conversation should end based on the turns so far."""
        return False  # default value

    @staticmethod
    def get_metric_from_dict(metric_dict: Dict[str, Any], metric_name: str) -> float:
        metric_full_name = list(
            filter(lambda x: x.endswith(metric_name), metric_dict.keys())
        )
        if len(metric_full_name) == 0:
            raise ValueError(
                f"Key {metric_name} is not found in metric_dict {list(metric_dict.keys())}"
            )
        elif len(metric_full_name) > 1:
            raise ValueError(
                f"Key {metric_name} is found in more than one keys in metric_dict {list(metric_dict.keys())}"
            )
        else:
            return metric_dict[metric_full_name[0]]

    @staticmethod
    def support_role(role: str) -> bool:
        # all roles are supported by default
        return True

    @property
    def evaluator_metrics(self) -> List[str]:
        # add the rater names to the evaluator metrics
        base_names = self.get_evaluator_metrics()
        return [self.get_rater_metric_name(x) for x in base_names]

    def get_evaluator_metrics(self) -> List[str]:
        return [
            "rated_turns",
            "rater_err_rate",
        ]

    def get_rater_metric_name(self, base_metric_name: str) -> str:
        return self.name + "-" + base_metric_name

    @final
    def eval(
        self,
        turn: ChatTurn,
        metadata: Dict[str, Any] | None = None,
        conv_id: str | None = None,
    ) -> EvalResult | None:
        """NOTE: this method should not be overriden, only the `_eval()` method."""
        if self.support_role(turn["role"]):
            return self._eval(turn, metadata, conv_id)
        else:
            return None

    def _eval(
        self,
        turn: ChatTurn,
        metadata: Dict[str, Any] | None = None,
        conv_id: str | None = None,
    ) -> EvalResult | None:
        if self.support_async:
            raise ValueError(f"{self.__class__.__name__} uses async eval!")
        else:
            raise NotImplementedError("Base method should be overriden and not called!")

    @final
    async def async_eval(
        self,
        turn: ChatTurn,
        metadata: Dict[str, Any] | None = None,
        conv_id: str | None = None,
    ) -> EvalResult | None:
        """NOTE: this method should not be overriden, only the `_async_eval()` method."""
        if self.support_role(turn["role"]):
            return await self._async_eval(turn, metadata, conv_id)
        else:
            return None

    async def _async_eval(
        self,
        turn: ChatTurn,
        metadata: Dict[str, Any] | None = None,
        conv_id: str | None = None,
    ) -> EvalResult | None:
        if not self.support_async:
            raise ValueError(f"{self.__class__.__name__} does not support async eval!")
        else:
            raise NotImplementedError("Base method should be overriden and not called!")

    @final
    def get_conv_metrics(
        self, conv: List[ChatTurn], metadata: Dict[str, Any] | None = None
    ) -> Dict[str, float | int]:
        # call the actual function that computes the functions
        metric_dict = self._get_conv_metrics(conv, metadata)

        # add some postprocessing on the metrics name
        new_metric_dict = {}
        for metric_name, metric_info in metric_dict.items():
            new_metric_dict[self.get_rater_metric_name(metric_name)] = metric_info

        # add some quality assurance
        assert set(new_metric_dict.keys()) == set(
            self.evaluator_metrics
        ), f"Delivered metrics ({set(new_metric_dict.keys())}), but promised ({set(self.evaluator_metrics)})"

        return new_metric_dict

    def _get_conv_metrics(
        self, conv: List[ChatTurn], metadata: Dict[str, Any] | None = None
    ) -> Dict[str, float | int]:
        assert metadata is not None and "max_turns" in metadata

        # the base class only handles the basic stats
        rated_turns, rater_err_turns = rater_err_out(conv, self.name)
        agreement, agreement_correctness = agreement_eval(conv, self.name)
        return {
            "rated_turns": rated_turns,
            "rater_err_rate": (
                (rater_err_turns / rated_turns) if rated_turns > 0 else 0.0
            ),
        }


class CorrectnessEvaluator(Evaluator):
    """Evaluate whether the turn gives a correct answer."""

    @overrides
    def get_evaluator_metrics(self) -> List[str]:
        parent_metrics = super().get_evaluator_metrics()

        return parent_metrics + [
            "n_turns_to_success",
            "final_success_rate",
        ]

    @staticmethod
    def is_correctness_evaluator(evaluator_name: str) -> bool:
        if evaluator_name.endswith("MatchEvaluator") or evaluator_name.endswith(
            "ExecEvaluator"
        ):
            return True
        else:
            return False

    def _get_conv_metrics(
        self, conv: List[ChatTurn], metadata: Dict[str, Any] | None = None
    ) -> Dict[str, float | int]:
        assert metadata is not None and "max_turns" in metadata

        # get the common metrics
        metric_dict = super()._get_conv_metrics(conv, metadata)

        # get the correctness-based metrics
        n_turns = n_turns_to_success(conv, self.name)
        metric_dict.update(
            {
                "n_turns_to_success": (
                    n_turns if n_turns != -1 else metadata["max_turns"]
                ),
                "final_success_rate": 1.0 if n_turns != -1 else 0.0,
            }
        )

        return metric_dict


class PromptingEvaluator(Evaluator):
    """Evaluates the quality of a generated response via prompting."""

    sys_prompt: str = ""
    support_async: bool = True

    def __init__(
        self,
        rater_temp: float,
        model_name: str,
        app_name: str,
        max_tokens: int = 1024,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
        use_matrix: bool = True,
        ray_cluster_id: Optional[str] = None,
        ray_rdv_dir: Optional[str] = None,
    ):
        super().__init__()
        self.name = (
            model_name.split("/")[-1].replace("-", "_").replace(".", "_")
            + "-"
            + self.name
        )

        self.rater_temp = rater_temp
        self.request_timeout = request_timeout
        self.max_tokens = max_tokens
        self.use_matrix = use_matrix

        self.rater_factory = ChatRoleFactory(
            name="rater",
            app_name=app_name,
            model_name=model_name,
            sampling_temp=rater_temp,
            sys_prompt=self.sys_prompt,
            ray_cluster_id=ray_cluster_id,
            ray_rdv_dir=ray_rdv_dir,
        )

    def promptify_eval_response(
        self,
        response: str,
        role: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError("Base method should be overriden and not called!")

    def extract_score(self, rating_str: str) -> float:
        raise NotImplementedError("Base method should be overriden and not called!")

    async def _async_eval(
        self,
        turn: ChatTurn,
        metadata: Dict[str, Any] | None = None,
        conv_id: str | None = None,
    ) -> EvalResult:
        rater_role = await self.rater_factory.get_chat_role(uuid=conv_id)
        rating_prompt = self.promptify_eval_response(
            turn["content"], turn["role"], metadata
        )
        prompt_turn = ChatTurn(role=rater_role.name, content=rating_prompt, info={})
        generate_turn = await generate_next_turn(
            [prompt_turn],
            rater_role,
            self.request_timeout,
            use_matrix=self.use_matrix,
            endpoint_cache=self.rater_factory.endpoint_cache,
        )

        if generate_turn["info"]["status_ok"]:
            score = self.extract_score(generate_turn["content"])
        else:
            score = 0.0

        return EvalResult(
            score=score,
            info={
                "prompt": prompt_turn["content"],
                "response": generate_turn["content"],
                "status_ok": generate_turn["info"]["status_ok"],
            },
        )


async def rate_every_turn(
    raters: List[Evaluator],
    next_turn: ChatTurn,
    conv_list: List[ChatTurn],
    metadata: Dict[str, Any],
):
    """Run raters after each turn, and return whether the genration should stop."""
    for rater in raters:
        if rater.immediate_eval:
            # run the evaluators that require immediate eval
            if "annotation" not in next_turn["info"]:
                next_turn["info"]["annotation"] = {}

            if rater.support_async:
                next_turn["info"]["annotation"][rater.name] = await rater.async_eval(
                    next_turn, metadata
                )
            else:
                next_turn["info"]["annotation"][rater.name] = rater.eval(
                    next_turn, metadata
                )

    # AFTER all the immediate raters finished, ask them if they wanna stop
    rater_ask_stop = False
    for rater in raters:
        if rater.immediate_eval and rater.stop_conv(conv_list):
            rater_ask_stop = True

    return rater_ask_stop


async def rate_whole_conv(
    raters: List[Evaluator],
    conv_list: List[ChatTurn],
    metadata: Dict[str, Any],
    conv_id: str,
) -> List[ChatTurn]:
    """Rate the whole conv and add to the turn["info"]["annotation"] fields."""
    for rater in raters:
        # mark the turns to eval and create the tasks
        eval_turns: List[ChatTurn] = []
        eval_tasks: List[Dict[str, Any]] = []  # function kwargs
        for chat_turn in conv_list[1:]:
            if "annotation" not in chat_turn["info"]:
                chat_turn["info"]["annotation"] = {}

            if rater.name not in chat_turn["info"]["annotation"]:
                eval_turns.append(chat_turn)
                eval_tasks.append({"turn": chat_turn, "metadata": metadata})

        # actually rater the turns
        if rater.support_async:
            # rating simultaneously using asyncio
            rater_results = await asyncio.gather(
                *[
                    rater.async_eval(
                        **eval_task,
                        conv_id=conv_id,
                    )
                    for eval_task in eval_tasks
                ]
            )
        else:
            # rating sequentially
            rater_results = [
                rater.eval(**eval_task, conv_id=conv_id) for eval_task in eval_tasks
            ]

        # store the results back to the turns
        assert len(eval_turns) == len(rater_results)
        for eval_turn, rater_result in zip(eval_turns, rater_results):
            eval_turn["info"]["annotation"][rater.name] = rater_result

    return conv_list
