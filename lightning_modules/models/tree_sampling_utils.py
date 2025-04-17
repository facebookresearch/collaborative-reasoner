# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import random
import uuid
from collections import Counter, defaultdict
from itertools import chain
from typing import Any, Dict, List, Tuple

from evaluators.base_evaluator import Evaluator, rate_every_turn, rate_whole_conv
from finetuning.data_creation.filtering_utils import get_turn_correctness
from utils.chat_utils import (
    ChatRoleFactory,
    ChatTurn,
    generate_next_turn,
    generate_next_turn_candidates,
)
from utils.eval_utils import get_avg_ts_len, is_conv_err_out
from utils.http_utils import EMPTY_USAGE_DICT


def get_belief(turn: ChatTurn) -> str:
    if "extracted_answer" in turn["info"]:
        return turn["info"]["extracted_answer"]
    else:
        return "None"


def pick_by_diversity(
    finished_convs: List[List[ChatTurn]],
    conv_prefixes: List[List[ChatTurn]],
    next_turns: List[List[ChatTurn]],
    beam_size: int,
) -> List[List[ChatTurn]]:
    assert len(conv_prefixes) == len(next_turns)

    def get_conv_belief_set(conv: List[ChatTurn]) -> Tuple[str, ...]:
        conv_belief_seq = [get_belief(turn) for turn in conv]
        return tuple(
            sorted(
                list(
                    set(
                        x
                        for x in conv_belief_seq
                        if x.lower() not in ["none", "not sure yet"]
                        and not x.startswith("Error")
                    )
                )
            )
        )

    # those should also be considered when diversifying
    finished_believes = Counter(get_conv_belief_set(x) for x in finished_convs)

    belief_set_dict = defaultdict(list)
    for conv_prefix, next_turn_cands in zip(conv_prefixes, next_turns):
        for next_turn_cand in next_turn_cands:
            if not next_turn_cand["info"]["status_ok"]:
                continue

            full_conv = conv_prefix + [next_turn_cand]
            conv_believes = get_conv_belief_set(full_conv)
            belief_set_dict[conv_believes].append(full_conv)

    beam_convs = [] + finished_convs
    while len(beam_convs) < beam_size:
        conv_added = False
        for belief_set_key, full_conv_list in belief_set_dict.items():
            if (
                belief_set_key in finished_believes
                and finished_believes[belief_set_key] > 1
            ):
                finished_believes[belief_set_key] -= 1
                conv_added = True
                continue

            if len(full_conv_list) > 0:
                beam_convs.append(full_conv_list.pop())
                conv_added = True
            if len(beam_convs) >= beam_size:
                break

        if not conv_added:
            break

    assert len(beam_convs) == beam_size
    return beam_convs


def rand_pick_one(
    finished_convs: List[List[ChatTurn]],
    conv_prefixes: List[List[ChatTurn]],
    next_turns: List[List[ChatTurn]],
    beam_size: int,
    stratified_sampling: bool = False,
) -> List[List[ChatTurn]]:
    assert len(finished_convs) == 0
    assert len(conv_prefixes) == len(next_turns) == 1
    assert beam_size == len(next_turns[0])

    next_turns_correctness = [get_turn_correctness(turn) for turn in next_turns[0]]
    pos_indices = [
        i for i, correctness in enumerate(next_turns_correctness) if correctness
    ]
    neg_indices = [
        i for i, correctness in enumerate(next_turns_correctness) if not correctness
    ]

    if stratified_sampling and len(pos_indices) > 0 and len(neg_indices) > 0:
        if random.random() > 0.5:
            pick_idx = random.choice(pos_indices)
        else:
            pick_idx = random.choice(neg_indices)
    else:
        pick_idx = random.randint(0, beam_size - 1)

    next_turns[0][pick_idx]["info"]["siblings"] = (
        next_turns[0][:pick_idx] + next_turns[0][pick_idx + 1 :]
    )

    return [conv_prefixes[0] + [next_turns[0][pick_idx]]]


def shrink_beam(
    finished_convs: List[List[ChatTurn]],
    conv_prefixes: List[List[ChatTurn]],
    next_turns: List[List[ChatTurn]],
    beam_size: int,
    strategy: str = "random",
) -> List[List[ChatTurn]]:
    """Given the conv prefixes and the candidates of next turns, reduce it back to the
    original beam size."""

    if strategy == "diversity":
        return pick_by_diversity(finished_convs, conv_prefixes, next_turns, beam_size)
    elif strategy == "random":
        return rand_pick_one(finished_convs, conv_prefixes, next_turns, beam_size)
    else:
        raise ValueError(f"Unknown strategy {strategy}")


async def ts_interaction_with_expansion(
    # the roles and rater
    teacher_factory: ChatRoleFactory,
    student_factory: ChatRoleFactory,
    raters: List[Evaluator],
    # the topic of the interaction
    init_prompt: str,
    init_role: str,
    # generation args
    max_turns: int,
    metadata: Dict[str, Any],
    request_timeout: int,
    max_turn_tokens: int | None,
    # potentially resuming from previous run with too many network timeouts
    use_matrix: bool = False,
    expand_size: int = 1,
) -> List[Dict[str, Any]]:

    convs: List[List[ChatTurn]] = [
        [ChatTurn(role=init_role, content=init_prompt, info=EMPTY_USAGE_DICT)]
    ]

    # this uuid is used for affinity purposes
    conv_id = str(uuid.uuid1())
    teacher = await teacher_factory.get_chat_role(uuid=conv_id)
    student = await student_factory.get_chat_role(uuid=conv_id)

    finished_convs = []
    for turn_i in range(max_turns):
        endpoint_cache_list = []
        roles = []
        for conv in convs:
            # for each conv in the beam
            if conv[-1]["role"] == "student":
                role = teacher
                endpoint_cache = teacher_factory.endpoint_cache
            elif conv[-1]["role"] == "teacher":
                role = student
                endpoint_cache = student_factory.endpoint_cache
            else:
                raise ValueError(f"Unexpected role {conv[-1]['role']}")

            roles.append(role)
            endpoint_cache_list.append(endpoint_cache)

        # batch send the requests for completion
        next_turn_candidates_list = await asyncio.gather(
            *[
                generate_next_turn_candidates(
                    chat_turns=conv,
                    chat_role=role,
                    request_timeout=request_timeout,
                    max_tokens=max_turn_tokens,
                    use_matrix=use_matrix,
                    endpoint_cache=endpoint_cache,
                    n_generations=expand_size,
                )
                for role, conv, endpoint_cache in zip(roles, convs, endpoint_cache_list)
            ]
        )

        # batch eval (e.g., extract) the turns
        assert len(convs) == len(next_turn_candidates_list)
        assert all(len(x) == expand_size for x in next_turn_candidates_list)
        flatten_next_turn_list = list(chain(*next_turn_candidates_list))
        stop_conv_flags = await asyncio.gather(
            *[
                rate_every_turn(
                    raters,
                    flatten_next_turn_list[i],
                    convs[i // expand_size] + [flatten_next_turn_list[i]],
                    metadata,
                )
                for i in range(len(flatten_next_turn_list))
            ]
        )

        for turn, stop_flag in zip(flatten_next_turn_list, stop_conv_flags):
            turn["info"]["stop_conv"] = stop_flag

        # shrink back to the beam size
        new_convs = shrink_beam(
            finished_convs,
            convs,
            next_turn_candidates_list,
            expand_size,
        )
        if all(conv[-1]["info"]["stop_conv"] for conv in new_convs):
            convs = new_convs
            break
        elif turn_i == max_turns - 1:
            convs = new_convs
        else:
            convs = list(filter(lambda x: not x[-1]["info"]["stop_conv"], new_convs))
            finished_convs = list(
                filter(lambda x: x[-1]["info"]["stop_conv"], new_convs)
            )

    # the chat completed, now do all the ratings
    conv_result_dicts = []
    for conv in convs:
        conv = await rate_whole_conv(raters, conv, metadata, conv_id)

        # collect conversation metrics
        conv_err = int(is_conv_err_out(conv))
        t_len, s_len, total_len = get_avg_ts_len(conv)

        metric_dict = {
            "conv_err": conv_err,
            "total_turns": len(conv),
            "t_len": t_len,
            "s_len": s_len,
            "conv_len": total_len,
        }

        # get rater specific metrics
        for rater in raters:
            rater_metrics = rater.get_conv_metrics(conv, {"max_turns": max_turns})
            metric_dict.update(rater_metrics)

        conv_result_dict = {
            "conv_result": conv,
            "metadata": metadata,
            "metrics": metric_dict,
        }
        conv_result_dicts.append(conv_result_dict)

    assert len(conv_result_dicts) == expand_size or len(conv_result_dicts) == 1
    return conv_result_dicts


async def ts_interaction_single(
    # the roles and rater
    teacher_factory: ChatRoleFactory,
    student_factory: ChatRoleFactory,
    raters: List[Evaluator],
    # the topic of the interaction
    init_prompt: str,
    init_role: str,
    # generation args
    max_turns: int,
    metadata: Dict[str, Any],
    request_timeout: int,
    max_turn_tokens: int | None,
    # potentially resuming from previous run with too many network timeouts
    use_matrix: bool = False,
) -> List[Dict[str, Any]]:
    """Simulate a conversation between two agents, each with a different system prompt."""

    conv_list: List[ChatTurn] = [
        ChatTurn(role=init_role, content=init_prompt, info=EMPTY_USAGE_DICT)
    ]

    # this uuid is used for affinity purposes
    conv_id = str(uuid.uuid1())
    teacher = await teacher_factory.get_chat_role(uuid=conv_id)
    student = await student_factory.get_chat_role(uuid=conv_id)

    for _ in range(max_turns):
        if conv_list[-1]["role"] == "student":
            role = teacher
            endpoint_cache = teacher_factory.endpoint_cache
        elif conv_list[-1]["role"] == "teacher":
            role = student
            endpoint_cache = student_factory.endpoint_cache
        else:
            raise ValueError(f"Unexpected role {conv_list[-1]['role']}")

        next_turn = await generate_next_turn(
            chat_turns=conv_list,
            chat_role=role,
            request_timeout=request_timeout,
            max_tokens=max_turn_tokens,
            use_matrix=use_matrix,
            endpoint_cache=endpoint_cache,
        )

        conv_list.append(next_turn)

        if not next_turn["info"]["status_ok"]:
            # timeout or other error happened despite retry, terminate
            break

        rater_ask_stop = await rate_every_turn(raters, next_turn, conv_list, metadata)
        if rater_ask_stop:
            break

    # the chat completed, now do the ratings
    conv_list = await rate_whole_conv(raters, conv_list, metadata, conv_id)

    # collect conversation metrics
    conv_err = int(is_conv_err_out(conv_list))
    t_len, s_len, total_len = get_avg_ts_len(conv_list)

    metric_dict = {
        "conv_err": conv_err,
        "total_turns": len(conv_list),
        "t_len": t_len,
        "s_len": s_len,
        "conv_len": total_len,
    }

    # get rater specific metrics
    for rater in raters:
        rater_metrics = rater.get_conv_metrics(conv_list, {"max_turns": max_turns})
        metric_dict.update(rater_metrics)

    return [{"conv_result": conv_list, "metadata": metadata, "metrics": metric_dict}]
