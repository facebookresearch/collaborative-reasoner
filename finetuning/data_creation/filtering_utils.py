# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
from collections import defaultdict
from typing import Any, Dict, List

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from utils.chat_utils import ChatTurn
from utils.chat_utils import apply_chat_template as ts_apply_chat_template
from utils.file_utils import load_jsonl


def get_accuracy_metric(metric_dict: dict, metric_name: str) -> float:
    acc_keys = list(filter(lambda x: x.endswith(metric_name), metric_dict.keys()))
    assert len(acc_keys) == 1

    return metric_dict[acc_keys[0]]


def get_acc_at_k(example: Dict[str, Any], metric_name: str) -> float:
    acc_list = [
        get_accuracy_metric(x["metrics"], metric_name) for x in example["conv_results"]
    ]

    return sum(acc_list) / len(acc_list)


def get_agreement(turn: ChatTurn) -> bool:
    extraction_raters = list(
        filter(
            lambda x: x.endswith("ExtractionPromptingEvaluator"),
            turn["info"]["annotation"].keys(),
        )
    )
    assert len(extraction_raters) == 1

    return turn["info"]["annotation"][extraction_raters[0]]["info"]["agreement"]


def get_turn_correctness(turn: ChatTurn) -> bool:
    correctness_raters = list(
        filter(
            lambda x: x.endswith("MatchEvaluator"), turn["info"]["annotation"].keys()
        )
    )
    assert len(correctness_raters) == 1

    return turn["info"]["annotation"][correctness_raters[0]]["score"]


def get_turn_length(turn: ChatTurn) -> int:
    try:
        return turn["info"]["usage"]["completion_tokens"]
    except KeyError:
        print(f"Can't get completion_tokens for turn {turn['content']}")
        return 1


def get_latest_result_timestamp(results_dir: str) -> str | None:
    prefix_set = set()

    for filename in os.listdir(results_dir):
        if filename.endswith(".jsonl"):
            try:
                prefix_set.add(int(filename.split("_")[0]))
            except:  # noqa: E722
                pass

    if len(prefix_set) > 0:
        prefix_list = list(prefix_set)
        lastest_prefix = sorted(prefix_list)[-1]
        # print(f"All timestamps: {prefix_list}, latest one: {lastest_prefix}")

        return lastest_prefix
    else:
        return None


def get_all_rank_results(
    results_dir: str, prefix: str = "predictions", sort_for_matrix: bool = False
):
    if prefix == "latest":
        timestamp = get_latest_result_timestamp(results_dir)
        assert timestamp is not None
        prefix = f"{timestamp}_outputs"

    all_files = []
    pattern = re.compile(rf"^{prefix}_step_0_rank_(\d+)\.jsonl$")
    for filename in os.listdir(results_dir):
        if pattern.match(filename):
            all_files.append(f"{results_dir}/{filename}")

    all_results = []
    # print(f"Loading {results_dir}/{prefix}_*...")
    iter = tqdm(all_files) if len(all_files) > 1 else all_files
    for result_file_name in iter:
        # print(f"Loading data from {result_file_name}")
        # rank_results = list(load_jsonl_with_progress(result_file_name))
        rank_results = list(load_jsonl(result_file_name))
        if sort_for_matrix:
            rank_results = sort_matrix_results(rank_results)

        all_results.extend(rank_results)

    return all_results


def sort_matrix_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    question_example_dict = defaultdict(list)

    for result in results:
        question_example_dict[result["metadata"]["init_prompt"]].append(result)

    if len(results) % len(question_example_dict) != 0:
        print(
            f"Warning: got {len(results)} results and {len(question_example_dict)} questions"
        )

    all_combined_examples = []
    for _, example_list in question_example_dict.items():
        example = {
            "conv_results": [
                {"conv": x["conv_result"], "metrics": x["metrics"]}
                for x in example_list
            ],
            "metadata": example_list[0]["metadata"],
        }
        all_combined_examples.append(example)

    return all_combined_examples


def get_single_turn_instance(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    conv: List[ChatTurn],
    turn_i: int,
    teacher_inst: str,
    student_inst: str,
    add_submission_prompt: bool,
) -> Dict[str, str]:
    """get the training instance from a certain good turn."""
    assert turn_i > 0, "There is nothing to learn from the first turn!"

    if conv[turn_i]["role"] == "teacher":
        sys_prompt = teacher_inst
    elif conv[turn_i]["role"] == "student":
        sys_prompt = student_inst
    else:
        raise ValueError(f"Unknown role {conv[turn_i]['role']}")

    msg_list = ts_apply_chat_template(conv[:turn_i], sys_prompt)
    training_src = tokenizer.apply_chat_template(
        msg_list, tokenize=False, add_generation_prompt=True, add_special_tokens=False
    )
    training_tgt = conv[turn_i]["content"]

    if add_submission_prompt and "submission" in conv[turn_i]["info"]:
        training_tgt += (
            conv[turn_i]["info"]["submission_prompt"]
            + conv[turn_i]["info"]["submission"]
        )

    instance = {"src": training_src, "tgt": training_tgt}

    return instance


def get_preference_pairs_from_turn(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    conv: List[ChatTurn],
    turn_i: int,
    teacher_inst: str,
    student_inst: str,
    max_pairs: int = 1,
    discourage_agreement: bool = False,
    discourage_length: bool = False,
) -> List[Dict[str, str]]:
    """get the training instance from a certain good turn."""
    assert turn_i > 0, "There is nothing to learn from the first turn!"

    if conv[turn_i]["role"] == "teacher":
        sys_prompt = teacher_inst
    elif conv[turn_i]["role"] == "student":
        sys_prompt = student_inst
    else:
        raise ValueError(f"Unknown role {conv[turn_i]['role']}")

    msg_list = ts_apply_chat_template(conv[:turn_i], sys_prompt)
    training_src = tokenizer.apply_chat_template(
        msg_list, tokenize=False, add_generation_prompt=True, add_special_tokens=False
    )

    assert "siblings" in conv[turn_i]["info"]
    tgt_turns = [conv[turn_i]] + conv[turn_i]["info"]["siblings"]
    tgt_turn_correctness = [get_turn_correctness(turn) for turn in tgt_turns]
    if discourage_agreement:
        tgt_turn_agreement = [get_agreement(turn) for turn in tgt_turns]
        tgt_turn_scores = [
            x - y * 0.1 * (1 - x)  # 0, 0 -> 0; 0, 1 -> -0.1; 1, 1 -> 1; 1, 0 -> 1
            for x, y in zip(tgt_turn_correctness, tgt_turn_agreement)
        ]
    elif discourage_length:
        tgt_turn_length = [get_turn_length(turn) for turn in tgt_turns]
        norm_tgt_turn_length = [
            (x - min(tgt_turn_length))
            / (1 + max(tgt_turn_length) - min(tgt_turn_length))  # avoid divide by zero
            for x in tgt_turn_length
        ]
        assert all([0 <= x <= 1 for x in norm_tgt_turn_length])

        tgt_turn_scores = [
            (x - 0.1 * y) for x, y in zip(tgt_turn_correctness, norm_tgt_turn_length)
        ]
    else:
        tgt_turn_scores = tgt_turn_correctness

    # get all possible pairs and their score diffs
    all_pairs = []
    for i in range(len(tgt_turns)):
        for j in range(i + 1, len(tgt_turns)):
            if (score_diff := tgt_turn_scores[i] - tgt_turn_scores[j]) >= 0:
                all_pairs.append((i, j, score_diff))
            else:
                all_pairs.append((j, i, -score_diff))

    # add the pairs based on the largest diffs
    sorted_pairs = sorted(all_pairs, key=lambda x: x[2], reverse=True)
    return_instances = []
    for i1, i2, score_diff in sorted_pairs:
        if score_diff > 0.5:
            return_instances.append(
                {
                    "src": training_src,
                    "tgt_chosen": tgt_turns[i1]["content"],
                    "tgt_rejected": tgt_turns[i2]["content"],
                    "pairwise": True,
                }
            )

    return return_instances[:max_pairs]
