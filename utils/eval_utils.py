# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

from utils.chat_utils import ChatTurn


def is_turn_err_out(chat_turn: ChatTurn) -> bool:
    return not chat_turn["info"]["status_ok"]


def is_conv_err_out(result_conv: List[ChatTurn]) -> bool:
    return is_turn_err_out(result_conv[-1])


def rater_err_out(result_conv: List[ChatTurn], rater_name: str) -> Tuple[int, int]:
    total_n = 0
    err_n = 0
    for turn in result_conv:
        if (
            "annotation" in turn["info"]
            and rater_name in turn["info"]["annotation"]
            and turn["info"]["annotation"][rater_name] is not None
        ):
            total_n += 1
            if not turn["info"]["annotation"][rater_name]["info"]["status_ok"]:
                err_n += 1

    return total_n, err_n


def n_turns_to_success(result_conv: List[ChatTurn], rater_name: str) -> int:
    """Return the number of turns it takes for the rater to get 1.0 score."""
    n_turns = 0  # because first turn always is student asking the question
    for turn in result_conv:
        if (
            "annotation" in turn["info"]
            and rater_name in turn["info"]["annotation"]
            and turn["info"]["annotation"][rater_name] is not None
            and turn["info"]["annotation"][rater_name]["score"] == 1.0
        ):
            return n_turns
        else:
            n_turns += 1

    return -1


def normalize_submission(submission: str) -> str:
    return submission.strip().rstrip(".")


def is_valid_submission(submission: str) -> bool:
    if len(submission.strip()) == 0:
        return False

    if "not sure yet" in submission.lower():
        return False

    return True


def agreement_eval(result_conv: List[ChatTurn], rater_name: str) -> Tuple[bool, bool]:
    """
    Returns the flag indicating whether there was agreement and whether the agreeed
        answer was formally correct.
    """

    if len(result_conv) <= 2:  # reminder: the first turn is just the question prompt
        return False, False

    last_turn = result_conv[-1]

    if (
        "submission" not in last_turn["info"]
        or "annotation" not in last_turn["info"]
        or not last_turn["info"]["annotation"].get(rater_name)
        or not is_valid_submission(last_turn["info"]["submission"])
    ):
        return False, False

    agreement = "agreement" in last_turn["info"] and last_turn["info"]["agreement"]

    if agreement:
        agreement_correctness = (
            last_turn["info"]["annotation"][rater_name]["score"] == 1.0
        )
    else:
        agreement_correctness = False

    return agreement, agreement_correctness


def get_avg_ts_len(result_conv: List[ChatTurn]) -> Tuple[float, float, int]:
    assert result_conv[0]["role"] in [
        "student",
        "teacher",
    ], "Unsupported role! Only student and teacher roles are supported right now!"
    t_lens = []
    s_lens = []
    for turn in result_conv[1:]:
        if turn["role"] == "student" and not is_turn_err_out(turn):
            s_lens.append(turn["info"]["usage"]["completion_tokens"])
        elif turn["role"] == "teacher" and not is_turn_err_out(turn):
            t_lens.append(turn["info"]["usage"]["completion_tokens"])

    avg_t_len = (sum(t_lens) / len(t_lens)) if len(t_lens) > 0 else 0
    avg_s_len = (sum(s_lens) / len(s_lens)) if len(s_lens) > 0 else 0

    total_conv_len = sum(t_lens) + sum(s_lens)

    return avg_t_len, avg_s_len, total_conv_len


def mutual_success(result_conv: List[ChatTurn], rater_name: str) -> float:
    for i, turn in enumerate(result_conv):
        if i < 3:
            continue

        if turn["role"] == "teacher":
            turn_rater_think_correct = turn["info"]["annotation"][rater_name]["score"]

            if turn_rater_think_correct:
                last_turn = result_conv[i - 1]
                match_evaluators = [
                    x
                    for x in last_turn["info"]["annotation"].keys()
                    if (x.endswith("MatchEvaluator") or x.endswith("ExecEvaluator"))
                ]
                assert len(match_evaluators) == 1, f"{match_evaluators}"
                return last_turn["info"]["annotation"][match_evaluators[0]]["score"]

    return 0.0


def turn_correct_by_match_exec(turn: ChatTurn) -> bool:
    if "annotation" not in turn["info"]:
        return False

    correctness_evaluators = [
        x
        for x in turn["info"]["annotation"].keys()
        if (x.endswith("MatchEvaluator") or x.endswith("ExecEvaluator"))
    ]

    assert (
        len(correctness_evaluators) == 1
    ), f"Only one correctness evaluator expected, got {correctness_evaluators}"

    correctness_evaluator_name = correctness_evaluators[0]

    return turn["info"]["annotation"][correctness_evaluator_name]["score"] == 1.0


def turn_correctness_score(turn: ChatTurn, not_sure_score: float = 0.5) -> float:
    if "extracted_answer" not in turn["info"]:
        return 0.0
    elif turn["info"]["extracted_answer"] == "not sure yet":
        return not_sure_score

    if "annotation" not in turn["info"]:
        return 0.0

    correctness_evaluators = [
        x
        for x in turn["info"]["annotation"].keys()
        if (x.endswith("MatchEvaluator") or x.endswith("ExecEvaluator"))
    ]

    assert (
        len(correctness_evaluators) == 1
    ), f"Only one correctness evaluator expected, got {correctness_evaluators}"

    correctness_evaluator_name = correctness_evaluators[0]

    return turn["info"]["annotation"][correctness_evaluator_name]["score"]


def social_behavior_metrics(
    conv: List[ChatTurn], not_sure_score: float
) -> Tuple[List[float], List[float], List[float]]:
    turn_scores = [not_sure_score] + [
        turn_correctness_score(x, not_sure_score) for x in conv[1:]
    ]
    turn_answers = ["not sure yet"] + [x["info"]["extracted_answer"] for x in conv[1:]]

    assert len(turn_answers) == len(conv)

    assertive_turn_scores = []
    persuasive_turn_scores = []
    persuasion_quality_scores = []
    for i, (turn_c_score, turn_c_ans) in enumerate(zip(turn_scores, turn_answers)):
        if i == 0:
            pass
        else:
            self_prev_ans = turn_answers[i - 2] if i > 1 else not_sure_score
            partner_prev_ans = turn_answers[i - 1]
            partner_future_ans = (
                turn_answers[i + 1] if (i < len(conv) - 1) else turn_answers[i - 1]
            )
            partner_future_score = (
                turn_scores[i + 1] if (i < len(conv) - 1) else turn_scores[i - 1]
            )

            if partner_prev_ans != self_prev_ans and turn_c_ans == self_prev_ans:
                # print("assertive!")
                assertive_turn_scores.append(turn_c_score)
            if (
                partner_prev_ans != partner_future_ans
                and turn_c_ans == partner_future_ans
            ):
                # print("persuasive!")
                persuasion_quality = (
                    partner_future_score - turn_scores[i - 1]
                )  # whether the persuasion happened for the better
                persuasion_quality_scores.append(persuasion_quality)
                persuasive_turn_scores.append(turn_c_score)

    return assertive_turn_scores, persuasive_turn_scores, persuasion_quality_scores


def behavior_quality(
    scores: List[float], not_sure_score: float
) -> Tuple[float, float, float]:
    correct_rate = scores.count(1.0) / len(scores) if len(scores) > 0 else 0.0
    incorrect_rate = scores.count(0.0) / len(scores) if len(scores) > 0 else 0.0
    unsure_rate = scores.count(not_sure_score) / len(scores) if len(scores) > 0 else 0.0

    if len(scores) > 0:
        assert abs(correct_rate + incorrect_rate + unsure_rate - 1.0) < 1e9
    return correct_rate, incorrect_rate, unsure_rate
