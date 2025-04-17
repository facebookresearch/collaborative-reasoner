# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from typing import List, Tuple

import fire
import submitit
from submitit.core.core import Job
from transformers import AutoTokenizer

from finetuning.data_creation.filtering_utils import (
    get_acc_at_k,
    get_accuracy_metric,
    get_all_rank_results,
    get_latest_result_timestamp,
    get_preference_pairs_from_turn,
    get_single_turn_instance,
    get_turn_correctness,
)
from finetuning.slurm_utils import SLURM_ACC, SLURM_QOS
from utils.file_utils import save_json, save_jsonl


def correctness_filtering(
    sampling_results_dirs: List[str],
    save_data_dir: str,
    tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    hard_threshold: float = 1.0,
    max_instance_per_example: int = 1_000_000,
    first_n_turns_only: int = 1_000_000,
    add_prev_n_turns: int = 1_000_000,
    success_conv_only: bool = False,
    filter_by_turns: bool = False,
    correctness_metric: str = "agreement_correctness",
    generate_preference_pairs: bool = False,
    max_pairs_per_turn: int = 1,
    discourage_agreement: bool = False,
    discourage_length: bool = False,
):
    # save the config used to generate this file
    os.makedirs(save_data_dir, exist_ok=True)
    save_json(f"{save_data_dir}/settings.json", locals())

    # make sure the options are not in conflict
    assert not (
        generate_preference_pairs and filter_by_turns
    ), "Can not filter by turns and also generate preference pairs"

    # load all the results
    print("Loading data from: \n" + "\n".join(sampling_results_dirs))
    sampling_timestamp_dirs = []
    for sampling_results_dir in sampling_results_dirs:
        timestamp = get_latest_result_timestamp(sampling_results_dir)
        assert timestamp is not None, f"Got none timestamp from {sampling_results_dir}"
        sampling_timestamp_dirs.append((int(timestamp), sampling_results_dir))

    # combine the results from different timestamped dirs, newest first
    sampling_timestamp_dirs = sorted(
        sampling_timestamp_dirs, key=lambda x: x[0], reverse=True
    )
    sampling_results_dict = {}
    for timestamp, sampling_results_dir in sampling_timestamp_dirs:
        sampling_results = get_all_rank_results(
            sampling_results_dir, prefix=f"{timestamp}_outputs", sort_for_matrix=True
        )
        for r in sampling_results:
            k = r["metadata"]["init_prompt"]
            if k in sampling_results_dict:
                sampling_results_dict[k]["conv_results"].extend(r["conv_results"])
            else:
                sampling_results_dict[k] = r
    sampling_results = list(sampling_results_dict.values())

    # get some stuff needed
    teacher_inst = sampling_results[0]["metadata"]["teacher_inst"]
    student_inst = sampling_results[0]["metadata"]["student_inst"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # filter by examples
    results_filtered_by_example = [
        x
        for x in sampling_results
        if get_acc_at_k(x, correctness_metric) <= hard_threshold
    ]
    print(f"Before example-level filtering: {len(sampling_results)} example.")
    print(f"After example-level filtering: {len(results_filtered_by_example)} example.")

    # filter by conversation
    results_filtered_by_conv = []
    if success_conv_only:
        for result in results_filtered_by_example:
            filtered_convs = []
            for conv in result["conv_results"]:
                if get_accuracy_metric(conv["metrics"], correctness_metric) == 1.0:
                    filtered_convs.append(conv)
            sorted_filtered_convs = sorted(filtered_convs, key=lambda x: len(x["conv"]))

            results_filtered_by_conv.append(
                {"conv_results": sorted_filtered_convs, "metadata": result["metadata"]}
            )
    else:
        results_filtered_by_conv = results_filtered_by_example

    avg_conv_f = lambda x: sum(len(y["conv_results"]) for y in x) / len(x)  # noqa: E731
    print(
        f"Before conv-level filtering: {avg_conv_f(results_filtered_by_example)} convs per example."
    )
    print(
        f"After conv-level filtering: {avg_conv_f(results_filtered_by_conv)} convs per example."
    )

    # filter by turns
    training_instances = []
    if generate_preference_pairs:
        for result in results_filtered_by_conv:
            example_instances = []
            for conv in result["conv_results"]:
                for turn_i, turn in enumerate(conv["conv"]):
                    if turn_i == 0 or turn_i >= first_n_turns_only:
                        continue

                    example_instances.extend(
                        get_preference_pairs_from_turn(
                            tokenizer=tokenizer,
                            conv=conv["conv"],
                            turn_i=turn_i,
                            teacher_inst=teacher_inst,
                            student_inst=student_inst,
                            max_pairs=max_pairs_per_turn,
                            discourage_agreement=discourage_agreement,
                            discourage_length=discourage_length,
                        )
                    )

            random.shuffle(example_instances)
            training_instances.extend(example_instances[:max_instance_per_example])
    elif not filter_by_turns:
        for result in results_filtered_by_conv:
            for conv in result["conv_results"][:max_instance_per_example]:
                for turn_i, turn in enumerate(conv["conv"]):
                    if turn_i == 0 or turn_i >= first_n_turns_only:
                        continue
                    training_instances.append(
                        get_single_turn_instance(
                            tokenizer=tokenizer,
                            conv=conv["conv"],
                            turn_i=turn_i,
                            teacher_inst=teacher_inst,
                            student_inst=student_inst,
                            add_submission_prompt=False,
                        )
                    )
    else:
        for result in results_filtered_by_conv:
            turns_to_add: List[Tuple[int, int]] = []
            for conv_i, conv in enumerate(result["conv_results"]):
                for turn_i, turn in enumerate(conv["conv"]):
                    if turn_i == 0:
                        continue
                    if turn_i < first_n_turns_only and get_turn_correctness(turn):
                        turns_to_add.append((conv_i, turn_i))

            turns_to_add = sorted(turns_to_add, key=lambda x: x[1])
            for conv_i, turn_i in turns_to_add[:max_instance_per_example]:
                for add_turn_i in range(max(1, turn_i - add_prev_n_turns), turn_i + 1):
                    training_instances.append(
                        get_single_turn_instance(
                            tokenizer=tokenizer,
                            conv=result["conv_results"][conv_i]["conv"],
                            turn_i=add_turn_i,
                            teacher_inst=teacher_inst,
                            student_inst=student_inst,
                            add_submission_prompt=False,
                        )
                    )

    # save the results
    random.shuffle(training_instances)
    save_data_dir = f"{save_data_dir}/data"
    if not os.path.exists(save_data_dir):
        os.mkdir(save_data_dir)
    save_jsonl(os.path.join(save_data_dir, "train.jsonl"), training_instances)
    print(
        f"{len(training_instances)} training instances saved to {save_data_dir}/train.jsonl"
    )


def submit_filtering_job(
    sampling_results_dirs: List[str],
    save_dir: str,
    **kwargs,
) -> Job:
    os.makedirs(save_dir, exist_ok=True)

    executor = submitit.AutoExecutor(folder=save_dir)
    executor.update_parameters(
        timeout_min=120,
        slurm_account=SLURM_ACC,
        slurm_qos=SLURM_QOS,
        slurm_ntasks_per_node=1,
        slurm_mem="256G",
        nodes=1,
        cpus_per_task=32,
        name="filtering",
    )

    convert_job = executor.submit(
        correctness_filtering,
        sampling_results_dirs=sampling_results_dirs,
        save_data_dir=save_dir,
        **kwargs,
    )
    print(f"Submitted filtering job {convert_job.job_id}")

    return convert_job


if __name__ == "__main__":
    fire.Fire(correctness_filtering)
