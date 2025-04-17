# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fire
import submitit

from evaluators.base_evaluator import Evaluator
from finetuning.data_creation.filtering_utils import get_latest_result_timestamp
from finetuning.data_creation.ts_correctness_filtering import submit_filtering_job
from finetuning.run_fairseq2_training import fairseq2_finetuning
from finetuning.slurm_utils import (
    SLURM_ACC,
    SLURM_QOS,
    run_ts_interaction,
    wait_job_completion,
)
from utils.file_utils import load_jsonl, save_json, save_jsonl

try:
    from matrix import Cli
    from matrix.cluster.ray_cluster import RayCluster, ClusterInfo

    from matrix.app_server import deploy_app
except ModuleNotFoundError:
    raise ValueError("Matrix is needed for iterative eval!")

logger = logging.getLogger()

EVAL_NODES = 0.5
SUPPORTED_DATASETS = ["math", "mmlu_pro", "explore_tom", "gqpa", "hi_tom"]
ITER_EXP_HOME = os.environ.get("ITER_EXP_HOME", "unset")

if ITER_EXP_HOME == "unset":
    raise ValueError("ITER_EXP_HOME is not set!")

MODEL_PRESET = {
    "8b": {
        "ori_hf_dir": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "tp": 1,
        "size": "8b",
        "max_ongoing_requests": 200,
    },
    "70b": {
        "ori_hf_dir": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "tp": 4,
        "size": "70b",
        "max_ongoing_requests": 100,
    },
    "fs2_8b": {
        "ori_hf_dir": None,
        "tp": 1,
        "size": "8b",
        "max_ongoing_requests": 200,
    },
    "fs2_70b": {
        "ori_hf_dir": None,
        "tp": 8,
        "size": "70b",
        "max_ongoing_requests": 200,
    },
}

TASK_PRESET = {
    "math": {"train_data": "data/math/train.jsonl"},
    "mmlu_pro": {"train_data": "data/mmlu_pro/resplit/train.jsonl"},
    "explore_tom": {"train_data": "data/explore_tom/train.jsonl"},
}

DEBUG = False
iter_i = 0


def ilog(msg: str, **kwargs):
    logger.info(f"[Iter {iter_i}]: {msg}", **kwargs)


def safe_get_head_info(cluster_cli: RayCluster) -> ClusterInfo:
    head_info = cluster_cli.cluster_info()
    if head_info is None:
        raise ValueError("Matrix Head service is down!")
    return head_info


def matrix_append_deploy(
    hf_dir: str,
    size: str,
    serving_replicas: int = -1,
    beam_size: int = 1,
    ray_cluster_id: str | None = None,
    ray_rdv_dir: str | None = None,
):
    """Deploy a HF model as a tmp application using matrix."""
    cluster_state = Cli(cluster_id=ray_cluster_id, matrix_dir=ray_rdv_dir).cluster
    head_info = safe_get_head_info(cluster_state)

    # check if this is a native fs2 ckpt
    if (Path(hf_dir) / "fs2_native").is_file() and not size.startswith("fs2_"):
        preset = MODEL_PRESET[f"fs2_{size}"]
    else:
        preset = MODEL_PRESET[size]

    if serving_replicas == -1:
        serving_replicas = int(EVAL_NODES * 8 // preset["tp"])

    # use beam size to estimate the max_ongoing_requests
    max_ongoing_requests = int(preset["max_ongoing_requests"] / math.sqrt(beam_size))

    # deploy hf_converted model
    app_name = deploy_app.append_deploy(
        cluster_state.cluster_dir,
        head_info,
        {
            "model_name": hf_dir,
            "model_size": size.upper(),
            "min_replica": serving_replicas,
            "max_replica": serving_replicas * 20,
            "use_grpc": "true",
            "max_ongoing_requests": max_ongoing_requests,
            "tensor-parallel-size": preset["tp"],
        },
    )

    return app_name, cluster_state


def ts_eval(
    hf_dir: str,
    app_name: str,
    save_dir: str,
    size: str,
    dataset: str,
    add_eval_args: List[str],
) -> Dict[str, Any] | None:
    """Evaluate the trained model, return the metrics dict."""

    run_ts_interaction(
        hf_dir=hf_dir,
        ts_eval_dir=save_dir,
        eval_logs_dir=save_dir,
        size=size,
        dataset=dataset,
        max_n_processes=200,
        app_name=app_name,
        add_args=add_eval_args
        + (
            ["--data.val_max_instances", "10"]
            if DEBUG
            else ["--model.max_n_processes", "1000"]
        ),
    )


def ts_sampling(
    hf_dir: str,
    app_name: str,
    save_dir: str,
    size: str,
    dataset: str,
    sample_size: int,
    add_eval_args: List[str],
    beam_size: int,
    data_file_path: str | None = None,
) -> None:
    """Sample from the served model using matrix, return sampling"""
    sampling_add_args = add_eval_args + [
        "--data.val_set_init_args.file_path",
        TASK_PRESET[dataset]["train_data"] if not data_file_path else data_file_path,
        "--model.sampling_n",
        str(sample_size),
        "--model.sampling_temp",
        "0.8",
        "--model.max_n_processes",
        str(int(1000 / beam_size)),  # NOTE: ad-hoc
        "--data.val_max_instances",
        "1_000_000",
        "--model.beam_size",
        str(beam_size),
    ]

    if DEBUG:
        sampling_add_args.extend(["--data.val_max_instances", "10"])
    else:
        sampling_add_args.extend(["--trainer.devices", "16"])

    run_ts_interaction(
        hf_dir=hf_dir,
        ts_eval_dir=save_dir,
        eval_logs_dir=save_dir,
        size=size,
        dataset=dataset,
        max_n_processes=200,
        app_name=app_name,
        add_args=sampling_add_args,
    )


def eval_and_sample(
    hf_dir: str,
    size: str,
    exp_dir: str,
    dataset: str,
    sample_size: int,
    last_iter: bool,
    add_eval_args: List[str],
    correctness_metric: str,
    beam_size: int,
    sample_data_path: str | None = None,
) -> Tuple[str, str]:
    """Given a model ckpt, sample from the model and eval with it at the same time."""
    # get some folder names
    eval_exp_dir = f"{exp_dir}/eval_exp"
    sampling_exp_dir = f"{exp_dir}/sampling_exp"

    # first check if everything in this step is done
    eval_done = os.path.isfile(f"{eval_exp_dir}/metrics.json")
    sampling_done = os.path.isfile(f"{sampling_exp_dir}/metrics.json") or last_iter

    if not (sampling_done and eval_done):
        # first serve this model and its companion model
        app_name, cluster_state = matrix_append_deploy(
            hf_dir=hf_dir, size=size, beam_size=beam_size
        )
        ilog(f"serving app_name {app_name}, wait 2 min for the service to come up...")
        time.sleep(2 * 60)

        if not eval_done:
            ts_eval(hf_dir, app_name, eval_exp_dir, size, dataset, add_eval_args)
        else:
            ilog("Skip eval as eval results already exists!")

        if not sampling_done:  # TODO: sampling could have been done async
            ts_sampling(
                hf_dir,
                app_name,
                sampling_exp_dir,
                size,
                dataset,
                sample_size,
                add_eval_args,
                beam_size=beam_size,
                data_file_path=sample_data_path,
            )
        else:
            ilog("Skip sampling as sampling results already exists or it's last iter!")

        deploy_app.remove_temp_app(
            cluster_state.cluster_dir, safe_get_head_info(cluster_state), app_name
        )
    else:
        ilog(
            "Skipped model serving, sampling and eval as "
            "sampling + eval results already exists!"
        )

    # get the sampling results
    if not last_iter:
        result_timestamp = get_latest_result_timestamp(sampling_exp_dir)
        if result_timestamp is not None:
            ilog(f"sampling results in {sampling_exp_dir}/{result_timestamp}_*.jsonl")
        else:
            raise ValueError("failed to get the sampling results!")
    else:
        ilog("skipped getting the sampling results as current iter is the last iter!")

    # get the eval results
    if os.path.exists(f"{eval_exp_dir}/metrics.json"):
        with open(f"{eval_exp_dir}/metrics.json", "r") as f:
            metrics_dict = json.load(f)

        perf = Evaluator.get_metric_from_dict(metrics_dict, correctness_metric)
        conv_err = Evaluator.get_metric_from_dict(metrics_dict, "conv_err")
        rater_err = Evaluator.get_metric_from_dict(
            metrics_dict, "ExtractionPromptingEvaluator-rater_err_rate"
        )
        ilog(
            f"loaded eval metrics, {correctness_metric} {perf*100:.2f} "
            f"conv_err {conv_err} rater_err {rater_err}!"
        )
    else:
        raise ValueError("the eval metrics json file doesn't exist!")

    return sampling_exp_dir, eval_exp_dir


def convert_model_weights(
    ckpt_dir: str,
    size: str,
    log_dir: str,
    link_only: bool = True,
):
    executor = submitit.AutoExecutor(folder=log_dir)
    convert_cmd = ["bash", "finetuning/convert_ckpt.sh", str(size), str(ckpt_dir)]
    executor.update_parameters(
        timeout_min=120,
        slurm_account=SLURM_ACC,
        slurm_qos=SLURM_QOS,
        slurm_ntasks_per_node=1,
        slurm_mem="1024G",
        nodes=1,
        cpus_per_task=32,
        name="weight_conversion",
    )
    if link_only:
        convert_cmd.append("true")
        s_result = subprocess.run(convert_cmd)
        if s_result.returncode == 0:
            print("Link-only conversion successful!")
        else:
            raise ValueError(
                f"Link-only conversion got exited with status {s_result.returncode}"
            )
    else:
        convert_function = submitit.helpers.CommandFunction(convert_cmd)
        convert_job = executor.submit(convert_function)
        ilog(f"Submitted model weights conversion job {convert_job.job_id}")
        wait_job_completion(convert_job)


def launch_training_and_conversion(
    iter_exp_dir: str,
    exp_name: str,
    training_steps: int,
    size: str,
    training_data_dir: str,
    dataset: str,
    training_method: str,
    continuous_training: bool,
    add_training_args: List[str],
    update_reference_model: bool,
) -> str:
    if training_method in ["dpo"]:
        assert (
            update_reference_model is not None
        ), "update_reference_model must be explicitly set for preference tuning!"

    training_save_dir = f"{iter_exp_dir}/training"
    training_exp_name = f"ts_iter-{exp_name}-iter_{iter_i}"

    ckpt_dir = (
        f"{training_save_dir}/{training_exp_name}/checkpoints/step_{training_steps}"
    )
    hf_ckpt_dir = f"{training_save_dir}/{training_exp_name}/checkpoints/step_{training_steps}_hf_converted"

    if os.path.isdir(hf_ckpt_dir):
        ilog(f"Skip training + conversion, as hf ckpt {hf_ckpt_dir} exists!")
    else:
        if not os.path.isdir(ckpt_dir):
            # For preference tuning
            # Continuous training + update ref model ~= online DPO
            # Continuous training + NO update ref model => mismatch between M(data) and M(ref)
            # NO continuous training + update ref model =>
            # No continuous training + NO update ref model ~= STaR w/ DPO
            reload_add_args = []
            if iter_i > 1 and (continuous_training or update_reference_model):
                last_iter_ckpt_dir = f"{iter_exp_dir}/../iter_{iter_i-1}/training/ts_iter-{exp_name}-iter_{iter_i-1}/checkpoints"
                reload_add_args.append(
                    f"common.assets.checkpoint_dir={last_iter_ckpt_dir}"
                )

                if continuous_training:
                    reload_add_args.append(
                        f"model.name=checkpoint_step_{training_steps}"
                    )

                if update_reference_model:
                    reload_add_args.append(
                        f"criterion.config.reference_model.name=checkpoint_step_{training_steps}"
                    )

            # start the training script
            fairseq2_finetuning(
                exp_name=training_exp_name,
                size=size,
                data=training_data_dir,
                task=dataset,
                save_dir=training_save_dir,
                training_method=training_method,
                no_eval=True,
                add_args=[
                    f"regime.num_steps={training_steps}",
                    f"regime.checkpoint_every_n_steps={training_steps}",
                    "common.metric_recorders.wandb.project=ts-finetuning-iter",
                ]
                + reload_add_args
                + add_training_args,
            )

            # wait for the training job to complete
            print("Checking for training save directory..")
            while not os.path.isdir(ckpt_dir):
                time.sleep(5 * 60)
        else:
            ilog(f"Skip training, as ckpt {ckpt_dir} already exists!")

        # convert the model weights
        convert_model_weights(ckpt_dir=ckpt_dir, size=size, log_dir=training_save_dir)
        ilog("Model weight conversion completed!")

    return hf_ckpt_dir


def create_iter_split_data(
    task: str, max_iter: int, exp_dir: str, strategy: str
) -> Dict[int, str]:
    iter_split_data_dir = f"{exp_dir}/iter_split_data"

    if os.path.isdir(iter_split_data_dir):
        print("Found existing iter data split!")
        return {
            i + 1: f"{iter_split_data_dir}/iter_{i+1}_data.jsonl"
            for i in range(max_iter)  # last iter would be dummy anyways
        }

    os.makedirs(iter_split_data_dir, exist_ok=True)

    iter_splits = []
    original_train_data = load_jsonl(TASK_PRESET[task]["train_data"])
    split_size = math.ceil(len(original_train_data) / (max_iter - 1))

    if strategy == "random_sample":
        for i in range(max_iter - 1):
            iter_splits.append(random.sample(original_train_data, k=split_size))
    elif strategy == "even_split":
        random.shuffle(original_train_data)
        for i in range(max_iter - 1):
            iter_splits.append(
                original_train_data[i * split_size : (i + 1) * split_size]
            )
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    assert len(iter_splits) == max_iter - 1
    print(
        f"Created iter split data for sizes: {','.join([str(len(x)) for x in iter_splits])}"
    )

    return_dict = {}
    for i, examples in enumerate(iter_splits):
        save_iter_file_path = f"{iter_split_data_dir}/iter_{i+1}_data.jsonl"
        save_jsonl(save_iter_file_path, examples)
        return_dict[i + 1] = save_iter_file_path
    # in the last iter there is only eval no sampling, use a dummy instead
    return_dict[max_iter] = "dummy_data.jsonl"

    return return_dict


def main(
    model: str,
    exp_name: str,
    dataset: str,
    max_iter: int = 5,
    sample_size: int = 20,
    filter_max_per_example: int = 10,
    filter_max_pairs_per_turn: int = 1,
    training_steps: int = 1000,
    training_method: str = "sft",
    continuous_training: bool = False,
    add_eval_args: List[str] | None = None,
    add_training_args: List[str] | None = None,
    add_filtering_args: Dict[str, Any] | None = None,
    correctness_metric: str = "agreement_correctness",
    stack_data: bool = False,
    beam_size: int = 1,
    update_reference_model: bool = False,
    data_split_strategy: str | None = None,
    iterative_lr_multiplier: float = 1.0,
    iterative_init_lr: float = 1e-6,
):
    # make sure the options are valid and process some of the options
    assert dataset in SUPPORTED_DATASETS
    assert training_method in ["sft", "dpo"]
    if training_method in ["dpo"]:
        assert beam_size > 1

    exp_dir = f"{ITER_EXP_HOME}/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)

    # save a copy of the settings
    settings_dict = locals().copy()
    save_json(f"{exp_dir}/settings.json", settings_dict)

    # make some preps
    training_steps = 100 if DEBUG else training_steps
    add_eval_args = add_eval_args or []
    add_training_args = add_training_args or []
    add_filtering_args = add_filtering_args or {}
    hf_dir = MODEL_PRESET[model]["ori_hf_dir"]
    size = MODEL_PRESET[model]["size"]

    # prepare the data for potential split
    if data_split_strategy:
        iter_data_dict = create_iter_split_data(
            dataset, max_iter, exp_dir, data_split_strategy
        )
    else:
        iter_data_dict = None

    # start the iteration
    global iter_i
    while (iter_i := iter_i + 1) <= max_iter:
        last_iter = iter_i == max_iter
        iter_exp_dir = f"{exp_dir}/iter_{iter_i}"

        #################################################
        #      Step 1: Sample and eval from a model     #
        #################################################
        sampling_exp_dir, _ = eval_and_sample(
            hf_dir=hf_dir,
            size=size,
            exp_dir=iter_exp_dir,
            dataset=dataset,
            sample_size=sample_size,
            last_iter=last_iter,
            add_eval_args=add_eval_args,
            correctness_metric=correctness_metric,
            beam_size=beam_size,
            sample_data_path=iter_data_dict[iter_i] if iter_data_dict else None,
        )
        if last_iter:
            ilog(f"Completed {max_iter} iterations! Terminate training!")
            break

        #################################################
        #      Step 2: Filtering and SFT data creation  #
        #################################################
        filtering_save_dir = f"{iter_exp_dir}/filtering"
        os.makedirs(filtering_save_dir, exist_ok=True)
        training_data_file = f"{filtering_save_dir}/data/train.jsonl"

        if os.path.isfile(training_data_file):
            ilog("Skip filtering, as training data already exists!")
        else:
            if stack_data:
                sampling_results_dirs = [
                    f"{exp_dir}/iter_{stack_i}/sampling_exp"
                    for stack_i in range(1, iter_i + 1)
                ]
            else:
                sampling_results_dirs = [sampling_exp_dir]

            filtering_job = submit_filtering_job(
                sampling_results_dirs=sampling_results_dirs,
                save_dir=filtering_save_dir,
                success_conv_only=False,
                max_instance_per_example=filter_max_per_example,
                correctness_metric=correctness_metric,
                generate_preference_pairs=(training_method in ["dpo"]),
                max_pairs_per_turn=filter_max_pairs_per_turn,
                **add_filtering_args,
            )
            wait_job_completion(filtering_job)

        training_data = load_jsonl(training_data_file)
        ilog(f"Training data size: {len(training_data)}")
        del training_data

        #################################################
        #      Step 3: Launch SFT with the data
        #################################################
        if iterative_lr_multiplier != 1.0:
            iter_lr = iterative_init_lr * (iterative_lr_multiplier ** (iter_i - 1))
            ilog(f"Changing to lr = {iter_lr} for iteration {iter_i}")
            add_training_args += [f"optimizer.config.lr={iter_lr}"]

        hf_ckpt_dir = launch_training_and_conversion(
            iter_exp_dir=iter_exp_dir,
            exp_name=exp_name,
            training_steps=training_steps,
            size=size,
            training_data_dir=f"{filtering_save_dir}/data",
            training_method=training_method,
            dataset=dataset,
            continuous_training=continuous_training,
            add_training_args=add_training_args,
            update_reference_model=update_reference_model,
        )

        # set the new model
        hf_dir = hf_ckpt_dir


if __name__ == "__main__":
    fire.Fire(main)
    # main("8b", "mmlu_pro_8b-iter_5-sample_25", 5, "mmlu_pro", 25)
