# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from typing import List

import submitit
from submitit.core.core import Job

SLURM_ACC = os.environ.get("SLURM_ACC", "unset")
SLURM_QOS = os.environ.get("SLURM_QOS", "unset")

if SLURM_ACC == "unset" or SLURM_QOS == "unset":
    raise ValueError("SLURM_ACC and SLURM_QOS must be set in the environment.")


def wait_job_completion(
    job: Job, interval_s: int = 60, exit_on_err: bool = True
) -> bool:
    printed = False
    while True:
        job_status = job.state
        if job_status == "COMPLETED":
            print(f"\nJob {job.job_id} completed")
            return True
        elif job_status == "FAILED" or job_status == "TIMEOUT":
            if exit_on_err:
                raise ValueError(f"Job {job.job_id} failed")
            else:
                return False
        else:
            if not printed:
                print(f"Job {job.job_id} is still running ({job_status})", end="")
                printed = True
            else:
                print(".", end="", flush=True)
        time.sleep(interval_s)


def run_ts_interaction(
    hf_dir: str,
    ts_eval_dir: str,
    eval_logs_dir: str,
    size: str,
    dataset: str,
    max_n_processes: int,
    app_name: str | None = None,
    add_args: List[str] | None = None,
):
    # run the interaction exps
    os.environ["EXP_NAME"] = ts_eval_dir
    os.environ["WANDB_MODE"] = "offline"
    executor = submitit.AutoExecutor(folder=eval_logs_dir)

    ts_exp_cmd = [
        "python",
        "lightning_modules/trainer.py",
        "validate",
        "--trainer",
        "lightning_modules/configs/common/api_inference_trainer.yaml",
        "--model",
        "lightning_modules/configs/common/ts_interaction_model.yaml",
        "--config",
        f"lightning_modules/configs/ts_{dataset}.yaml",
        "--model.max_n_processes",
        str(max_n_processes),
        "--model.request_timeout",
        "1800",
        "--model.teacher_model",
        hf_dir,
        "--model.student_model",
        hf_dir,
        "--model.extractor.model_name",
        f"meta-llama/Meta-Llama-3.1-{size.upper()}-Instruct",
        "--model.max_turn_tokens",
        "1024",
        "--model.sampling_temp",
        "0.0",
    ]
    if app_name is not None:
        # this means that matrix is used
        ts_exp_cmd.extend(
            [
                "--model.teacher_app_name",
                app_name,
                "--model.student_app_name",
                app_name,
                "--model.extractor.app_name",
                f"{size.upper()}_grpc",
                "--model.use_matrix",
                "True",
                "--model.extractor.use_matrix",
                "True",
            ]
        )

    if add_args is not None:
        ts_exp_cmd.extend(add_args)

    ts_function = submitit.helpers.CommandFunction(
        ts_exp_cmd,
        env=dict(os.environ),
    )

    executor.update_parameters(
        timeout_min=24 * 60,
        slurm_account=SLURM_ACC,
        slurm_qos=SLURM_QOS,
        slurm_ntasks_per_node=1,
        nodes=1,
        cpus_per_task=32,
        name=f"ts_eval_{size}",
    )
    ts_job = executor.submit(ts_function)
    print(f"Submitted model inference job {ts_job.job_id}")
    wait_job_completion(ts_job)

    del os.environ["EXP_NAME"]
    del os.environ["WANDB_MODE"]
