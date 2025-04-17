# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List

import fire
from jinja2 import Template

from finetuning.slurm_utils import SLURM_ACC, SLURM_QOS

FS2_ENV = os.environ.get("FS2_ENV", "unset")
if FS2_ENV == "unset":
    raise ValueError("FS2_ENV is not set!")

NODES_PRESETS = {
    "8b": {"sft": 1, "dpo": 1, "simpo": 1},
    "70b": {"sft": 4, "dpo": 16, "simpo": 4},
}

FT_MODES = {
    "dpo": "preference_finetune",
    "simpo": "preference_finetune",
    "sft": "instruction_finetune",
}

slurm_template = """\
#!/usr/bin/env bash

#SBATCH --job-name={{ name }}
#SBATCH --output={{ save_dir }}/%j.out
#SBATCH --error={{ save_dir }}/%j.err
#SBATCH --nodes={{ nodes }}
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=8
#SBATCH --account={{ s_account }}
#SBATCH --qos={{ s_qos }}

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "Neither anaconda3 nor miniconda3 found."
    exit 1
fi

source "$CONDA_PATH"
conda activate {{ fs2_env }}

srun fairseq2 lm {{ ft_mode }} "{{ save_dir }}" \
    --preset llama3_1_instruct \
    --no-sweep-dir \
    --config-file "{{ base_cfg }}" "{{ model_cfg }}" \
    --config dataset.path={{ data }} common.metric_recorders.wandb.run={{ name }} {{ remaining_args }}
"""


def fairseq2_finetuning(
    exp_name: str,
    size: str,
    data: str,
    task: str,
    save_dir: str | Path,
    training_method: str = "sft",
    no_eval: bool = False,
    add_args: List[str] | None = None,
) -> str | None:
    assert size in ["8b", "70b"]
    assert task in ["explore_tom", "math", "mmlu_pro"]
    assert training_method in ["dpo", "sft", "simpo"]

    if not add_args:
        add_args = []

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    # Prepare directories and configurations
    save_dir = Path(save_dir) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = Path(f"finetuning/configs/{training_method}.yaml").resolve()
    model_cfg = Path(f"finetuning/configs/{training_method}_{size}.yaml").resolve()

    if not base_cfg.is_file():
        raise ValueError(f"Error: Base configuration file {base_cfg} not found")
    else:
        print(f"Using base configuration {base_cfg}")

    if not model_cfg.is_file():
        raise ValueError(f"Error: Configuration file {model_cfg} not found")
    else:
        print(f"Using configuration {model_cfg}")
    print(f"Remaining args {add_args}")

    # Render the template
    slurm_script_vars = {
        "ft_mode": FT_MODES[training_method],
        "name": exp_name,
        "save_dir": str(save_dir),
        "nodes": NODES_PRESETS[size][training_method],
        "base_cfg": str(base_cfg),
        "model_cfg": str(model_cfg),
        "data": data,
        "remaining_args": " ".join(add_args),
        "s_account": SLURM_ACC,
        "s_qos": SLURM_QOS,
        "fs2_env": FS2_ENV,
    }
    rendered_script = Template(slurm_template).render(slurm_script_vars)
    # Create a temporary file for the SLURM script
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".sh") as tmp_file:
        tmp_file.write(rendered_script)
        tmp_file_path = tmp_file.name
    print(f"SLURM script generated at temporary location: {tmp_file_path}")

    # Submit the job
    submit_run = subprocess.run(["sbatch", tmp_file_path], capture_output=True)
    if (submit_output := submit_run.stdout.decode()).startswith("Submitted batch job"):
        job_id = submit_output.split()[3]
        print(f"Job accepted by SLURM: {job_id}")
    else:
        raise ValueError(
            "Failed to submit the job!"
            + f"STDOUT: \n{submit_output}"
            + f"STDERR: \n{submit_run.stderr.decode()}"
        )

    if no_eval:
        return job_id
    else:
        # to avoid circular importation for now
        from finetuning.eval_finetuned_model import eval_model

    # Monitor checkpoints and run evaluations
    print("Start monitoring output dir for ckpts and evaluations...")
    time.sleep(10 * 60)
    initial_ckpts = set(save_dir.glob("checkpoints/step_*00"))
    last_update_time = time.time()

    while True:
        current_ckpts = set(save_dir.glob("checkpoints/step_*00"))
        new_ckpts = current_ckpts - initial_ckpts

        if new_ckpts:
            for ckpt in new_ckpts:
                step = int(ckpt.name.split("_")[1])
                step_eval_logs_dir = save_dir / f"checkpoints/eval_logs/step_{step}"
                step_eval_logs_dir.mkdir(parents=True, exist_ok=True)

                print(f"Found new ckpt dir {ckpt}, running evals...")
                eval_model(
                    str(save_dir / "checkpoints"), step, size, task, skip_convert=True
                )

            initial_ckpts = current_ckpts
            last_update_time = time.time()
        else:
            current_time = time.time()
            if current_time - last_update_time > 36000:  # 10 hours
                print("No new checkpoints found in the last 10 hours. Stopping.")
                break

        time.sleep(300)  # Sleep for 5 minutes


if __name__ == "__main__":
    fire.Fire(fairseq2_finetuning)
