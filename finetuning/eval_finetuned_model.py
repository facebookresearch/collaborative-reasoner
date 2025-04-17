# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import List

from fire import Fire

from finetuning.iterative_training import convert_model_weights, eval_and_sample


def eval_model(
    ckpt_root: str,
    step: int,
    size: str,
    task: str,
    logs_dir_name: str = "eval_logs",
    skip_convert: bool = False,
    add_eval_args: List[str] | None = None,
):
    if not add_eval_args:
        add_eval_args = []

    # make the save dir
    ckpt_dir = f"{ckpt_root}/step_{step}"
    hf_ckpt_dir = f"{ckpt_root}/step_{step}_hf_converted"
    eval_save_dir = f"{ckpt_root}/{logs_dir_name}/step_{step}"
    os.makedirs(eval_save_dir, exist_ok=True)

    # first convert model weights
    if os.path.isdir(hf_ckpt_dir):
        print("Converted hf ckpt already exists, skip converting!")
    else:
        convert_model_weights(
            ckpt_dir=ckpt_dir, size=size, log_dir=eval_save_dir, link_only=skip_convert
        )

    # start the eval run
    eval_and_sample(
        hf_ckpt_dir,
        size,
        exp_dir=eval_save_dir,
        dataset=task,
        sample_size=-1,
        last_iter=True,
        add_eval_args=add_eval_args,
        correctness_metric="agreement_correctness",
        beam_size=1,
    )


if __name__ == "__main__":
    Fire(eval_model)
