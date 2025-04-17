# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


class PatchedWandbLogger(WandbLogger):
    def __init__(
        self,
        entity: str,
        project: str,
        name: str,
        log_model: bool,
        save_code: bool,
        save_dir: str,
        tags: List[str] | None = None,
        offline: bool = False,
        *args,
        **kwargs,
    ):
        # try to get the exp name and save dir from env var to support multiple runs
        env_var_name = os.getenv("EXP_NAME")
        if env_var_name is not None:
            name = env_var_name
            save_dir = env_var_name

        # put necessary init vars for wandb
        kwargs["entity"] = entity
        kwargs["save_code"] = save_code
        kwargs["save_dir"] = save_dir
        kwargs["dir"] = save_dir  # fix a bug for wandb logger

        # remove the preceeding folder name
        processed_name = name.split("/")[-1]
        if tags is None:
            kwargs["tags"] = processed_name.split("-")
        else:
            kwargs["tags"] = tags

        # fail-safe for uploading the tmp exp for debugging
        if "tmp" in processed_name and not offline:
            print(
                f"WandbLogger: {processed_name} is a tmp exp so running in offline mode"
            )
            kwargs["offline"] = True

        # create the save_dir if it doesn't exist
        if not os.path.exists(save_dir):
            print(f"ready to create save_dir: {save_dir}", flush=True)
            os.makedirs(save_dir)
        else:
            print(f"reusing existing save_dir {save_dir}", flush=True)

        super().__init__(
            name=processed_name, project=project, log_model=log_model, *args, **kwargs
        )

    @rank_zero_only
    def log_code(self):

        # log the yaml and py files
        root = "."
        print(f"saving all files in {os.path.abspath(root)}")
        if not self._offline:
            result = self.experiment.log_code(
                root=root,
                include_fn=(
                    lambda path: path.endswith(".py") or path.endswith(".yaml")
                ),
                exclude_fn=(
                    lambda path: ".venv" in path
                    or "results/" in path
                    or "libs/" in path
                    or "wandb/" in path
                ),
            )

            if result is not None:
                print("########################################")
                print("######## Logged code to wandb. #########")
                print("########################################")
            else:
                print("######## logger inited but not successfully saved #########")
