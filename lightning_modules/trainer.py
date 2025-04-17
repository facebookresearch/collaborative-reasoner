# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.cli import LightningCLI

cli = LightningCLI(
    LightningModule,
    LightningDataModule,
    subclass_mode_model=True,
    subclass_mode_data=True,
    save_config_callback=None,
)
