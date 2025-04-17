# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""All the datasets must be registered here."""

from typing import Any, Dict, List, Type

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from lightning_modules.datasets.base_dataset import BaseDataset


def customized_collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    result_dict = {}

    for k in examples[0].keys():
        result_dict[k] = [ex[k] for ex in examples]

    return result_dict


class BaseDataModule(LightningDataModule):

    # this will be overriden in the subclass
    dataset_cls: Type[BaseDataset] = None  # type: ignore

    def __init__(
        self,
        train_batch_size: int | None = None,
        val_batch_size: int | None = None,
        # the following settings will override the default values in the init args of the dataset classes
        train_max_instances: int | None = None,
        val_max_instances: int | None = None,
        train_file_path: str | None = None,
        val_file_path: str | None = None,
        # the following dictionaries are used as default values, as the settings above will override them
        train_set_init_args: Dict[str, Any] | None = None,
        val_set_init_args: Dict[str, Any] | None = None,
        set_common_init_args: Dict[str, Any] | None = None,
    ):
        super().__init__()

        if train_set_init_args is None:
            train_set_init_args = {}
        if val_set_init_args is None:
            val_set_init_args = {}
        if set_common_init_args is None:
            set_common_init_args = {}

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        # delegate the initialization of the train and val datasets to the dataset classes
        self.train_set_init_args = train_set_init_args
        self.train_set_init_args.update(set_common_init_args)
        self.val_set_init_args = val_set_init_args
        self.val_set_init_args.update(set_common_init_args)

        if train_max_instances is not None:
            self.train_set_init_args["max_instances"] = train_max_instances
        if val_max_instances is not None:
            self.val_set_init_args["max_instances"] = val_max_instances
        if train_file_path is not None:
            self.train_set_init_args["file_path"] = train_file_path
        if val_file_path is not None:
            self.val_set_init_args["file_path"] = val_file_path

        self.train_data = None
        self.val_data = None

    def setup(self, stage: str | None = None):
        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        assert stage in ["validate"]

        if self.val_data is None:
            assert (
                self.dataset_cls is not None
            ), "dataset_cls must be overriden in the DataModule subclasses!"

            val_data = self.dataset_cls(mode="test", **self.val_set_init_args)
            self.val_data = val_data

    def train_dataloader(self):
        if self.train_data is None:
            self.setup(stage="fit")

        assert self.train_data is not None, "`self.train_data` is not created yet!"

        dtloader = DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=customized_collate_fn,
        )
        return dtloader

    def val_dataloader(self):
        if self.val_data is None:
            self.setup(stage="validate")

        assert self.val_data is not None, "`self.train_data` is not created yet!"

        dtloader = DataLoader(
            self.val_data,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=customized_collate_fn,
        )
        return dtloader

    def test_dataloader(self):
        raise NotImplementedError
