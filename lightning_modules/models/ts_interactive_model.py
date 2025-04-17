# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from overrides import overrides
from pytorch_lightning import LightningModule
from torch.nn.modules.container import ModuleDict
from torchmetrics import MeanMetric, MetricCollection
from tqdm import tqdm

from evaluators.base_evaluator import Evaluator
from evaluators.extraction_evaluators import ExtractionPromptingEvaluator
from lightning_modules.loggers.patched_loggers import PatchedWandbLogger
from lightning_modules.models.tree_sampling_utils import (
    ts_interaction_single,
    ts_interaction_with_expansion,
)
from utils.chat_utils import ChatRoleFactory
from utils.http_utils import DEFAULT_REQUEST_TIMEOUT


async def ts_interaction(expand_size: int = 1, **kwargs) -> List[Dict[str, Any]]:
    init_role = kwargs["init_role"]
    assert init_role in ["student", "teacher"], f"Unknown role {init_role}"

    if expand_size == 1:
        return await ts_interaction_single(**kwargs)
    else:
        return await ts_interaction_with_expansion(**kwargs, expand_size=expand_size)


lock = threading.Lock()


def save_to_jsonl(
    data: List[Dict[str, Any]],
    filename: str,
    write_mode: str,
) -> None:
    with lock:
        with open(filename, write_mode) as file:
            for item in data:
                file.write(json.dumps(item) + "\n")


class TSInteractiveModel(LightningModule):
    def __init__(
        self,
        max_turns: int,
        teacher_model: str,
        student_model: str,
        extractor: ExtractionPromptingEvaluator,
        raters: List[Evaluator],
        max_turn_tokens: int | None = None,
        sampling_n: int = 1,
        sampling_temp: float = 0.2,
        beam_size: int = 1,
        max_n_processes: int = 10,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
        debug: bool = False,
        use_matrix: bool = False,
        teacher_app_name: Optional[str] = None,
        student_app_name: Optional[str] = None,
        ray_cluster_id: Optional[str] = None,
        ray_rdv_dir: Optional[str] = None,
    ):
        super().__init__()

        # factories to produce the chat roles
        self.teacher_factory = ChatRoleFactory(
            name="teacher",
            model_name=teacher_model,
            sampling_temp=sampling_temp,
            app_name=teacher_app_name,
            ray_cluster_id=ray_cluster_id,
            ray_rdv_dir=ray_rdv_dir,
        )
        self.student_factory = ChatRoleFactory(
            name="student",
            model_name=student_model,
            sampling_temp=sampling_temp,
            app_name=student_app_name,
            ray_cluster_id=ray_cluster_id,
            ray_rdv_dir=ray_rdv_dir,
        )

        # init the evaluators
        self.raters = [extractor] + raters

        # some generation args
        self.max_turns = max_turns
        self.sampling_n = sampling_n
        self.sampling_temp = sampling_temp
        self.beam_size = beam_size
        self.max_turn_tokens = max_turn_tokens

        # http requests args
        self.max_n_processes = max_n_processes
        self.request_timeout = request_timeout

        self.debug = debug
        self.use_matrix = use_matrix

        # init the metrics
        self.metrics_dict: ModuleDict = MetricCollection({})
        self.metrics_dict["conv_err"] = MeanMetric()
        self.metrics_dict["total_turns"] = MeanMetric()
        self.metrics_dict["t_len"] = MeanMetric()
        self.metrics_dict["s_len"] = MeanMetric()
        self.metrics_dict["conv_len"] = MeanMetric()

        # add the metric from the raters
        for rater in self.raters:
            for metric_name in rater.evaluator_metrics:
                self.metrics_dict[metric_name] = MeanMetric()

        self.prev_predictions = []

    @overrides
    def setup(self, stage: str) -> None:
        if self.logger and isinstance(self.logger, PatchedWandbLogger):
            # if logger is initialized, save the code
            self.logger.log_code()
        else:
            print("logger is not initialized or not Wandb, code will not be saved")

    async def batch_ts_interaction(
        self,
        init_prompts: List[str],
        init_roles: List[str],
        metadata: List[Dict[str, Any]],
    ) -> None:
        # the progress bar progresses every time an interaction concludes
        pbar = tqdm(total=len(init_prompts), desc="Processing async tasks")

        # all the batched tasks would share this semaphore
        batch_tasks = set()
        batch_results = []
        first_time = True
        timestamp = int(time.time()) // 30
        log_dir = self.get_output_dir()
        global_rank = self.trainer.global_rank
        global_step = self.trainer.global_step

        async def save_outputs(flush=False):
            nonlocal batch_tasks, batch_results, first_time
            output_batch_size = 32

            if batch_tasks:
                completed, batch_tasks = await asyncio.wait(
                    batch_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for completed_task in completed:
                    batch_results.extend(await completed_task)
                    pbar.update(1)
            if flush or len(batch_results) >= output_batch_size:
                save_output_file_path = os.path.join(
                    log_dir,
                    f"{str(timestamp)}_outputs_step_{global_step}"
                    + f"_rank_{global_rank}.jsonl",
                )
                await asyncio.to_thread(
                    save_to_jsonl,
                    batch_results,
                    save_output_file_path,
                    "w" if first_time else "a",
                )
                # update global metrics
                for result in batch_results:
                    for k, v in result["metrics"].items():
                        self.metrics_dict[k](v)
                batch_results = []
                first_time = False

        for example_prompt, example_role, example_metadata in zip(
            init_prompts, init_roles, metadata
        ):
            batch_tasks.add(
                asyncio.create_task(
                    ts_interaction(
                        teacher_factory=self.teacher_factory,
                        student_factory=self.student_factory,
                        raters=self.raters,
                        init_prompt=example_prompt,
                        init_role=example_role,
                        max_turns=self.max_turns,
                        metadata=example_metadata,
                        request_timeout=self.request_timeout,
                        max_turn_tokens=self.max_turn_tokens,
                        use_matrix=self.use_matrix,
                        expand_size=self.beam_size,
                    )
                )
            )
            if len(batch_tasks) >= self.max_n_processes:
                await save_outputs()

        while batch_tasks:
            await save_outputs()
        if batch_results:
            await save_outputs(True)
        pbar.close()

    def forward(
        self,
        init_prompts: List[str],
        init_roles: List[str],
        teacher_inst: List[str],
        student_inst: List[str],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Use asyncio + semaphore to simultaneously process the conversations up to the
        semaphore number.
        """
        # put the insts in the metadata
        for t_inst, s_inst, md in zip(teacher_inst, student_inst, metadata):
            md["teacher_inst"] = t_inst
            md["student_inst"] = s_inst

        # the system prompt for this batch should be the same for the same role
        self.teacher_factory.set_role_system_prompt(teacher_inst[0])
        self.student_factory.set_role_system_prompt(student_inst[0])

        asyncio.run(
            self.batch_ts_interaction(
                init_prompts,
                init_roles,
                metadata,
            )
        )

    def validation_step(self, batch: Dict[str, List[Any]], batch_idx: int) -> None:
        # input_tokens, target_mask, context_tokens, target_tokens, metadata = batch
        self.forward(
            [x for x in batch["init_prompt"] for _ in range(self.sampling_n)],
            [x for x in batch["init_role"] for _ in range(self.sampling_n)],
            [x for x in batch["teacher_inst"] for _ in range(self.sampling_n)],
            [x for x in batch["student_inst"] for _ in range(self.sampling_n)],
            [x for x in batch["metadata"] for _ in range(self.sampling_n)],
        )

    def on_validation_epoch_end(self) -> None:
        print("start to compute metrics")
        # compute the metrics in the end
        eval_metrics_dict = {}
        for k in self.metrics_dict.keys():
            eval_metrics_dict[k] = float(self.metrics_dict[k].compute())

        # log and save the evalution metrics
        print(f"validation result: {eval_metrics_dict}")
        self.log_dict(eval_metrics_dict, sync_dist=True)

        # reset all the metrics
        for k in self.metrics_dict.keys():
            self.metrics_dict[k].reset()

        with open(os.path.join(self.get_output_dir(), "metrics.json"), "w") as f:
            json.dump(eval_metrics_dict, f)

    def get_output_dir(self) -> str:
        if (log_dir := self.trainer.log_dir) is None:
            log_dir = os.getcwd()
            print(
                f"WARNING: `Trainer.log_dir` is not set, using cwd instead: {log_dir}"
            )
        return str(log_dir)
