import copy
import itertools
import queue
import random
import threading
import asyncio
import uuid
import time
import hashlib
import base64
import os
import json
from collections import defaultdict
from typing import Any, Union, Optional, Dict, List, Set, Tuple

import traceback

from roll.utils.constants import FixedTimeOut

import numpy as np
import ray
import torch
from datasets import Dataset
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import set_seed

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto, collate_fn
from roll.models.model_providers import default_tokenizer_provider
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.functionals import (
    postprocess_generate,
    reduce_metrics,
    concatenate_input_and_output,
    GenerateRequestType,
)
from roll.utils.logging import get_logger
from roll.utils.multi_thread_utils import ThreadSafeDict
import os

logger = get_logger()


@ray.remote(concurrency_groups={"single_thread": 1, "multi_thread": 128})
class GenerateScheduler:

    def __init__(self, pipeline_config=None):
        self.cluster: Union[Any, Cluster] = None
        self.pipeline_config = pipeline_config
        self.progress_bar: Optional[tqdm] = None
        self.request_counter = itertools.count()
        self.dp_fetch_count = {}
        self.load_balance_coordinator = {}
        self.mp_rank_zero = {}
        self.data: Optional[DataProto] = None
        self.responses: Dict[int, List[DataProto]] = defaultdict(list)
        self.request_id_2_prompt_id: Dict[str, int] = {}
        self.prompt_id_2_request_ids: Dict[int, set] = defaultdict(set)
        self.response_batch_size: Optional[int] = None
        self.abort_request_ids: set[str] = set()
        self.input_data: Optional[DataProto] = None
        self.is_completed = False
        self.request_id_2_dp_rank = {}
        self.completed_count = set()
        self.prompt_count = 0
        self.max_running_requests = 128
        self.alive_check_interval = 10
        self.last_alive_check = time.time()
        self.lock = threading.Lock()
        self.response_callback_fn = None

    def generate(self, data: DataProto, actor_cluster: Union[Any, Cluster], pipeline_config) -> DataProto:
        self.response_callback_fn = data.meta_info["response_callback_fn"]
        self.pipeline_config = pipeline_config
        self.cluster = actor_cluster
        if len(self.mp_rank_zero) == 0:
            dp_ranks: List[int] = [rank_info.dp_rank for rank_info in self.cluster.worker_rank_info]
            for i, dp_rank in enumerate(dp_ranks):
                rank_info = self.cluster.get_rank_info(rank=i)
                if rank_info.tp_rank == 0 and rank_info.pp_rank == 0 and rank_info.cp_rank == 0:
                    self.mp_rank_zero[dp_rank] = self.cluster.workers[i]
        self.dp_fetch_count = {dp_rank: 0 for dp_rank in self.mp_rank_zero.keys()}
        self.load_balance_coordinator = {dp_rank: 0 for dp_rank in self.mp_rank_zero.keys()}
        self.request_id_2_prompt_id.clear()
        self.prompt_id_2_request_ids.clear()
        self.abort_request_ids.clear()
        self.request_id_2_dp_rank.clear()
        self.completed_count.clear()

        generate_opt_level = pipeline_config.generate_opt_level
        num_return_sequences = actor_cluster.worker_config.generating_args.num_return_sequences

        is_num_return_sequences_expand = pipeline_config.is_num_return_sequences_expand
        if generate_opt_level == 0 and is_num_return_sequences_expand:
            logger.warning("is_num_return_sequences_expand=True and generate_opt_level may reduce performance.")

        data.batch["prompt_id"] = torch.arange(data.batch.batch_size[0], device=data.batch.device)
        self.input_data = data
        data.meta_info["is_num_return_sequences_expand"] = is_num_return_sequences_expand
        data.meta_info["num_return_sequences"] = num_return_sequences

        self.prompt_count = self.input_data.batch.batch_size[0]

        generation_config = self.cluster.worker_config.generating_args.to_dict()
        generation_config["num_return_sequences"] = num_return_sequences
        if is_num_return_sequences_expand:
            generation_config["num_return_sequences"] = 1
        data.meta_info["generation_config"] = generation_config

        if generate_opt_level == 0:
            if is_num_return_sequences_expand:
                batch_size = data.batch.batch_size[0]
                output_batch_size = batch_size * num_return_sequences
                input_ids = data.batch["input_ids"]
                attention_mask = data.batch["attention_mask"]
                position_ids = data.batch["position_ids"]
                input_ids = input_ids.unsqueeze(1).repeat(1, num_return_sequences, 1).view(output_batch_size, -1)
                attention_mask = (
                    attention_mask.unsqueeze(1).repeat(1, num_return_sequences, 1).view(output_batch_size, -1)
                )
                if position_ids.dim() == 3:  # (bsz, 3, seqlen)
                    # qwen2vl mrope, maybe use a placeholder and let model generate position_ids
                    position_ids = (
                        position_ids.unsqueeze(1)
                        .repeat(1, num_return_sequences, 1, 1)
                        .view(output_batch_size, *position_ids.shape[-2:])
                    )
                else:
                    position_ids = (
                        position_ids.unsqueeze(1).repeat(1, num_return_sequences, 1).view(output_batch_size, -1)
                    )

                non_tensor_batch = dict(
                    (k, np.repeat(v, num_return_sequences)) for k, v in data.non_tensor_batch.items()
                )

                data = DataProto(
                    batch=TensorDict(
                        {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids},
                        batch_size=output_batch_size,
                    ),
                    non_tensor_batch=non_tensor_batch,
                    meta_info=data.meta_info,
                )
            ret = self.cluster.generate(data)
            self.input_data = None
            return ret
        elif generate_opt_level == 1:
            # async + load balance
            if is_num_return_sequences_expand:
                batch_size = data.batch.batch_size[0]
                output_batch_size = batch_size * num_return_sequences
                input_ids = data.batch["input_ids"]
                attention_mask = data.batch["attention_mask"]
                position_ids = data.batch["position_ids"]
                prompt_ids = data.batch["prompt_id"]
                input_ids = input_ids.repeat(num_return_sequences, 1)
                attention_mask = attention_mask.repeat(num_return_sequences, 1)
                if position_ids.dim() == 3:  # (bsz, 3, seqlen)
                    position_ids = position_ids.repeat(num_return_sequences, 1, 1)
                    non_tensor_batch = dict(
                        (k, np.tile(v, num_return_sequences))
                        for k, v in data.non_tensor_batch.items())
                else:
                    position_ids = position_ids.repeat(num_return_sequences, 1)
                    non_tensor_batch = {}
                prompt_ids = prompt_ids.unsqueeze(-1).repeat(num_return_sequences, 1)

                data = DataProto(
                    batch=TensorDict(
                        {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "position_ids": position_ids,
                            "prompt_id": prompt_ids,
                        },
                        batch_size=output_batch_size,
                    ),
                    non_tensor_batch=non_tensor_batch,
                    meta_info=data.meta_info,
                )
            self.is_completed = False
            ret = self.generate_opt_level_1(data)
            self.input_data = ret
            return ret
        else:
            raise NotImplementedError(f"not support generate_opt_level {generate_opt_level}")

    def get_available_dp_rank(self):
        while True:
            # 负载均衡逻辑，期望各dp 正在处理的条数基本接近
            sorted_ranks = sorted(
                self.load_balance_coordinator.keys(), key=lambda rank: (self.load_balance_coordinator[rank], rank)
            )
            if self.load_balance_coordinator[sorted_ranks[0]] < self.max_running_requests:
                yield sorted_ranks[0]

    def send_request_to_one_worker(self, data: DataProto):
        dp_rank = next(self.get_available_dp_rank())
        ray.get(self.cluster.workers[dp_rank].add_request.remote(command=GenerateRequestType.ADD, data=data))
        self.load_balance_coordinator[dp_rank] += 1
        self.dp_fetch_count[dp_rank] += 1

    def generate_opt_level_1(self, data: DataProto):
        # async++
        is_num_return_sequences_expand = self.pipeline_config.is_num_return_sequences_expand
        num_return_sequences = self.cluster.worker_config.generating_args.num_return_sequences

        response_batch_size = 1 if is_num_return_sequences_expand else num_return_sequences
        self.response_batch_size = response_batch_size
        self.progress_bar = tqdm(
            total=self.prompt_count, desc="generate progress(prompt)", mininterval=int(self.prompt_count * 0.1) + 1
        )

        self.data = data
        self.responses: Dict[int, List[DataProto]] = defaultdict(list)

        logger.info(
            f"request id size: {data.batch.batch_size[0]} "
            f"response_batch_size: {response_batch_size} "
            f"is_num_return_sequences_expand: {is_num_return_sequences_expand}"
        )
        self.cluster.start_server(data=DataProto(meta_info=data.meta_info), blocking=True)

        # 分发数据至收到target rollout 完成
        # 无限循环，把所有的response发送给dp worker
        send_request_count = 0
        request_refs = []
        data_index_counter = itertools.count()
        last_alive_check = time.time()
        while not self.is_completed:

            # 探测dp worker是否存活，dp worker的server thread可能由于异常退出，造成hang
            current_time = time.time()
            if current_time - last_alive_check >= self.alive_check_interval:
                self.cluster.add_request(command=GenerateRequestType.ALIVE_CHECK, data=DataProto())
                last_alive_check = current_time

            if send_request_count < data.batch.batch_size[0]:
                # 取一个可以发送request的dp worker
                dp_rank = next(self.get_available_dp_rank())

                # 还有数据需要发送, 取需要发送的数据
                # request_id 全局递增，否则vllm/sglang scheduler状态不对
                request_id = next(self.request_counter)
                data_index = next(data_index_counter)
                request_data = collate_fn([self.data[data_index]])
                request_data.meta_info["request_id"] = str(request_id)
                prompt_id = self.data[data_index].batch["prompt_id"].item()
                self.request_id_2_prompt_id[request_data.meta_info["request_id"]] = request_data.batch[
                    "prompt_id"
                ].item()
                self.request_id_2_dp_rank[request_data.meta_info["request_id"]] = dp_rank
                self.prompt_id_2_request_ids[prompt_id].add(request_data.meta_info["request_id"])
                # 需要注意上面的调用顺序, report_response中会更新request_id索引dp_rank，所以这里需要最后add request_id
                request_data.meta_info["response_callback_fn"] = self.response_callback_fn
                request_data.meta_info["generation_config"] = data.meta_info["generation_config"]
                request_refs.append(
                    self.cluster.workers[dp_rank].add_request.remote(
                        command=GenerateRequestType.ADD, data=request_data
                    )
                )
                with self.lock:
                    self.load_balance_coordinator[dp_rank] += 1
                self.dp_fetch_count[dp_rank] += 1
                send_request_count += 1
                if len(request_refs) % self.cluster.world_size == 0:
                    ray.get(request_refs)
                    request_refs = []

        gen_metrics = self.cluster.stop_server()
        generate_return_num = num_return_sequences
        response_ids_list_of_list = []
        eos_token_id = None
        pad_token_id = None
        for sample_index in range(len(self.responses)):
            response_ids_list = []
            for response in self.responses[sample_index]:
                eos_token_id = response.meta_info["eos_token_id"]
                pad_token_id = response.meta_info["pad_token_id"]
                response_ids_list.extend(response.meta_info["output_token_ids"])
            assert (
                len(response_ids_list) >= generate_return_num
            ), f"response_ids_list length {len(response_ids_list)} < generate_return_num {generate_return_num}"
            response_ids_list_of_list.extend(response_ids_list[:generate_return_num])

        response_ids_list_of_list = [torch.tensor(token_ids) for token_ids in response_ids_list_of_list]
        output_tensor = pad_sequence(response_ids_list_of_list, batch_first=True, padding_value=pad_token_id)
        output_tensor = concatenate_input_and_output(
            input_ids=self.input_data.batch["input_ids"],
            output_ids=output_tensor,
            num_return_sequences=generate_return_num,
        )
        output: DataProto = postprocess_generate(
            prompts=self.input_data,
            output=output_tensor,
            num_return_sequences=generate_return_num,
            sequence_length=self.pipeline_config.sequence_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        _, sorted_indices = torch.sort(output.batch["prompt_id"])
        output.reorder(indices=sorted_indices)
        output.pop("prompt_id")
        self.data = None
        output.meta_info["metrics"] = reduce_metrics(gen_metrics.meta_info.pop("metrics", {}))
        logger.info(f"dp_fetch_count: {self.dp_fetch_count}")
        return output

    @ray.method(concurrency_group="single_thread")
    def report_response(self, data: DataProto):
        """
        本质上也是维护了一个状态机
        """
        request_id = data.meta_info["request_id"]
        prompt_id = self.request_id_2_prompt_id[request_id]
        dp_rank = self.request_id_2_dp_rank[request_id]
        with self.lock:
            self.load_balance_coordinator[dp_rank] -= 1

        if self.is_completed:
            return

        self.responses[prompt_id].append(data)
        required_response_count = self.cluster.worker_config.generating_args.num_return_sequences
        self.prompt_id_2_request_ids[prompt_id].remove(data.meta_info["request_id"])
        if len(self.responses[prompt_id]) * self.response_batch_size >= required_response_count:
            # 取已经完成的prompt_id，对应的request_ids，需要都取消
            if prompt_id not in self.completed_count:
                self.progress_bar.update(1)
            self.completed_count.add(prompt_id)
            abort_refs = []
            for request_id in self.prompt_id_2_request_ids[prompt_id]:
                with self.lock:
                    self.load_balance_coordinator[dp_rank] -= 1
                abort_refs.append(
                    self.cluster.workers[dp_rank].add_request.remote(
                        command=GenerateRequestType.ABORT, data=DataProto(meta_info={"request_id": request_id})
                    )
                )
        if len(self.completed_count) >= self.prompt_count:
            self.is_completed = True

@ray.remote(concurrency_groups={"single_thread": 1, "multi_thread": 256})
class DynamicSamplingScheduler:

    def __init__(self, pipeline_config=None):
        self.pipeline_config = pipeline_config
        set_seed(seed=pipeline_config.seed)
        self.progress_bar: Optional[tqdm] = None
        self.request_counter = None
        self.dp_fetch_count = {}
        self.load_balance_coordinator = {}
        self.mp_rank_zero = {}
        self.request_id_2_prompt_id: Dict[str, int] = {}
        self.prompt_id_2_request_ids: Dict[int, set] = defaultdict(set)
        # prompt_id to unique prompt hash value
        self.prompt_id_2_hash_str: Dict[int, str] = {}
        # NOTE: 'difficult' prompts
        # self.long_prompt_request_data: Dict[str, Any] = {}
        self.promp_hash_to_request_data: Dict[str, DataProto] = {}
        self.response_batch_size: Optional[int] = None
        self.abort_request_ids: set[str] = set()
        self.request_id_2_dp_rank = {}
        self.requests_buffers: Dict[str, DataProto] = {}
        self.lock = threading.Lock()
        self.last_alive_check = time.time()
        self.dataset_iter_count = 0
        self.exception_queue = queue.Queue()
        self.running = False
        self.dataset_epoch = 0

        self.prompt_length_in_last_epoch: Dict[str, List[int]] = {}
        self.sensitive_prompts: List[str] = []

        # Flow control measures. max_running_requests limits the maximum number of concurrent requests for each dp.
        # max_prompts limits the number of prompts running simultaneously to avoid excessive consumption of prompts.

        self.actor_cluster = None
        self.reward_clusters = None
        self.reward_worker_iters = None
        self.dataset = None
        self.indices = []
        self.batch_size = None
        self.dataset_iter = None
        self.collect_fn_cls = None
        self.collect_fn_kwargs = None
        self.collect_fn = None
        self.tokenizer = None
        self.response_filter_fn = None
        self.query_filter_fn = None
        self.response_callback_fn = None
        self.generation_config = None

        self.completed_buffers = None
        self.query_group_buffers = None
        self.reward_times = None

        self.query_filter_count = 0
        self.response_filter_count = 0
        self.running_prompts = 0
        self.response_cache: Dict[str, List] = None
        self.prompt_use_count = 0
        self.fetch_long_prompts = False

    def set_scheduler(
        self,
        actor_cluster: Union[Any, Cluster],
        reward_clusters: Dict[str, Union[Any, Cluster]],
        dataset: Dataset,
        collect_fn_cls,
        collect_fn_kwargs,
        response_filter_fn=None,
        query_filter_fn=None,
        response_callback_fn=None,
        state: Dict[str, Any] = None,
        val: bool = False
    ):
        """
        GenerateScheduler可以由多个实例，不再局限于单例
        """
        if not val:
            self.dynamic_sampling = self.pipeline_config.dynamic_sampling
            self.max_running_requests = self.pipeline_config.max_running_requests
            self.max_prompts_ratio = self.pipeline_config.max_prompts_ratio
            self.max_num_return_sequences = self.pipeline_config.max_num_return_sequences
            
            self.alive_check_interval = self.pipeline_config.alive_check_interval
        else:
            self.dynamic_sampling = False
            self.max_running_requests = 1024
            self.max_prompts_ratio = 1
            self.max_num_return_sequences = 1
            self.alive_check_interval = 10

        self.actor_cluster = actor_cluster
        self.reward_clusters = reward_clusters
        self.reward_worker_iters = {}
        for domain, cluster in reward_clusters.items():
            self.reward_worker_iters[domain] = itertools.cycle(cluster.workers)

        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        # NOTE: debug run, non random sample
        # self.indices = list(range(self.pipeline_config.rollout_batch_size))
        if state is not None and state.get("dataset_iter_count", 0) > 0:
            for _ in range(state["dataset_iter_count"]):
                self.get_next_dataset_item()

        self.collect_fn_cls = collect_fn_cls
        self.collect_fn_kwargs = collect_fn_kwargs
        self.tokenizer = default_tokenizer_provider(model_args=self.actor_cluster.worker_config.model_args)
        self.collect_fn = self.collect_fn_cls(tokenizer=self.tokenizer, **self.collect_fn_kwargs)

        if self.dynamic_sampling:
            self.response_filter_fn = response_filter_fn
            self.query_filter_fn = query_filter_fn
        else:
            self.response_filter_fn = lambda data_list, config: True
            self.query_filter_fn = lambda data_list, config: True
            logger.info(f"use_additional_prompts is False, disable query and response filtering.")
        self.response_callback_fn = response_callback_fn
        dp_ranks: List[int] = [rank_info.dp_rank for rank_info in self.actor_cluster.worker_rank_info]
        for i, dp_rank in enumerate(dp_ranks):
            rank_info = self.actor_cluster.get_rank_info(rank=i)
            if rank_info.tp_rank == 0 and rank_info.pp_rank == 0 and rank_info.cp_rank == 0:
                self.mp_rank_zero[dp_rank] = self.actor_cluster.workers[i]

        self.request_counter = GlobalCounter.options(
            name=f"DynamicSchedulerRequestCounter",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote()

    def reset_actor_cluster(self, actor_cluster: Union[Any, Cluster]):
        self.actor_cluster = actor_cluster
        self.tokenizer = default_tokenizer_provider(model_args=self.actor_cluster.worker_config.model_args)
        dp_ranks: List[int] = [rank_info.dp_rank for rank_info in self.actor_cluster.worker_rank_info]
        self.mp_rank_zero = {}
        for i, dp_rank in enumerate(dp_ranks):
            rank_info = self.actor_cluster.get_rank_info(rank=i)
            if rank_info.tp_rank == 0 and rank_info.pp_rank == 0 and rank_info.cp_rank == 0:
                self.mp_rank_zero[dp_rank] = self.actor_cluster.workers[i]

    def reset_status(self):
        self.completed_buffers: Dict[int, List[DataProto]] = defaultdict(list)
        self.query_group_buffers: Dict[int, List[DataProto]] = defaultdict(list)
        self.reward_times: Dict[int, List[DataProto]] = defaultdict(list)

        self.dp_fetch_count = {dp_rank: 0 for dp_rank in self.mp_rank_zero.keys()}
        self.load_balance_coordinator = {dp_rank: 0 for dp_rank in self.mp_rank_zero.keys()}
        self.request_id_2_prompt_id.clear()
        self.prompt_id_2_request_ids.clear()
        self.prompt_id_2_hash_str.clear()
        # NOTE: manually maintain this list across iterations
        # DEBUG: how to deal with issues across epochs? 
        # self.long_prompt_request_data = []
        # self.promp_hash_to_request_data.clear()
        self.abort_request_ids.clear()
        self.request_id_2_dp_rank.clear()
        self.requests_buffers.clear()
        self.response_filter_count = 0
        self.query_filter_count = 0
        self.running_prompts = 0
        self.prompt_use_count = 0
        self.response_cache = defaultdict(list)
        self.exception_queue = queue.Queue()
        bar_name = "-".join(self.reward_clusters.keys())
        self.progress_bar = tqdm(
            total=self.batch_size,
            desc=f"{bar_name} generate progress(prompt)",
            mininterval=int(self.batch_size * 0.1) + 1,
        )
        self.code_worker_priority_list: List[int] = list(range(8))
        self.fetch_cpu_util_ids = []
        self.code_worker_cpu_utilizations = [0 for _ in range(16)]
        self.code_worker_iter = itertools.cycle(self.code_worker_priority_list)

    def fetch_worker_util(self, domain):
        if len(self.fetch_cpu_util_ids) > 0:
            ready, _ = ray.wait(self.fetch_cpu_util_ids, timeout=0)
            if not ready:
                return
            else:
                fetch_cpu_util_id = ready[0]
            utilization = ray.get(fetch_cpu_util_id)
            self.code_worker_cpu_utilizations[self.fetch_cpu_util_ids.index(fetch_cpu_util_id)] = utilization
            self.fetch_cpu_util_ids.remove(fetch_cpu_util_id)
            if len(self.fetch_cpu_util_ids) == 0:
                self.code_worker_priority_list = []
                self.fetch_cpu_util_ids = [
                    _worker.get_cpu_utilization.remote() 
                    for _worker in self.reward_clusters[domain].workers
                ]
                print(f"Fetched Util from Ray!")
                worker_utils = sorted(
                    zip(range(len(self.code_worker_cpu_utilizations)), self.code_worker_cpu_utilizations), 
                    key=lambda x: x[1]
                )
                print(f"CPU utilizations: {self.code_worker_cpu_utilizations}, {time.time()}")
                top_k_workers_idx = 8
                utilization_gap_threshold = 25
                for _cur_idx in range(top_k_workers_idx):
                    self.code_worker_priority_list.append(worker_utils[_cur_idx][0])
                    if worker_utils[_cur_idx + 1][1] - worker_utils[0][1] >= utilization_gap_threshold:
                        break
                self.code_worker_iter = itertools.cycle(self.code_worker_priority_list)
                self.code_worker_cpu_utilizations = [0 for _ in range(16)]

        else:
            self.fetch_cpu_util_ids = [
                _worker.get_cpu_utilization.remote() 
                for _worker in self.reward_clusters[domain].workers
            ]


    def _submit_request_with_allocate_dp(self, prompt_id, prompt_hash, request_data_list, domain):
        print(f"[get_batch_{domain}] submit start")

        for req in request_data_list:
            dp_rank = sorted(
                self.load_balance_coordinator.keys(), key=lambda rank: (self.load_balance_coordinator[rank], rank)
            )[0]
            request_id = ray.get(self.request_counter.get_value.remote())
            req.meta_info["request_id"] = f"{request_id}"
            req.meta_info["response_callback_fn"] = self.response_callback_fn
            self.request_id_2_prompt_id[req.meta_info["request_id"]] = prompt_id
            self.request_id_2_dp_rank[req.meta_info["request_id"]] = dp_rank
            self.prompt_id_2_request_ids[prompt_id].add(req.meta_info["request_id"])
            self.requests_buffers[req.meta_info["request_id"]] = req
            if int(os.environ.get("REPORT_LENGTH_AND_REWARDS", "0")):
                self.prompt_id_2_hash_str[prompt_id] = prompt_hash
            print(f"[get_batch_{domain}] submit {prompt_hash} request_id {request_id} to dp_rank {dp_rank}, {time.time()}")
            self.actor_cluster.workers[dp_rank].add_request.remote(
                command=GenerateRequestType.ADD, data=req
            )
            req.meta_info.pop("response_callback_fn")
            self.load_balance_coordinator[dp_rank] += 1
            self.dp_fetch_count[dp_rank] += 1

        

    

    def get_batch(self, data: DataProto, batch_size: int) -> DataProto:
        """
        Dynamic Sampling with reward filtering and dual-stage scheduling.
        NOTE: This version only contains basic wortflow of dual-stage scheduling, 
              Advanced features like prompt priority and auto-scaling resources TBA.
        """

        print(f"[get_batch] start, epoch {self.dataset_epoch}, {time.time()}")
        get_batch_start_time = time.time()
        self.batch_size = batch_size
        self.reset_status()
        self.running = True
        prompt_id_counter = itertools.count()
        self.generation_config = copy.deepcopy(data.meta_info["generation_config"])
        num_return_sequences = self.generation_config["num_return_sequences"]
        
        num_submit_sequences = self.pipeline_config.max_num_return_sequences
        max_prompts_submit = int(self.max_prompts_ratio * batch_size)

        added_prompts = 0
        prompts_to_submit = []

        iteration_submit_bs = self.batch_size
        while True:
            with self.lock:
                if np.sum([
                    len(v) >= num_return_sequences for v in list(self.completed_buffers.values())[:]
                ]) == self.batch_size:
                    self.running = False
                    print(f"[get_batch] finish, epoch {self.dataset_epoch}, {time.time()}")
                    break
            
            self.check_worker_alive(self.actor_cluster)
            self.check_response_callback()
            
            request_data_list: List[DataProto] = []

            while added_prompts < self.batch_size:
                dataset_item = self.get_next_dataset_item()
                prompt_digest = hashlib.md5(dataset_item['prompt'].encode() + dataset_item['messages'].encode()).digest()
                prompt_hash = base64.urlsafe_b64encode(prompt_digest).decode().rstrip('=')
                domain = dataset_item.get("domain", "default")
                collect_data = self.collect_fn([dataset_item])
                request_data: DataProto = DataProto.from_single_dict(collect_data, meta_info=data.meta_info)
                self.promp_hash_to_request_data[prompt_hash] = request_data
                prompts_to_submit.append(prompt_hash)
                added_prompts += 1

            if len(prompts_to_submit) > 0:
                assert len(prompts_to_submit) == self.batch_size, "Prompt to submit must equal to submit count!"
                with self.lock: 
                    # import pdb; pdb.set_trace()
                    for _ in range(self.batch_size):
                        print(len(prompts_to_submit), self.batch_size, _)
                        prompt_hash = prompts_to_submit[0]
                        prompts_to_submit.remove(prompt_hash)
                        request_data = self.promp_hash_to_request_data[prompt_hash]
                        domain = request_data.non_tensor_batch['domain'][0]
                        prompt_id = next(prompt_id_counter)
                        print(f"[get_batch_{domain}] prompt: {prompt_hash}, resp_num: {self.generation_config['num_return_sequences']}, time={time.time()}")
                        request_data_list = self.expand_requests(request_data)
                        self.running_prompts += 1
                        self._submit_request_with_allocate_dp(prompt_id, prompt_hash, request_data_list, domain)

        completed_buffers = {k: v for k, v in self.completed_buffers.items()}
        
        collect_data = [item for sublist in list(completed_buffers.values())[:] for item in sublist]
        query_use_count = next(prompt_id_counter)
        logger.info(
            f"total collect data: {len(collect_data)}, collect queries: {len(completed_buffers)} "
            f"used queries: {query_use_count}  query_filter_count: {self.query_filter_count} "
            f"response_filter_count: {self.response_filter_count}"
        )
        # TODO: 这里 len(collect_data) > rollout_batch_size, 可以尝试动态扩大batch_size
        batch = DataProto.concat(collect_data[: self.batch_size * num_return_sequences])
        batch.meta_info["metrics"] = {
            f"scheduler/query_filter_count": self.query_filter_count,
            f"scheduler/response_filter_count": self.response_filter_count,
            f"scheduler/collect_query_count": len(completed_buffers),
            f"scheduler/query_use_count": query_use_count,
        }

        # 统计全部response metrics
        metrics = {}
        for domain, response_batches in self.response_cache.items():
            response_batch = DataProto.concat(response_batches[:])
            sequence_score = response_batch.batch["scores"]
            metrics[f"scheduler/{domain}/score/mean"] = torch.mean(sequence_score).detach().item()
            metrics[f"scheduler/{domain}/score/max"] = torch.max(sequence_score).detach().item()
            metrics[f"scheduler/{domain}/score/min"] = torch.min(sequence_score).detach().item()

        batch.meta_info["metrics"].update(metrics)
        self.reset_status()

        get_batch_end_time = time.time()
        return batch

    @ray.method(concurrency_group="multi_thread")
    def report_response(self, data: DataProto):
        """
        这里需要考虑多线程数据访问
        data 返回可能有多条的 NOTE: 加锁！
        """
        try:
            request_id = data.meta_info["request_id"]
            prompt_id = self.request_id_2_prompt_id.pop(request_id)

            num_return_sequences = self.generation_config["num_return_sequences"]
            
            batch = self.postprocess_output_ids(data)
            output_count = batch.batch.batch_size[0]
            reward_worker_idx = -1
            with self.lock:
                self.load_balance_coordinator[self.request_id_2_dp_rank[request_id]] -= 1
                self.prompt_id_2_request_ids[prompt_id].remove(request_id)
                domain = "default"
                if "domain" in batch.non_tensor_batch.keys():
                    domain = batch.non_tensor_batch["domain"][0]
                reward_worker = next(self.reward_worker_iters[domain])
            
            print(f"[fin_rollout] request_id {request_id} in, {self.code_worker_priority_list}, {time.time()}")
                
            print(f"[report_response_{domain}] process request_id {request_id}, reward_worker {reward_worker_idx}, {time.time()}")
            if not self.running:
                return
            rewards_start = time.time()
            if domain == "code_sandbox":
                rewards = None
                try:
                    rewards: DataProto = ray.get(reward_worker.compute_rewards.remote(batch), timeout=FixedTimeOut)
                except Exception as e:
                    token_level_rewards = torch.zeros_like(batch.batch["responses"], dtype=torch.float16)
                    response_level_rewards = torch.zeros(len(batch.non_tensor_batch["id"]), dtype=torch.float16)
                    scores = torch.zeros(len(batch.non_tensor_batch["id"]), dtype=torch.float16)

                    rewards = DataProto.from_dict(
                        tensors={
                            "token_level_rewards": token_level_rewards,
                            "response_level_rewards": response_level_rewards,
                            "scores": scores
                        }
                    )
            else:
                rewards: DataProto = ray.get(reward_worker.compute_rewards.remote(batch))
            batch.union(rewards)
            rewards_end = time.time()

            response_buffers: List[DataProto] = []
            batch_expanded = [batch[[idx]] for idx in range(output_count)]
                
            # response_filter, 不太需要response filter
            # NOTE: 暂时没用就是True过了一遍
            for batch_item in batch_expanded:
                if self.response_filter_fn(batch_item, self.pipeline_config):
                    response_buffers.append(batch_item)
                else:
                    self.response_filter_count += 1

            with self.lock:
                self.response_cache[domain].extend(batch_expanded)

                if len(response_buffers) == 0:
                    if len(self.prompt_id_2_request_ids[prompt_id]) == 0:
                        self.running_prompts -= 1
                        prompt_hash = self.prompt_id_2_hash_str.pop(prompt_id)
                        if prompt_hash in self.promp_hash_to_request_data:
                            self.promp_hash_to_request_data.pop(prompt_hash)
                    return

                if len(self.completed_buffers[prompt_id]) >= num_return_sequences:
                    return

                # expand batch to response
                self.query_group_buffers[prompt_id].extend(response_buffers)
                # NOTE: only valid for expanded requests!
                self.reward_times[prompt_id].append(rewards_end - rewards_start)

                # no query_filter in this version
                response_count = len(self.query_group_buffers[prompt_id])
                if response_count == num_return_sequences:
                    if not self.running:
                        return
                    dropped = False
                    if np.sum([
                        len(v) >= num_return_sequences for v in list(self.completed_buffers.values())[:]
                    ]) < self.batch_size:
                        self.completed_buffers[prompt_id].extend(
                            self.query_group_buffers[prompt_id][:num_return_sequences]
                        )
                    else:
                        dropped = True
                    if len(self.prompt_id_2_request_ids[prompt_id]) > 0:
                        self.abort_requests(self.prompt_id_2_request_ids[prompt_id])
                    self.running_prompts -= 1

                    if prompt_id in self.prompt_id_2_hash_str:
                        prompt_hash = self.prompt_id_2_hash_str.pop(prompt_id)
                        if prompt_hash in self.promp_hash_to_request_data and not dropped:
                            self.promp_hash_to_request_data.pop(prompt_hash)

                        if int(os.environ.get("REPORT_LENGTH_AND_REWARDS", "0")) and not dropped:
                            # report response level rewards
                            response_level_rewards = [data.batch["response_level_rewards"] for data in self.query_group_buffers[prompt_id]]
                            response_rewards = torch.cat(response_level_rewards, dim=0).long().cpu().tolist()
                            prompt_response_proto = DataProto.concat(self.query_group_buffers[prompt_id][:num_return_sequences])
                            # report response level lengths
                            response_lengths = torch.sum(prompt_response_proto.batch["response_mask"], dim=1).cpu().tolist()

                            print(f"[report_response_{domain}] prompt {prompt_hash}, finish rollout, {time.time()}")
                            print(f"[report_response] max_length: {np.max(response_lengths)}")
                            lengths_and_rewards = {
                                'domain': domain,
                                'prompt_hash': prompt_hash,
                                'response_lengths': response_lengths,
                                'response_rewards': response_rewards,
                                'rewards_times': self.reward_times[prompt_id],
                                'unfinished_prompts': len(self.promp_hash_to_request_data.keys()),
                                'long_iteration': self.fetch_long_prompts
                            }
                            self.reward_times[prompt_id] = []
                            self.prompt_length_in_last_epoch[prompt_hash] = response_lengths
                            length_dir = os.path.join(self.pipeline_config.profiler_output_dir, "length")
                            os.makedirs(length_dir, exist_ok=True)
                            print(f"report log here!!!, {length_dir}")
                            filename = f"response-length-and-rewards-{domain}-ep{self.dataset_epoch}.jsonl"
                            length_file_path = os.path.join(length_dir, filename)
                            with open(length_file_path, "a") as f:
                                f.write(json.dumps(lengths_and_rewards) + "\n")

        except Exception as e:
            print(f"ERROR! {e}")
            traceback.print_exc()
            self.exception_queue.put(e)

    def get_next_dataset_item(self):
        print(f"DATA SIZE CHECK: {len(self.indices)}")
        if self.dataset_iter is None:
            random.seed(self.pipeline_config.seed + self.dataset_epoch)
            random.shuffle(self.indices)
            self.dataset_iter = iter(self.indices)
            logger.info(f"{'-'.join(self.reward_clusters.keys())} dataset epoch: {self.dataset_epoch}")

        try:
            dataset_item = self.dataset[next(self.dataset_iter)]
        except StopIteration:
            self.dataset_epoch += 1
            random.seed(self.pipeline_config.seed + self.dataset_epoch)
            random.shuffle(self.indices)
            self.dataset_iter = iter(self.indices)
            dataset_item = self.dataset[next(self.dataset_iter)]
            logger.info(f"{'-'.join(self.reward_clusters.keys())} dataset epoch: {self.dataset_epoch}")
        self.dataset_iter_count += 1
        print(f"GET DATA ITEM")
        return dataset_item

    def get_scheduler_state(self):
        return {"dataset_iter_count": self.dataset_iter_count}

    def abort_requests(self, request_ids: Set[str]):
        abort_refs = []
        for request_id in request_ids:
            dp_rank = self.request_id_2_dp_rank[request_id]
            self.load_balance_coordinator[dp_rank] -= 1
            abort_refs.append(
                self.actor_cluster.workers[dp_rank].add_request.remote(
                    command=GenerateRequestType.ABORT, data=DataProto(meta_info={"request_id": request_id})
                )
            )

    def postprocess_output_ids(self, data: DataProto) -> DataProto:
        # postprocess_generate, input_ids, attention_mask, left pad
        request_id = data.meta_info["request_id"]
        request: DataProto = self.requests_buffers.pop(request_id)

        eos_token_id = data.meta_info["eos_token_id"]
        pad_token_id = data.meta_info["pad_token_id"]
        output_token_ids = data.meta_info["output_token_ids"]
        output_tokens = [torch.tensor(token_ids) for token_ids in output_token_ids]
        output_tensor = pad_sequence(output_tokens, batch_first=True, padding_value=pad_token_id)
        output_tensor = concatenate_input_and_output(
            input_ids=request.batch["input_ids"], output_ids=output_tensor, num_return_sequences=len(output_tokens)
        )
        output: DataProto = postprocess_generate(
            prompts=request,
            output=output_tensor,
            num_return_sequences=len(output_tokens),
            sequence_length=self.pipeline_config.sequence_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        request_repeat = request.repeat(repeat_times=len(output_tokens))
        output.non_tensor_batch = request_repeat.non_tensor_batch
        output.meta_info = request_repeat.meta_info
        output.meta_info["request_id"] = request_id
        return output

    def expand_requests(
        self, data: DataProto
    ) -> List[DataProto]:
        """
        replica, 以及redundancy
        """
        generate_opt_level = self.pipeline_config.generate_opt_level
        # is_num_return_sequences_expand = self.pipeline_config.is_num_return_sequences_expand
        # assert is_num_return_sequences_expand == False, "Baseline scheduler do not support reponse level control"
        assert generate_opt_level > 0, (
            f"generate_opt_level {generate_opt_level} should > 0, " f"in dynamic sampling scheduler."
        )
        assert "generation_config" in data.meta_info, f"data {data.meta_info} should have key 'generation_config'"

        target_requests = []
        data.meta_info["generation_config"]["num_return_sequences"] = self.generation_config["num_return_sequences"]
        target_requests.append(copy.deepcopy(data))

        return target_requests

    def check_worker_alive(self, cluster):
        # 探测dp worker是否存活，dp worker的server thread可能由于异常退出，造成hang
        current_time = time.time()
        if current_time - self.last_alive_check >= self.alive_check_interval:
            cluster.add_request(command=GenerateRequestType.ALIVE_CHECK, data=DataProto())
            self.last_alive_check = current_time

    def check_response_callback(self):
        if self.exception_queue.qsize() > 0:
            e = self.exception_queue.get()
            logger.error(f"report_response get exception {e}")
            raise e

    def check_send_new_request(self) -> bool:
        if self.running_prompts >= (self.batch_size + self.max_additional_running_prompts):
            return False
        if not self.is_use_additional_prompts and self.prompt_use_count >= self.batch_size:
            return False
        return True

    def get_available_dp_rank(self):
        while True:
            # 负载均衡逻辑，期望各dp 正在处理的条数基本接近
            sorted_ranks = sorted(
                self.load_balance_coordinator.keys(), key=lambda rank: (self.load_balance_coordinator[rank], rank)
            )
            print(f"[DEBUG] self.load_balance_coordinator {self.load_balance_coordinator}, max_running_requests {self.max_running_requests}")
            if self.load_balance_coordinator[sorted_ranks[0]] < self.max_running_requests:
                yield sorted_ranks[0]
            else:
                time.sleep(0.1)
            


@ray.remote
class GlobalCounter:
    def __init__(self):
        self.value = -1

    def get_value(self):
        self.value += 1
        return self.value


@ray.remote
class RequestScheduler:
    def __init__(self, infer_cluster, pipeline_config):
        self.infer_cluster = infer_cluster
        self.pipeline_config = pipeline_config
        self.request_id = uuid.uuid4()
        self.request_counter = 0
        self.src_rank2_dp_rank = {}
        self.request_id_2_dp_rank = {}
        self.inflight_requests: List[Dict[str, asyncio.Future]] = [{} for _ in range(self.infer_cluster.world_size)]
        self.worker_iter = itertools.cycle(range(self.infer_cluster.world_size))

        self.need_suspend = False
        self.suspend_notifier = asyncio.Event()

    async def generate_one_request(self, data: DataProto):
        await self._check_suspend()

        src_rank = data.meta_info["src_rank"]
        if src_rank not in self.src_rank2_dp_rank:
            dp_rank = next(self.worker_iter)
            self.src_rank2_dp_rank[src_rank] = dp_rank
        dp_rank = self.src_rank2_dp_rank[src_rank]
        request_id = f"{self.request_id}_{self.request_counter}"
        self.request_counter += 1
        data.meta_info["request_id"] = request_id
        fut = asyncio.Future()
        self.request_id_2_dp_rank[request_id] = dp_rank
        self.inflight_requests[dp_rank][request_id] = fut
        ref = self.infer_cluster.workers[dp_rank].add_request.remote(command=GenerateRequestType.ADD, data=data)
        await asyncio.wrap_future(ref.future())
        response_data = await fut
        if response_data is None:
            # request aborted
            return None

        # postprocess_generate, input_ids, attention_mask, left pad
        eos_token_id = response_data.meta_info["eos_token_id"]
        pad_token_id = response_data.meta_info["pad_token_id"]
        output_token_ids = response_data.meta_info["output_token_ids"]
        output_tokens = [torch.tensor(token_ids) for token_ids in output_token_ids]
        output_tensor = pad_sequence(output_tokens, batch_first=True, padding_value=pad_token_id)
        output_tensor = concatenate_input_and_output(
            input_ids=data.batch["input_ids"], output_ids=output_tensor, num_return_sequences=len(output_tokens)
        )
        output: DataProto = postprocess_generate(
            prompts=data,
            output=output_tensor,
            num_return_sequences=len(output_tokens),
            sequence_length=output_tensor.shape[-1],
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        request_repeat = data.repeat(repeat_times=len(output_tokens))
        output.non_tensor_batch = request_repeat.non_tensor_batch
        output.meta_info = request_repeat.meta_info
        return output

    async def report_response(self, data: DataProto, is_abort=False):
        request_id = data.meta_info["request_id"]
        if request_id not in self.request_id_2_dp_rank:
            return
        dp_rank = self.request_id_2_dp_rank.pop(request_id)
        fut = self.inflight_requests[dp_rank].pop(request_id)
        if is_abort:
            fut.set_result(None)
        else:
            fut.set_result(data)

    async def abort_request(self):
        futures = []
        for i in range(self.infer_cluster.world_size):
            if len(self.inflight_requests[i]) == 0:
                continue
            ref = self.infer_cluster.workers[i].add_request.remote(
                    command=GenerateRequestType.ABORT, data=DataProto(
                        meta_info={"request_id": [request_id for request_id in self.inflight_requests[i].keys()]}
                    )
                )
            futures.append(ref)
            for request_id in self.inflight_requests[i].keys():
                futures.append(self.report_response(data=DataProto(meta_info={"request_id": request_id}), is_abort=True))
        # must await at last, because report_response will mut inflight_requests
        await asyncio.gather(*futures)

    async def _check_suspend(self):
        while self.need_suspend:
            await self.suspend_notifier.wait()

    async def suspend(self):
        if self.need_suspend:
            return
        self.suspend_notifier.clear()
        self.need_suspend = True
        await self.abort_request()

    def resume(self):
        if not self.need_suspend:
            return
        self.need_suspend = False
        self.suspend_notifier.set()