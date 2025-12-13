import copy
import itertools
import queue
import random
import threading
import time
import hashlib
import base64
from collections import defaultdict
from typing import Any, Union, Optional, Dict, List, Set

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

logger = get_logger()


@ray.remote(concurrency_groups={"single_thread": 1, "multi_thread": 256})
class AsyncDynamicSamplingScheduler:

    def __init__(self, pipeline_config=None):
        self.pipeline_config = pipeline_config
        set_seed(seed=pipeline_config.seed)
        self.progress_bar: Optional[tqdm] = None
        self.request_counter = None
        self.dp_fetch_count = {}
        self.load_balance_coordinator = {}
        self.mp_rank_zero = {}
        # prompt_id to unique prompt hash value
        self.prompt_id_2_hash_str: Dict[int, str] = {}
        self.request_id_2_prompt_id: Dict[str, int] = {}
        self.prompt_id_2_request_ids: Dict[int, set] = defaultdict(set)
        self.response_batch_size: Optional[int] = None
        self.abort_request_ids: set[str] = set()
        self.request_id_2_dp_rank = {}
        self.requests_buffers: Dict[str, DataProto] = {}
        self.lock = threading.Lock()
        self.migrate_lock = threading.Lock()
        self.post_process_lock = threading.Lock()
        self.last_alive_check = time.time()
        self.dataset_iter_count = 0
        self.exception_queue = queue.Queue()
        self.running = False
        self.dataset_epoch = 0
        self.migrate_waiting_reqs = list()
        self.pre_pass_query_prompts: Dict[int, bool] = {}

        # Flow control measures. max_running_requests limits the maximum number of concurrent requests for each dp.
        # max_additional_running_prompts limits the number of prompts running simultaneously to avoid excessive consumption of prompts.
        self.dynamic_sampling = self.pipeline_config.dynamic_sampling
        self.is_use_pre_pass_filter = self.pipeline_config.is_use_pre_pass_filter
        self.max_running_requests = self.pipeline_config.max_running_requests
        self.max_additional_running_prompts = self.pipeline_config.max_additional_running_prompts
        self.is_use_additional_prompts = self.pipeline_config.is_use_additional_prompts
        self.alive_check_interval = self.pipeline_config.alive_check_interval

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
        self.pre_pass_query_filter_fn = None
        self.response_callback_fn = None
        self.generation_config = None

        self.completed_buffers = None
        self.query_group_buffers = None

        self.query_filter_count = 0
        self.response_filter_count = 0
        self.pre_pass_prompt_count = 0
        self.running_prompts = 0
        self.response_cache: Dict[str, List] = None
        self.prompt_use_count = 0
        self.has_scaled_down = False 
        self.stop_second_half_servers = False
        self.stop_all_servers = False
        self.infer_scaling_down_progress_ratio = self.pipeline_config.infer_scaling_down_progress_ratio 
        self.scaling_down_train_batch_size = self.pipeline_config.scaling_down_train_batch_size
        self.progress_profile = (self.pipeline_config.record_time_profiler_log_dir is not None)
        self.reward_worker_costs = list()
        self.init_record_time()
    
    def init_record_time(self, ):
        if self.progress_profile: 
            self.start_time = time.time() 
            self.record_time_list = list() 
    
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
        migrate_callback_fn=None,
        pre_pass_query_filter_fn=None,
    ):
        """
        GenerateScheduler可以由多个实例，不再局限于单例
        """
        self.actor_cluster = actor_cluster
        self.reward_clusters = reward_clusters
        self.reward_worker_iters = {}
        for domain, cluster in reward_clusters.items():
            self.reward_worker_iters[domain] = itertools.cycle(cluster.workers)

        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        if state is not None and state.get("dataset_iter_count", 0) > 0:
            for _ in range(state["dataset_iter_count"]):
                self.get_next_dataset_item()

        self.collect_fn_cls = collect_fn_cls
        self.collect_fn_kwargs = collect_fn_kwargs
        self.tokenizer = default_tokenizer_provider(model_args=self.actor_cluster.worker_config.model_args)
        self.collect_fn = self.collect_fn_cls(tokenizer=self.tokenizer, **self.collect_fn_kwargs)

        if self.is_use_additional_prompts:
            self.response_filter_fn = response_filter_fn
            self.query_filter_fn = query_filter_fn
            self.pre_pass_query_filter_fn = pre_pass_query_filter_fn
        else:
            self.response_filter_fn = lambda data_list, config: True
            self.query_filter_fn = lambda data_list, config: True
            self.pre_pass_query_filter_fn = lambda data_list, config: True
            logger.info(f"use_additional_prompts is False, disable query and response filtering.")
        if not self.is_use_pre_pass_filter: 
            self.pre_pass_query_filter_fn = None

        self.response_callback_fn = response_callback_fn
        self.migrate_callback_fn = migrate_callback_fn
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

    def reset_status(self, batch_size=None):
        if batch_size: 
            self.batch_size = batch_size
        self.completed_buffers: Dict[int, List[DataProto]] = defaultdict(list)
        self.query_group_buffers: Dict[int, List[DataProto]] = defaultdict(list)

        self.dp_fetch_count = {dp_rank: 0 for dp_rank in self.mp_rank_zero.keys()}
        self.load_balance_coordinator = {dp_rank: 0 for dp_rank in self.mp_rank_zero.keys()}
        self.request_id_2_prompt_id.clear()
        self.prompt_id_2_request_ids.clear()
        self.prompt_id_2_hash_str.clear()
        self.abort_request_ids.clear()
        self.request_id_2_dp_rank.clear()
        self.requests_buffers.clear()
        self.pre_pass_query_prompts.clear() 

        self.response_filter_count = 0
        self.query_filter_count = 0
        self.running_prompts = 0
        self.prompt_use_count = 0
        self.pre_pass_prompt_count = 0
        self.has_pre_abort_requests = False
        
        self.response_cache = defaultdict(list)
        self.exception_queue = queue.Queue()
        bar_name = "-".join(self.reward_clusters.keys())
        self.progress_bar = tqdm(
            total=self.batch_size,
            desc=f"{bar_name} generate progress(prompt)",
            mininterval=int(self.batch_size * 0.1) + 1,
        )
        self.has_scaled_down = False
        self.number_of_finished_all_requests_for_each_prompt = 0
        self.migrate_completion_ranks = dict()
        self.stop_second_half_servers = False 
        self.force_max_new_tokens = None
        self.prefetch_tags = dict()
        self.stop_all_servers = False
        self.disable_preftch_requests = False
        self.reward_worker_costs = list()
        


    def get_batch(self, data: DataProto, batch_size: int, first_half: bool=False, disable_stop_server: bool=False) -> DataProto:
        """
        从dataset里，按给定策略sample batch
        1. 常规无过滤
        2. 动态过滤
        """
        self.batch_size = batch_size
        self.reset_status()
        self.running = True
        prompt_id_counter = itertools.count()
        self.generation_config = copy.deepcopy(data.meta_info["generation_config"])
        num_return_sequences = self.generation_config["num_return_sequences"]
        send_request_count = 0
        self.meta_info = data.meta_info

        self.init_record_time() 

        world_size = len(self.actor_cluster.workers)
        all_ranks = list(range(world_size))

        while True:
            number_of_completed_requests = sum([len(v) for v in list(self.completed_buffers.values())[:]])
            if (
                number_of_completed_requests >= self.batch_size * num_return_sequences
            ):
                self.running = False
                break
            self.check_worker_alive(self.actor_cluster, first_half=first_half)
            self.check_response_callback()

            # place before `check_send_new_request``
            if not self.has_scaled_down and self.infer_scaling_down_progress_ratio > 0 and \
                    self.number_of_finished_all_requests_for_each_prompt >= int(self.batch_size * self.infer_scaling_down_progress_ratio): 
                print('has completed requests {} / number_of_finished_all_requests_for_each_prompt {}'.format(number_of_completed_requests, self.number_of_finished_all_requests_for_each_prompt), flush=True)
                dp_ranks = sorted(self.mp_rank_zero.keys())
                length = len(dp_ranks) // 2
                for dp_rank in dp_ranks[length:]: 
                    self.migrate_requests(request_id=None, dp_rank=dp_rank)
                    
                # self.actor_cluster.stop_server_second_half() 
                # TODO: clean up remaining things 
                # 1. wait for a signal: that all requests has been migrated to somewhere 
                # 2. we can stop this server
                # 3. signal main process half of servers has been completed 
                self.has_scaled_down = True
                first_half = True
                # self.max_running_requests *= 2 # avoid the head-of-line issues of migrated requests because most of them has short remaining response length
                continue 
            
            if self.has_scaled_down and (not self.stop_second_half_servers) and len(self.migrate_completion_ranks) == len(self.mp_rank_zero) // 2: 
                dp_ranks = sorted(self.mp_rank_zero.keys())
                length = len(dp_ranks) // 2
                for dp_rank in dp_ranks[length:]: 
                    self.load_balance_coordinator.pop(dp_rank)

                self.actor_cluster.stop_server_second_half(_ranks=copy.deepcopy(self.pipeline_config.second_half_ranks))
                self.stop_second_half_servers = True


            if len(self.migrate_waiting_reqs) > 0:
                while self.migrate_waiting_reqs:
                    dp_rank, _ = next(self.get_available_dp_rank(first_half=first_half))
                    with self.migrate_lock:
                        req = self.migrate_waiting_reqs.pop()
                    with self.lock:
                        # self.request_id_2_prompt_id[req.meta_info["request_id"]] = prompt_id
                        self.request_id_2_dp_rank[req.meta_info["request_id"]] = dp_rank
                        # self.prompt_id_2_request_ids[prompt_id].add(req.meta_info["request_id"])  # 用于replica情况
                        self.requests_buffers[req.meta_info["request_id"]] = req
                        
                        ray.get(
                            self.actor_cluster.workers[dp_rank].add_request.remote(
                                command=GenerateRequestType.RESUME, data=req
                            )
                        )
                        req.meta_info.pop("response_callback_fn")
                        self.load_balance_coordinator[dp_rank] += 1
                        self.dp_fetch_count[dp_rank] += 1
                continue 

            if not self.check_send_new_request():
                time.sleep(1)
                continue

            # get a query from dataset
            prompt_id = next(prompt_id_counter)
            dataset_item = self.get_next_dataset_item()
            
            domain = dataset_item.get("domain", "default")
            collect_data = self.collect_fn([dataset_item])
            request_data: DataProto = DataProto.from_single_dict(collect_data, meta_info=data.meta_info)
            
            # replica, redundancy
            request_data_list = self.expand_requests(request_data)
            
            
            self.prompt_use_count += 1
            self.running_prompts += 1
            dp_rank, _ = next(self.get_available_dp_rank(first_half=first_half))
            with self.lock:
                # get a available worker, 需要控制max_running_request, 当前策略会始终保持worker的满载
                for req in request_data_list:
                    request_id = ray.get(self.request_counter.get_value.remote())
                    req.meta_info["request_id"] = f"{request_id}"
                    req.meta_info["response_callback_fn"] = self.response_callback_fn
                    self.request_id_2_prompt_id[req.meta_info["request_id"]] = prompt_id
                    self.request_id_2_dp_rank[req.meta_info["request_id"]] = dp_rank
                    self.prompt_id_2_request_ids[prompt_id].add(req.meta_info["request_id"])  # 用于replica情况
                    self.requests_buffers[req.meta_info["request_id"]] = req
                    send_request_count += 1
                    
                    prompt_digest = hashlib.md5(dataset_item['prompt'].encode()).digest()
                    self.prompt_id_2_hash_str[prompt_id] = base64.urlsafe_b64encode(prompt_digest).decode().rstrip('=') # prompt_id 对应 unique prompt

                    ray.get(
                        self.actor_cluster.workers[dp_rank].add_request.remote(
                            command=GenerateRequestType.ADD, data=req
                        )
                    )
                    req.meta_info.pop("response_callback_fn")
                    self.load_balance_coordinator[dp_rank] += 1
                    self.dp_fetch_count[dp_rank] += 1

        
        if self.progress_profile: 
            self.record_time_list.append(
                {
                    'event_type': 'start_post_process',
                    'duration': time.time() - self.start_time
                }
            )
        if not first_half:
            if not disable_stop_server: 
                print("stop all servers ", flush=True)
                self.actor_cluster.stop_server()
        else:
            if not disable_stop_server:
                print("stop_server_first_half ", flush=True)
                self.actor_cluster.stop_server_first_half(_ranks=copy.deepcopy(self.pipeline_config.first_half_ranks))
        
        if (not self.stop_second_half_servers) and (self.has_scaled_down): 
            if not disable_stop_server:
                print("stop_server_second_half ", flush=True)
                self.actor_cluster.stop_server_second_half(_ranks=copy.deepcopy(self.pipeline_config.second_half_ranks))
            self.stop_second_half_servers = True
        
        print("="*40, flush=True)
        print("stop all servers ", flush=True)
        self.stop_all_servers = True 
        
        return_batch_size = self.batch_size
        with self.post_process_lock:
            completed_buffers = {k: v for k, v in self.completed_buffers.items() if len(v) > 0}
            if self.prefetch_tags is not None and len(self.prefetch_tags) > 0:
                completed_buffers = {k: v for k, v in self.completed_buffers.items() if len(v) > 0 and k not in self.prefetch_tags} # the remaining requests has been used to async training 
                return_batch_size = self.batch_size - len(self.prefetch_tags)
            self.disable_preftch_requests = True

        collect_data = [item for sublist in list(completed_buffers.values())[:] for item in sublist]
        query_use_count = next(prompt_id_counter)
        logger.info(
            f"total collect data: {len(collect_data)}, collect queries: {len(completed_buffers)} "
            f"used queries: {query_use_count}  query_filter_count: {self.query_filter_count} "
            f"response_filter_count: {self.response_filter_count}"
        )
        # TODO: 这里 len(collect_data) > rollout_batch_size, 可以尝试动态扩大batch_size
        print(" self.prefetch_tags is not None and len(self.prefetch_tags) > 0 is {}".format(self.prefetch_tags is not None and len(self.prefetch_tags) > 0))
        batch = DataProto.concat(collect_data[: return_batch_size * num_return_sequences]) # this step has something wrong
        batch.meta_info["metrics"] = {
            f"scheduler/query_filter_count": self.query_filter_count,
            f"scheduler/response_filter_count": self.response_filter_count,
            f"scheduler/collect_query_count": len(completed_buffers) + len(self.prefetch_tags) if (self.prefetch_tags is not None and len(self.prefetch_tags) > 0) else len(completed_buffers),
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

        reward_worker_cost_min = max([cost_tup[1] / cost_tup[0] for cost_tup in self.reward_worker_costs])
        reward_worker_cost_max = max([cost_tup[1] / cost_tup[0] for cost_tup in self.reward_worker_costs])
        reward_worker_cost_mean = np.mean([cost_tup[1] / cost_tup[0] for cost_tup in self.reward_worker_costs])
        batch.meta_info['time/reward_worker_costs'] = {
            f'time/{domain}/reward_worker_costs/max': reward_worker_cost_max,
            f'time/{domain}/reward_worker_costs/mean': reward_worker_cost_mean,
            f'time/{domain}/reward_worker_costs/min': reward_worker_cost_min,
        }

        if self.progress_profile: 
            self.record_time_list.append(
                {
                    'event_type': 'finish_post_process',
                    'duration': time.time() - self.start_time
                }
            )
        return batch

    @ray.method(concurrency_group="multi_thread")
    def prefetch_completed_requests(self, prefetch_prompt_count=-1, div_multipler=0) -> List[DataProto]: 
        print("self.disable_preftch_requests {}, not (self.stop_second_half_servers or self.stop_all_servers) {}".format(self.disable_preftch_requests, not (self.stop_second_half_servers or self.stop_all_servers)), flush=True)
        if self.disable_preftch_requests: 
            return DataProto(meta_info={'disable_preftch_requests': True})
        
        if not (self.stop_second_half_servers or self.stop_all_servers): 
            return DataProto(meta_info={'disable_preftch_requests': False})

        with self.post_process_lock:
            if self.progress_profile: 
                self.record_time_list.append(
                    {
                        'event_type': 'start_prefetch',
                        'duration': time.time() - self.start_time
                    }
                )
            num_return_sequences = self.generation_config["num_return_sequences"]
            
            
            if self.scaling_down_train_batch_size > 0:
                completed_buffers = dict()
                cnt = 0
                for k, v in self.completed_buffers.items():
                    if (k in self.prefetch_tags): continue
                    if cnt < self.scaling_down_train_batch_size and len(v) >= num_return_sequences:
                        completed_buffers[k] = v[:num_return_sequences]
                        cnt += 1
                    if prefetch_prompt_count > 0 and cnt == prefetch_prompt_count: 
                        break

                    if cnt == self.scaling_down_train_batch_size: 
                        break
                
                while div_multipler > 0 and cnt % div_multipler > 0:
                    (key_to_remove, _) = next(iter(completed_buffers.items()))
                    completed_buffers.pop(key_to_remove)
                    cnt -= 1
            else: 
                completed_buffers = {k: v[:num_return_sequences] for k, v in self.completed_buffers.items() if len(v) >= num_return_sequences}
            
            if len(completed_buffers) == 0: 
                return DataProto(meta_info={'disable_preftch_requests': False})

            print("length of completed_buffers ", len(completed_buffers), flush=True)
            print("self.number_of_finished_all_requests_for_each_prompt == ", self.number_of_finished_all_requests_for_each_prompt, flush=True)
            collect_data = [item for sublist in list(completed_buffers.values())[:] for item in sublist]
            self.prefetch_tags.update({k: True for k, v in completed_buffers.items() if len(v) > 0})
            
            # TODO: 这里 len(collect_data) > rollout_batch_size, 可以尝试动态扩大batch_size
            print('length of collect_data {}'.format(len(collect_data)))
            if len(collect_data) == 0: 
                return DataProto()
            prefetch_batch = DataProto.concat(collect_data)
            prefetch_batch.meta_info["metrics"] = {
                f"scheduler/query_filter_count": self.query_filter_count,
                f"scheduler/response_filter_count": self.response_filter_count,
                f"scheduler/collect_query_count": len(completed_buffers),
            }
            if self.progress_profile: 
                self.record_time_list.append(
                    {
                        'event_type': 'finish_prefetch',
                        'duration': time.time() - self.start_time
                    }
                )
        return prefetch_batch

    @ray.method(concurrency_group="multi_thread")
    def can_prefetch_requests(self): 
        return self.stop_second_half_servers or self.stop_all_servers


    def migrate_requests(self, request_id: str, dp_rank:int = -1): 
        with self.lock: 
            if dp_rank >= 0: 
                ray.get(
                    self.actor_cluster.workers[dp_rank].add_request.remote(
                        command=GenerateRequestType.MIGRATE_ALL, data=DataProto(meta_info={"migrate_callback_fn": self.migrate_callback_fn, "dp_rank": dp_rank, "request_id": None})
                    )
                )
            else: 
                dp_rank = self.request_id_2_dp_rank[request_id]
                # ray.get(
                self.actor_cluster.workers[dp_rank].add_request.remote(
                    command=GenerateRequestType.MIGRATE, data=DataProto(meta_info={"request_id": request_id, "migrate_callback_fn": self.migrate_callback_fn, "dp_rank": -1})
                # )
                )
    
    @ray.method(concurrency_group="multi_thread")
    def migrate_response(self, data: List, dp_rank:int, request_id: str):
        if dp_rank == -1: # not migrate all requests from a single device
            raise NotImplementedError
        else: # migrate all requests in device `dp_rank`
            if True:
                if self.progress_profile: 
                    self.record_time_list.append(
                        {
                            'event_type': 'start_migration',
                            'duration': time.time() - self.start_time
                        }
                    )
                for req in data:
                    if isinstance(req, DataProto):
                        with self.migrate_lock:
                            self.migrate_waiting_reqs.append(data)
                    elif isinstance(req, dict):
                        has_generated_tokens = len(req['token_ids']) - req['prompt_length']
                        request_id = req['request_id']
                        ori_request: DataProto = self.requests_buffers.pop(request_id)
                        collect_data = {
                            "input_ids": torch.tensor([req['token_ids']], dtype=torch.int32),
                            "trim_ids": torch.tensor([req['prompt_token_ids']], dtype=torch.int32), # FIXME: @gaowei 
                            "padded_trim_ids": ori_request.batch['input_ids'],
                            "trim_attention_mask": ori_request.batch['attention_mask'],
                            "position_ids": ori_request.batch['position_ids'],
                        }
                        meta_info = copy.deepcopy(self.meta_info)
                        gen_config = copy.deepcopy(self.generation_config)
                        gen_config.update({"num_return_sequences":1})
                        meta_info.update({
                            "request_id": request_id, 
                            "generation_config": gen_config,  # key step, update num_return_sequences
                            "max_new_tokens": (self.generation_config['max_new_tokens'] if self.force_max_new_tokens is None else self.force_max_new_tokens) -  has_generated_tokens, # for debug 
                            "response_callback_fn": self.response_callback_fn,
                        })
                        new_req = DataProto.from_single_dict(collect_data, meta_info=meta_info)
                        new_req.non_tensor_batch = ori_request.non_tensor_batch
                        with self.migrate_lock:
                            self.migrate_waiting_reqs.append(new_req)
                    else: 
                        raise NotImplementedError
                self.migrate_completion_ranks[dp_rank] = True 
                # print("length of migrate responses is {}".format(len(self.migrate_waiting_reqs), flush=True))
                print(f"complete rank migration {dp_rank}", flush=True)

                if self.progress_profile: 
                    self.record_time_list.append(
                        {
                            'event_type': 'finish_migration',
                            'duration': time.time() - self.start_time
                        }
                    )

    def reload_balance(self, first_half=False):
        length = len(self.load_balance_coordinator)
        
        filter_keys = sorted(self.load_balance_coordinator.keys())[:length//2] if first_half else sorted(self.load_balance_coordinator.keys())
        new_load_balance_coordinator = {key:self.load_balance_coordinator[key] for key in filter_keys}
        if max(new_load_balance_coordinator.values()) < 4: 
            return 
        while True:
            max_dp_rank = max(new_load_balance_coordinator, key=lambda k: new_load_balance_coordinator[k])
            min_dp_rank = min(new_load_balance_coordinator, key=lambda k: new_load_balance_coordinator[k])
            if new_load_balance_coordinator[max_dp_rank] < 1.4 * new_load_balance_coordinator[min_dp_rank]: 
                break
            max_request_ids, min_request_ids = list(), list()

            for request_id, dp_rank in self.request_id_2_dp_rank.items(): 
                if dp_rank == max_dp_rank: max_request_ids.append(request_id)
            
            assert len(max_request_ids) == new_load_balance_coordinator[max_dp_rank], 'the length should be equal'
            migrate_cnt = (new_load_balance_coordinator[max_dp_rank] - new_load_balance_coordinator[min_dp_rank]) / 2
            for i in range(migrate_cnt):
                request_id = max_request_ids[i]
                self.actor_cluster.workers[max_dp_rank].add_request.remote(
                    command=GenerateRequestType.MIGRATE, data=DataProto(meta_info={"request_id": request_id, "migrate_callback_fn": self.migrate_callback_fn, "dp_rank": -1})
                )
            break
            
        
    @ray.method(concurrency_group="multi_thread")
    def report_response(self, data: DataProto):
        """
        这里需要考虑多线程数据访问
        data 返回可能有多条的
        """
        try:
        # if True: 
            request_id = data.meta_info["request_id"]
            prompt_id = self.request_id_2_prompt_id[request_id]
            num_return_sequences = self.generation_config["num_return_sequences"]

            batch = self.postprocess_output_ids(data)
            output_count = batch.batch.batch_size[0]
            with self.lock:
                self.load_balance_coordinator[self.request_id_2_dp_rank[request_id]] -= 1
                self.prompt_id_2_request_ids[prompt_id].remove(request_id)
                domain = "default"
                if "domain" in batch.non_tensor_batch.keys():
                    domain = batch.non_tensor_batch["domain"][0]
                reward_worker = next(self.reward_worker_iters[domain])

            if not self.running:
                return

            # call reward
            # reward worker得能支持单条数据计算, dynamic sampling对需要batch计算reward的需要注意...
            # 多域的时候,llm as judge, 需要单独为reward worker分配gpu
            reward_start = time.time()
            rewards: DataProto = ray.get(reward_worker.compute_rewards.remote(batch))
            batch.union(rewards)
            reward_time_cost = time.time()-reward_start
            self.reward_worker_costs.append((len(batch), reward_time_cost, rewards.batch['scores'].sum().item()))

            response_buffers: List[DataProto] = []
            batch_expanded = [batch[[idx]] for idx in range(output_count)]

            # response_filter, 不太需要response filter
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
                    return

                if len(self.completed_buffers[prompt_id]) > 0:
                    return

                # expand batch to response
                self.query_group_buffers[prompt_id].extend(response_buffers)

                if self.pre_pass_query_filter_fn is not None and self.dynamic_sampling:
                    if len(self.query_group_buffers[prompt_id]) < num_return_sequences and \
                        prompt_id not in self.pre_pass_query_prompts and \
                        self.pre_pass_query_filter_fn(self.query_group_buffers[prompt_id], self.pipeline_config):
                        self.pre_pass_query_prompts[prompt_id] = True
                        self.pre_pass_prompt_count += 1
                    

                    if self.pre_pass_prompt_count >= self.batch_size and (not self.has_pre_abort_requests):
                        self.has_pre_abort_requests = True
                        # print([rhs_prompt_id not in self.pre_pass_query_prompts for rhs_prompt_id in self.prompt_id_2_request_ids])
                        print('prior early abort load cordinator', self.load_balance_coordinator, flush=True)
                        for rhs_prompt_id in self.prompt_id_2_request_ids:
                            if rhs_prompt_id not in self.pre_pass_query_prompts:
                                self.query_filter_count += 1
                                if rhs_prompt_id in self.query_group_buffers: 
                                    del self.query_group_buffers[rhs_prompt_id]
                                if len(self.prompt_id_2_request_ids[rhs_prompt_id]) > 0:
                                    print("abort rhs prompt id {}, {} requets".format(rhs_prompt_id, len(self.prompt_id_2_request_ids[rhs_prompt_id])))
                                    self.abort_requests(self.prompt_id_2_request_ids[rhs_prompt_id])
                                
                        print("length of self.query_group_buffers is ", len(self.query_group_buffers), flush=True)
                        print('after early abort load cordinator', self.load_balance_coordinator, flush=True)
                        # self.reload_balance()

                # query_filter, query has n responses
                if len(self.query_group_buffers[prompt_id]) >= num_return_sequences:
                    if True: 
                        uncompleted_prompt_ids = dict()
                        for sub_prompt_id in self.prompt_id_2_request_ids.keys(): 
                            if sub_prompt_id not in self.query_group_buffers:
                                uncompleted_prompt_ids[sub_prompt_id] = -1
                            elif len(self.query_group_buffers[sub_prompt_id]) < num_return_sequences: 
                                uncompleted_prompt_ids[sub_prompt_id] = len(self.query_group_buffers[sub_prompt_id])

                        with open('output/prompt_cnt.txt', 'a+') as f:
                            f.write(f"uncompleted prompt ids:\n{uncompleted_prompt_ids}\n")
                            f.write(f"uncompleted prompt ids length is {len(uncompleted_prompt_ids)}\n")
                            f.write(f"number_of_finished_all_requests_for_each_prompt is {self.number_of_finished_all_requests_for_each_prompt+1}\n\n\n")


                    if not self.query_filter_fn(self.query_group_buffers[prompt_id], self.pipeline_config):
                        with open('output/prompt_cnt.txt', 'a+') as f:
                            f.write(f"query filter failed, prompt_id: {prompt_id}, query_group_buffers length: {len(self.query_group_buffers[prompt_id])}\n")
                        self.query_filter_count += 1
                        del self.query_group_buffers[prompt_id]
                        self.abort_requests(self.prompt_id_2_request_ids[prompt_id])
                        return

                    assert len(self.query_group_buffers[prompt_id]) >= num_return_sequences, (
                        f"expect to generate {num_return_sequences} results from one prompt, "
                        f"but get {len(self.query_group_buffers[prompt_id])}."
                    )

                    self.completed_buffers[prompt_id] = self.query_group_buffers[prompt_id][:num_return_sequences]
                    self.progress_bar.update()
                    self.number_of_finished_all_requests_for_each_prompt += 1
                    self.abort_requests(self.prompt_id_2_request_ids[prompt_id])
                    if self.progress_profile and hasattr(self, "record_time_list"):
                        _prompt_response_proto = DataProto.concat(self.query_group_buffers[prompt_id])
                        _response_mask = _prompt_response_proto.batch["response_mask"]
                        _total_lengths = torch.sum(_response_mask, dim=1).cpu().tolist()
                        _prompt_hash = self.prompt_id_2_hash_str.pop(prompt_id)

                        # 记录所有 prompt_id, prompt_hash, domain, reward_time_cost, reward_score, prompt_ids, response_ids
                        prompt_length = batch.batch['prompt_mask'].sum()
                        prompt_ids = batch.batch["prompts"].tolist()[0][-prompt_length:]

                        # _response_length = _total_lengths
                        _response_ids = _prompt_response_proto.batch["responses"].tolist()
                        all_reward_score = _prompt_response_proto.batch['scores'].tolist() if 'scores' in _prompt_response_proto.batch else list()
                        _load_balance_coordinator = {key: value for key, value in self.load_balance_coordinator.items()}
                        
                        self.record_time_list.append(
                            {
                                "prompt_id": prompt_id,
                                "start_time": self.start_time,
                                "end_time": time.time(),
                                "duration": time.time() - self.start_time,
                                "domain": domain,
                                "reward_time_cost": reward_time_cost, 
                                "reward_score": all_reward_score, # rewards.batch['scores'].sum().item(),
                                "prompt_ids": prompt_ids,
                                "response_ids" : _response_ids,
                                "prompt_hash": _prompt_hash, 
                                "dataset_epoch": self.dataset_epoch, 
                                "total_lengths": _total_lengths,
                                "load_balance_coordinator": _load_balance_coordinator,
                            }
                        )
                    
        except Exception as e:
            self.exception_queue.put(e)


    def get_record_time_list(self):
        """
        获取record time list
        """
        if hasattr(self, "record_time_list"):
            return self.record_time_list
        else:
            return None
    
    def get_next_dataset_item(self):
        # get_next_dataset_item 可能消耗时间成本很高
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
        return dataset_item

    def get_scheduler_state(self):
        return {"dataset_iter_count": self.dataset_iter_count}

    def abort_requests(self, request_ids: Set[str]):
        abort_refs = []
        self.running_prompts -= 1
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
        # print('step 1', flush=True)
        eos_token_id = data.meta_info["eos_token_id"]
        pad_token_id = data.meta_info["pad_token_id"]
        output_token_ids = data.meta_info["output_token_ids"]
        if 'trim_ids' in request.batch: 
            assert 'padded_trim_ids' in request.batch
            input_ids = request.batch['input_ids'].tolist() 
            trim_ids = request.batch['trim_ids'].tolist()
            previous_round_output_token_ids = [input_id[len(trim_id):] for trim_id, input_id in zip(trim_ids, input_ids)]
            request.batch['input_ids'] = request.batch['padded_trim_ids']
            request.batch['attention_mask'] = request.batch['trim_attention_mask']
            assert len(output_token_ids) == len(previous_round_output_token_ids), '{} {} mismatch in postprocess_output_ids'.format(len(output_token_ids), len(previous_round_output_token_ids))
            output_tokens = [torch.tensor(prev_token_ids+list(token_ids)) for prev_token_ids, token_ids in zip(previous_round_output_token_ids, output_token_ids)]
        else: 
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
        return output

    def expand_requests(self, data: DataProto):
        """
        replica, 以及redundancy
        """
        generate_opt_level = self.pipeline_config.generate_opt_level
        is_num_return_sequences_expand = self.pipeline_config.is_num_return_sequences_expand
        num_return_sequences = self.generation_config["num_return_sequences"]
        drop_generation_num = self.pipeline_config.drop_generation_num

        assert generate_opt_level > 0, (
            f"generate_opt_level {generate_opt_level} should > 0, " f"in dynamic sampling scheduler."
        )
        assert "generation_config" in data.meta_info, f"data {data.meta_info} should have key 'generation_config'"
        generation_config = data.meta_info["generation_config"]

        target_requests = []
        if is_num_return_sequences_expand:
            generation_config["num_return_sequences"] = 1
            for _ in range(num_return_sequences+drop_generation_num):
                target_requests.append(copy.deepcopy(data))
        else:
            generation_config["num_return_sequences"] = num_return_sequences
            target_requests.append(copy.deepcopy(data))
        return target_requests

    def check_worker_alive(self, cluster, first_half: bool = False):
        # 探测dp worker是否存活，dp worker的server thread可能由于异常退出，造成hang
        current_time = time.time()
        if current_time - self.last_alive_check >= self.alive_check_interval:
            if not first_half:
                cluster.add_request(command=GenerateRequestType.ALIVE_CHECK, data=DataProto())
            else: 
                cluster.add_request_first_half(
                    command=GenerateRequestType.ALIVE_CHECK, data=DataProto(), _ranks=copy.deepcopy(self.pipeline_config.first_half_ranks)
                )
            self.last_alive_check = current_time

    def check_response_callback(self):
        if self.exception_queue.qsize() > 0:
            e = self.exception_queue.get()
            logger.error(f"report_response get exception {e}")
            raise e

    def check_send_new_request(self) -> bool:
        if self.has_pre_abort_requests:
            return False
        if self.running_prompts >= (self.batch_size + self.max_additional_running_prompts):
            return False
        if not self.is_use_additional_prompts and self.prompt_use_count >= self.batch_size:
            return False
        return True

    def get_all_available_dp_ranks(self, first_half: bool = False):
        length = len(self.load_balance_coordinator)
        filter_keys = sorted(self.load_balance_coordinator.keys())[:length//2] if first_half else sorted(self.load_balance_coordinator.keys())
        sorted_ranks = sorted(filter_keys
                , key=lambda rank: (self.load_balance_coordinator[rank], rank)
        )
        return sorted_ranks, [self.max_running_requests - self.load_balance_coordinator[rank] for rank in sorted_ranks]


    def get_available_dp_rank(self, first_half: bool = False):
        length = len(self.mp_rank_zero.keys()) # avoid certain dp_ranks has been pop out of load_balance_coordinator
        filter_keys = sorted(self.load_balance_coordinator.keys())[:length//2] if first_half else sorted(self.load_balance_coordinator.keys())
        while True:
            # 负载均衡逻辑，期望各dp 正在处理的条数基本接近
            sorted_ranks = sorted(filter_keys
                , key=lambda rank: (self.load_balance_coordinator[rank], rank)
            )
            if self.load_balance_coordinator[sorted_ranks[0]] < self.max_running_requests:
                yield sorted_ranks[0], self.max_running_requests - self.load_balance_coordinator[sorted_ranks[0]]


@ray.remote
class GlobalCounter:
    def __init__(self):
        self.value = -1

    def get_value(self):
        self.value += 1
        return self.value
