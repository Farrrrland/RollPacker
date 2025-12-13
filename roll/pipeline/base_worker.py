import os
import threading
import time
from typing import Union, Optional, Dict, List
import itertools
import copy

import ray
import torch
from codetiming import Timer
from tqdm import tqdm
import numpy as np 

from ray.actor import ActorHandle

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import register, Dispatch, Execute
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_actor_model_provider, default_value_model_provider, \
    default_reward_model_provider
from roll.utils.checkpoint_manager import download_model
from roll.utils.context_managers import state_offload_manger
from roll.utils.functionals import (
    append_to_dict,
    masked_mean,
    compute_approx_kl,
    postprocess_generate,
    GenerateRequestType,
    agg_loss,
)
from roll.utils.offload_states import OffloadStateType

from roll.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx

from roll.utils.functionals import (
    compute_advantage,
    reduce_metrics,
    RunningMoments,
    get_sample_level_mask,
    reward_postprocess,
    compute_token_reward,
    agg_loss,
    compute_gae_advantage_return, 
    compute_reinforce_return,
    compute_clip_fraction,
    masked_var
)
from roll.utils.kl_controller import FixedKLController, AdaptiveKLController
from roll.pipeline.rlvr.rlvr_config import RLVRConfig


def fill_log_probs(batch_data): 
    # print(f"[DATA CHECK] {batch_data}")
    # print(f"[DATA CHECK II] {batch_data.batch}")
    tok_len = batch_data.batch['response_mask'].size(1)
    batch_data.batch["old_log_probs"] = torch.randn((batch_data.batch.size(0), tok_len-1))
    batch_data.batch['ref_log_probs'] = batch_data.batch["old_log_probs"]
    return batch_data


def native_masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    # import pdb; pdb.set_trace()
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

@torch.no_grad()
def native_compute_advantage(
    data: "DataProto",
    gamma,
    lambd,
    adv_estimator,
    advantage_clip=None,
    whiten_advantages=False,
    whiten_rewards=False,
    response_mask=None,
):
    if response_mask is None:
        response_mask = data.batch["response_mask"][:, 1:]

    token_level_rewards = data.batch["token_level_rewards"].float()
    if whiten_rewards:
        token_level_rewards = native_masked_whiten(values=token_level_rewards, mask=response_mask)
    token_level_rewards = token_level_rewards * response_mask
    data.batch["token_level_rewards"] = token_level_rewards
    if adv_estimator == "gae":
        values = data.batch["values"].float()
        data.batch["values"] = values * response_mask
        advantages, returns = compute_gae_advantage_return(
            token_level_rewards=token_level_rewards, values=values, gamma=gamma, lambd=lambd
        )
    elif adv_estimator == "reinforce":
        advantages, returns = compute_reinforce_return(
            token_level_rewards=token_level_rewards, gamma=gamma, lambd=lambd
        )
    elif adv_estimator == "grpo":
        advantages, returns = compute_reinforce_return(
            token_level_rewards=token_level_rewards, gamma=gamma, lambd=lambd
        )
    elif adv_estimator == "gigpo":
        advantages, returns = compute_reinforce_return(
            token_level_rewards=token_level_rewards, gamma=gamma, lambd=lambd
        )
    else:
        raise NotImplementedError

    data.batch["raw_advantages"] = advantages
    if whiten_advantages:
        # TODO whiten过程中是否要考虑response的长度？
        advantages = native_masked_whiten(values=advantages, mask=response_mask)
    advantages = advantages * response_mask

    if advantage_clip is not None:
        adv_clip_frac = compute_clip_fraction(values=advantages, clip_min=-advantage_clip, clip_max=advantage_clip)
        data.meta_info["metrics"] = {"critic/advantage_clip_frac": adv_clip_frac}
        advantages = torch.clamp(advantages, min=-advantage_clip, max=advantage_clip)

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


def local_compute_advantage(prefetch_domain_batches: DataProto, pipeline_config: RLVRConfig, running, kl_ctrl): 
    # prefetch_domain_batches.batch["prompt_id"] = torch.arange(prefetch_domain_batches.batch.batch_size[0], device=prefetch_domain_batches.batch.device)
    batch_grouped: Dict[str, DataProto] = prefetch_domain_batches.group_by("domain")
    
    batch_list = []

    for domain, domain_batch in batch_grouped.items():
        # print("domain_batch is {}".format(domain_batch.batch), flush=True)
        domain_batch, mask_metrics = get_sample_level_mask(domain_batch, pipeline_config)
        
        # tmp_disabled_update = running[domain].disabed_update
        # running[domain].disabed_update = True
        domain_batch, response_level_metrics = reward_postprocess(
            domain_batch, pipeline_config, running[domain]
        )
        # running[domain].disabed_update = tmp_disabled_update
    
        domain_batch, token_level_metrics = compute_token_reward(
            domain_batch, pipeline_config, kl_ctrl
        )
        final_response_mask = domain_batch.batch["final_response_mask"].clone()
    
        domain_batch = compute_advantage(
            data=domain_batch,
            gamma=pipeline_config.gamma,
            lambd=pipeline_config.lambd,
            adv_estimator=pipeline_config.adv_estimator,
            advantage_clip=pipeline_config.advantage_clip,
            whiten_advantages=pipeline_config.whiten_advantages,
            whiten_rewards=pipeline_config.whiten_rewards,
            response_mask=final_response_mask,
        )
        batch_list.append(domain_batch)
        # gamma is 1, lambd is 0.95, adv_estimator is reinforce, withen_advantages is True, whiten_rewards is False  
        # kl_ctrl is <roll.utils.kl_controller.FixedKLController object at 0x1521640e6cb0>, running {'math_rule': <roll.utils.functionals.RunningMoments object at 0x1521640e6c50>} 



    # prefetch_domain_batches.pop("prompt_id")
    return prefetch_domain_batches



def batch_compute_rewards(batch: DataProto, pipeline_config, running, kl_ctrl):
    batch_grouped = batch.group_by('domain')
    batch_list = list()
    
    for domain, domain_batch in batch_grouped.items():
        # 1. 处理mask相关策略， 获取sample level mask
        domain_batch, _ = get_sample_level_mask(domain_batch, pipeline_config)

        # 2. 处理reward相关策略
        domain_batch, __cached__ = reward_postprocess(
            domain_batch, pipeline_config, running,
        )

        # 3. 计算token level rewards
        domain_batch, _ = compute_token_reward(
            domain_batch, pipeline_config, kl_ctrl
        )

        # 4. 计算advantage
        final_response_mask = domain_batch.batch["final_response_mask"].clone()
        domain_batch = native_compute_advantage(
            data=domain_batch,
            gamma=pipeline_config.gamma,
            lambd=pipeline_config.lambd,
            adv_estimator=pipeline_config.adv_estimator,
            advantage_clip=pipeline_config.advantage_clip,
            whiten_advantages=pipeline_config.whiten_advantages,
            whiten_rewards=pipeline_config.whiten_rewards,
            response_mask=final_response_mask,
        )
        batch_list.append(domain_batch)
    batch_with_rewards = DataProto.concat(batch_list)
    batch_with_rewards.reorder(indices=torch.argsort(batch_with_rewards.batch["prompt_id"]))
    return batch_with_rewards


class ActorWorker(Worker):
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None
        self.response_call_back_fns = {}
        self.response_callback_refs = []
        self.server_metrics = {}
        self.thread_server = None
        self.offload_manager = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        self.strategy.initialize(model_provider=default_actor_model_provider)
        self.tokenizer = self.strategy.tokenizer
        if self.pipeline_config.resume_from_checkpoint:
            load_dir = download_model(self.pipeline_config.resume_from_checkpoint)
            self.strategy.load_checkpoint(load_dir=load_dir, tag="checkpoint")
        self.logger.info(f"{self.worker_name} initialized")

        self.strategy.offload_states()

        # Cuda must have been initialized when calling torch.cuda.reset_max_memory_allocated
        # with arguments (inside state_offload_manager). We explicitly init cuda here because
        # current process is used as engine client when using vllm v1 engine, and
        # there is no chance to init cuda context.
        torch.cuda.init()


    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def train_step(self, data: DataProto):
        """
        return DataProto(meta_info={'metrics': metrics})
        """

        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        self.logger.info(f"{self.worker_name} generate global step {global_step}")

        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/train_step",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params, OffloadStateType.other_params]},
        ):
            data = data.to("cuda")

            # rank = torch.distributed.get_rank()
            # print("before get_batch data.batch {}, rank {}".format(data.batch, rank))
            data = self.strategy.get_data_input(data)
            # rank = torch.distributed.get_rank()
            # print("after get_batch data.batch {}, rank {}".format(data.batch, rank))
            torch.distributed.barrier()
            per_device_train_batch_size = self.worker_config.training_args.per_device_train_batch_size
            backward_batch_size = (
                per_device_train_batch_size * self.worker_config.training_args.gradient_accumulation_steps
            )
            backward_batch_size = data.batch.batch_size[0]
            dataloader = data.make_iterator(
                mini_batch_size=backward_batch_size,
                epochs=self.pipeline_config.ppo_epochs,
                seed=self.pipeline_config.seed,
                dataloader_kwargs={"shuffle": True},
            )

            for batch_idx, data in tqdm(
                enumerate(dataloader),
                desc=f"{self.worker_name} train global step {global_step}",
                total=data.batch.batch_size[0] * self.pipeline_config.ppo_epochs // backward_batch_size,
            ):
                pg_metrics = self.strategy.train_step(batch=data, loss_func=self.loss_func)
                append_to_dict(metrics, pg_metrics)

            metrics["actor/lr"] = self.strategy.scheduler.get_last_lr()[0]
            data.to("cpu")
            if 'actor_train/grad_norm' in metrics: 
                print("metrics in train_step {}".format(metrics['actor_train/grad_norm']), flush=True)
            # import pdb; pdb.set_trace()

        output = DataProto(meta_info={"metrics": metrics})
        return output



    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def train_step_full(self, data: DataProto):
        """
        return DataProto(meta_info={'metrics': metrics})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        self.logger.info(f"{self.worker_name} generate global step {global_step}")

        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/train_step",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params, OffloadStateType.other_params]},
        ):
            data = data.to("cuda")
            data = self.strategy.get_data_input(data)
            assert self.pipeline_config.ppo_epochs == 1, (
                f"ppo_epochs={self.pipeline_config.ppo_epochs} must be 1 for train_step_full."
            )
            backward_batch_size = data.batch.batch_size[0]
            dataloader = data.make_iterator(
                mini_batch_size=backward_batch_size,
                epochs=self.pipeline_config.ppo_epochs,
                seed=self.pipeline_config.seed,
                dataloader_kwargs={"shuffle": True},
            )
            for batch_idx, data in tqdm(
                enumerate(dataloader),
                desc=f"{self.worker_name} train global step {global_step}",
                total=data.batch.batch_size[0] * self.pipeline_config.ppo_epochs // backward_batch_size,
            ):
                pg_metrics = self.strategy.train_step(batch=data, loss_func=self.loss_func, no_sync=False, in_no_sync_train=True)
                append_to_dict(metrics, pg_metrics)
                if 'actor_train/grad_norm' in metrics: 
                    print("metrics in train_step_full {}".format(metrics['actor_train/grad_norm']), flush=True)

            metrics["actor/lr"] = self.strategy.scheduler.get_last_lr()[0]
            data.to("cpu")

        output = DataProto(meta_info={"metrics": metrics})
        return output


    @register(dispatch_mode=Dispatch.ONE_TO_HALF, execute_mode=Execute.SECOND_HALF)
    def train_step_second_half_with_func(self, meta_data: DataProto, prefetch_actor: ActorHandle, running: Dict[str, RunningMoments], kl_ctrl: Union[AdaptiveKLController, FixedKLController], _ranks: List[int]):
        """
        return DataProto(meta_info={'metrics': metrics})
        """
        denominator_in_world_size = self.strategy.worker.world_size // len(_ranks)
        global_step = meta_data.meta_info.get("global_step", 0)
        is_offload_states = meta_data.meta_info.get("is_offload_states", True)
        metrics = {}
        self.logger.info(f"{self.worker_name} generate global step {global_step} at second half")

        per_device_train_batch_size = self.worker_config.training_args.per_device_train_batch_size
        pg_world_size = self.strategy.worker.rank_info.dp_size // denominator_in_world_size
        pg_prompt_count = 2 * per_device_train_batch_size * pg_world_size // self.pipeline_config.num_return_sequences_in_group # assume it as FSDP strategy
        
        os.environ['ddp_no_async'] = "True"
        if not hasattr(self, 'subgroup'):
            if False: 
                ranks = [i for i in range(self.strategy.worker.rank_info.dp_size)]
                half_size = len(ranks) // 2
                selected_ranks = ranks[half_size:]
            else: 
                selected_ranks = _ranks
            self.subgroup = torch.distributed.new_group(selected_ranks) # 
            self.recv_data_rank = selected_ranks[0]

        prefetch_prompt_count = -1
        ori_gradient_accumulation_steps = self.worker_config.training_args.gradient_accumulation_steps
        record_time_list = list() 

        step = 0
        record_time_list.append(
            {
                'event_type': f"start_inside_step_train_second_half_with_func", 
                'absolute_time': time.time(),
            }
        )
        index = 0
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/train_step_second_half",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params, OffloadStateType.other_params]},
        ):
            while True:
                record_time_list.append(
                    {
                        'event_type': f"start_inside_fetch_iter", 
                        'absolute_time': time.time(),
                        "step": step,
                    }
                )
                batch_data = None
                if self.rank == self.recv_data_rank:
                    try:
                        if self.pipeline_config.fixed_async:
                            # Fixed async prefetch behavior
                            batch_data = ray.get(
                                prefetch_actor.one_shot_prefetch_completed_requests.remote(
                                    prefetch_prompt_count=prefetch_prompt_count,
                                    div_multipler=pg_prompt_count,
                                    running=running,
                                    kl_ctrl=kl_ctrl,
                                ),
                                timeout=2
                            )
                        else: 
                            batch_data = ray.get(
                                prefetch_actor.prefetch_completed_requests.remote(
                                    prefetch_prompt_count=prefetch_prompt_count,
                                    div_multipler=pg_prompt_count,
                                    running=running,
                                    kl_ctrl=kl_ctrl,
                                ),
                                timeout=2
                            )
                    except ray.exceptions.GetTimeoutError:
                        print("Timed out waiting for prefetch_completed_requests to return.")
                        batch_data = None  # 明确赋值为 None

                # Step 1: Broadcast whether batch_data is valid
                batch_available_state = 2  # 默认：无数据
                if self.rank == self.recv_data_rank:
                    if batch_data is not None:
                        # print("batch_data.meta_info is {}".format(batch_data.meta_info), flush=True)
                        # Check meta_info flag
                        if 'disable_preftch_requests' in batch_data.meta_info:
                            disable_flag = batch_data.meta_info.pop('disable_preftch_requests')
                            if disable_flag:
                                batch_available_state = 2  # exit train_step_second_half_with_func
                            else:
                                batch_available_state = 1  # continue
                        else:
                            batch_available_state = 0  # proceed normally

                # 同步 batch_available_state 到所有进程
                batch_available_state_tensor = torch.tensor([batch_available_state]).cuda()
                if self.subgroup is not None and self.subgroup.size() > 1:
                    torch.distributed.broadcast(batch_available_state_tensor, src=self.recv_data_rank, group=self.subgroup)
                batch_available_state = batch_available_state_tensor.item()
                print(f"\t batch_available_state is {batch_available_state}", flush=True)
                if batch_available_state == 1: # continue to fetch because of no enough samples in last round
                    time.sleep(0.5)
                    continue
                elif batch_available_state == 2: # the generate scheduler notifies that you should exit train_second_half function
                    break

                # Step 2: Prepare batch_data_list only when we are sure data is valid
                if self.rank == self.recv_data_rank:
                    if 'advantages' in batch_data.batch: 
                        pass 
                    elif 'indices' in batch_data.meta_info: 
                        print(f"recv batch_data.batch {len(batch_data.batch)} in train_step_second_half_with_func.", flush=True)
                        indices = batch_data.meta_info['indices']
                        batch = batch_data
                        # start compute relative advantages

                        full_batch_size = self.pipeline_config.rollout_batch_size * self.pipeline_config.num_return_sequences_in_group
                        full_batch = batch
                        if True:
                            batch = full_batch
                            batch.batch["prompt_id"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
                            tok_len = batch.batch['response_mask'].size(1)
                            batch.batch["old_log_probs"] = torch.randn((batch.batch.size(0), tok_len-1))
                            batch.batch['ref_log_probs'] = batch.batch["old_log_probs"]
                            start = time.time()
                            batch_with_rewards = batch_compute_rewards(batch=batch, pipeline_config=self.pipeline_config, running=running, kl_ctrl=kl_ctrl)
                            # import pdb; pdb.set_trace()
                            shuffiled_indices = np.random.permutation(indices)
                            # batch_data = batch_with_rewards.select_idxs(indices)
                            batch_data = batch_with_rewards.select_idxs(shuffiled_indices)
                    else:
                        batch_data = fill_log_probs(batch_data=batch_data)
                        batch_data = local_compute_advantage(batch_data, pipeline_config=self.pipeline_config, running=copy.deepcopy(running), kl_ctrl=kl_ctrl)
                    
                if 'megatron' in self.strategy.strategy_name:
                    if pg_world_size == 1:
                        batch_data_list = [batch_data] if batch_data is not None else [None]
                    else:
                        batch_data_list = batch_data.chunk(pg_world_size) if batch_data is not None else [None] * pg_world_size
                    torch.distributed.broadcast_object_list(batch_data_list, src=self.recv_data_rank, group=self.subgroup)
                    local_dp_rank_in_group = self.strategy.worker.rank_info.dp_rank - pg_world_size
                    # data = batch_data_list[local_dp_rank_in_group]
                    data = batch_data_list[local_dp_rank_in_group]
                    rank_info = self.strategy.worker.rank_info
                    # import pdb; pdb.set_trace()
                    # batch_data_list[0].batch['advantages'][1:, :].fill_(0.)
                    # batch_data_list[1].batch['advantages'][1:, :].fill_(0)
                    if not (rank_info.tp_rank == 0 and rank_info.cp_rank == 0 and rank_info.pp_rank == 0):
                        data = DataProto(batch=None, meta_info=data.meta_info)
                elif 'fsdp' in self.strategy.strategy_name:
                    # Step 3: Broadcast the list of objects
                    if self.rank == self.recv_data_rank:
                        batch_data_list = batch_data.chunk(pg_world_size)
                    else:
                        batch_data_list = [None] * pg_world_size  # 占位符

                    torch.distributed.broadcast_object_list(batch_data_list, src=self.recv_data_rank, group=self.subgroup)
                    local_dp_rank_in_group = torch.distributed.get_rank(group=self.subgroup)
                    print("local_rank in group is {}".format(local_dp_rank_in_group), flush=True)
                    print("batch_data_list length is {}".format(len(batch_data_list)), flush=True)
                    data = batch_data_list[local_dp_rank_in_group]
                else: 
                    raise NotImplementedError
                # print(local_dp_rank_in_group, torch.distributed.get_rank())
                # Step 4: Get local batch
                if data is None: 
                    import pdb; pdb.set_trace()
                torch.distributed.barrier(self.subgroup) 


                record_time_list.append(
                    {
                        'event_type': f"start_inside_train_iter", 
                        'absolute_time': time.time(),
                        "batch_size": data.batch.size(0) if data.batch is not None else 0,
                        "step": step,
                    }
                )

                data = data.to("cuda")
                data = self.strategy.get_data_input(data)

                per_device_train_batch_size = self.worker_config.training_args.per_device_train_batch_size
                
                backward_batch_size = data.batch.batch_size[0]
                self.worker_config.training_args.gradient_accumulation_steps = backward_batch_size // per_device_train_batch_size

                dataloader = data.make_iterator(
                    mini_batch_size=backward_batch_size,
                    epochs=self.pipeline_config.ppo_epochs,
                    seed=self.pipeline_config.seed,
                    dataloader_kwargs={"shuffle": True},
                )
                print("starting at second half", flush=True)
                for batch_idx, sub_data in tqdm(
                    enumerate(dataloader),
                    desc=f"{self.worker_name} train global step {global_step} at second half",
                    total=data.batch.batch_size[0] * self.pipeline_config.ppo_epochs // backward_batch_size,
                ):
                    pg_metrics = self.strategy.train_step(batch=sub_data, loss_func=self.loss_func, no_sync=True, in_no_sync_train=True)
                torch.distributed.barrier(self.subgroup)     

                prefetch_prompt_count = pg_prompt_count

                record_time_list.append(
                    {
                        'event_type': f"finish_inside_train_iter", 
                        'absolute_time': time.time(),
                        "batch_size": data.batch.size(0),
                        "step": step, 
                    }
                )
                step += 1
                data = None
                # break
            print('start destroy_subgroup', flush=True)
            self.destroy_subgroup()
            print('compl destroy_subgroup', flush=True)    


        self.worker_config.training_args.gradient_accumulation_steps = ori_gradient_accumulation_steps
        if False: 
            # import pdb; pdb.set_trace() 
            self.strategy.offload_states([OffloadStateType.model_params, OffloadStateType.other_params])
            for handle in self.model._all_handles:
                flat_param = handle.flat_param
                print(f"offload_fsdp_model_gradients_to_cpu {flat_param.grad is not None} {flat_param.grad.device}")
            self.strategy.load_states([OffloadStateType.model_params, OffloadStateType.other_params])
            
            for handle in self.model._all_handles:
                flat_param = handle.flat_param
                print(f"offload_fsdp_model_gradients_to_cpu {flat_param.grad is not None} {flat_param.grad.device}")
        
        output = DataProto(meta_info={"metrics": metrics})

        if self.pipeline_config.record_time_profiler_log_dir is not None and self.rank == self.recv_data_rank:
            profiler_log_dir = self.pipeline_config.record_time_profiler_log_dir
            if not os.path.exists(profiler_log_dir):
                os.makedirs(profiler_log_dir)
            import json
            with open(f'{profiler_log_dir}/train_second_iter_{global_step}.json', 'w') as fd:
                json.dump(record_time_list, fd)
        
        os.environ['ddp_no_async'] = "False"
        return output

    def sync_partial_gradients(self):
        world_size = torch.distributed.get_world_size(group=self.subgroup)
        for handle in self.strategy.model._all_handles:
            flat_param = handle.flat_param
            if flat_param.grad is not None:
                print(f"Syncing gradients for {flat_param.name} on rank {self.rank}", flush=True)
                torch.distributed.all_reduce(flat_param.grad, op=torch.distributed.ReduceOp.SUM, group=self.subgroup)
                flat_param.grad /= world_size  # Average the gradients
            else:
                print(f"No gradients to sync for {flat_param.name} on rank {self.rank}", flush=True)


    def destroy_subgroup(self):
        if getattr(self, 'subgroup', None) is not None:
            try:
                torch.distributed.destroy_process_group(self.subgroup)
            except Exception as e:
                print(f"Warning: Failed to destroy subgroup: {e}")
            finally:
                delattr(self, 'subgroup')

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE_SECOND_HALF, execute_mode=Execute.SECOND_HALF)
    def train_step_second_half(self, data: DataProto, _ranks=List[int]):
        """
        return DataProto(meta_info={'metrics': metrics})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        self.logger.info(f"{self.worker_name} generate global step {global_step} at second half")
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/train_step_second_half",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params, OffloadStateType.other_params]},
        ):
            data = data.to("cuda")
            data = self.strategy.get_data_input(data)
            per_device_train_batch_size = self.worker_config.training_args.per_device_train_batch_size
            # backward_batch_size = (
            #     per_device_train_batch_size * self.worker_config.training_args.gradient_accumulation_steps
            # )
            ori_gradient_accumulation_steps = self.worker_config.training_args.gradient_accumulation_steps
            backward_batch_size = data.batch.batch_size[0]
            self.worker_config.training_args.gradient_accumulation_steps = backward_batch_size // per_device_train_batch_size

            dataloader = data.make_iterator(
                mini_batch_size=backward_batch_size,
                epochs=self.pipeline_config.ppo_epochs,
                seed=self.pipeline_config.seed,
                dataloader_kwargs={"shuffle": True},
            )

            for batch_idx, data in tqdm(
                enumerate(dataloader),
                desc=f"{self.worker_name} train global step {global_step} at second half",
                total=data.batch.batch_size[0] * self.pipeline_config.ppo_epochs // backward_batch_size,
            ):
                pg_metrics = self.strategy.train_step(batch=data, loss_func=self.loss_func, no_sync=True, in_no_sync_train=True)
                append_to_dict(metrics, pg_metrics)

            metrics["actor/lr"] = self.strategy.scheduler.get_last_lr()[0]
            data.to("cpu")
            self.worker_config.training_args.gradient_accumulation_steps = ori_gradient_accumulation_steps

        
        if False: 
            # import pdb; pdb.set_trace() 
            self.strategy.offload_states([OffloadStateType.model_params, OffloadStateType.other_params])
            for handle in self.model._all_handles:
                flat_param = handle.flat_param
                print(f"offload_fsdp_model_gradients_to_cpu {flat_param.grad is not None} {flat_param.grad.device}")
            self.strategy.load_states([OffloadStateType.model_params, OffloadStateType.other_params])
            
            for handle in self.model._all_handles:
                flat_param = handle.flat_param
                print(f"offload_fsdp_model_gradients_to_cpu {flat_param.grad is not None} {flat_param.grad.device}")
        
        output = DataProto(meta_info={"metrics": metrics})
        return output


    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def generate(self, data: DataProto):
        """
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'old_log_probs': log_probs,
            },
            batch_size=batch_size)
        return DataProto(batch=batch)
        """
        if "generation_config" not in data.meta_info:
            generation_config = self.worker_config.generating_args.to_dict()
        else:
            generation_config = data.meta_info["generation_config"]

        generation_config["eos_token_id"] = [
            self.tokenizer.eos_token_id
        ] + self.tokenizer.additional_special_tokens_ids
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id

        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        self.logger.info(f"{self.worker_name} generate global step {global_step}")

        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/generate",
            is_offload_states=is_offload_states,
        ):
            data = data.to("cuda")
            # import torch
            if torch.distributed.is_initialized():
                print('==== '* 10 + 'dist get rank is {}, {}'.format(torch.distributed.get_rank(), self.worker_config.infer_batch_size), flush=True)
            print(data.batch)
            # exit(0)
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
            # import pdb; pdb.set_trace() 
            output = self.strategy.generate(batch=data, generation_config=generation_config)
            output = postprocess_generate(
                prompts=data,
                output=output,
                num_return_sequences=generation_config["num_return_sequences"],
                sequence_length=self.pipeline_config.sequence_length,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            data.to("cpu")
            output = output.to("cpu")

        output.meta_info = {"metrics": metrics}
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL_ONE)
    @torch.no_grad()
    def start_server(self, data: DataProto):
        """
        解决dp generate的长尾问题，async+ load balance
        """
        if self.thread_server is not None:
            return

        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)

        self.logger.info(f"{self.worker_name} generate server global step {global_step}")
        self.response_call_back_fns = {}

        self.response_callback_refs = []
        self.server_metrics = {}
        self.offload_manager = state_offload_manger(
            strategy=self.strategy,
            metrics=self.server_metrics,
            metric_infix=f"{self.cluster_name}/generate",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params]},
        )
        self.offload_manager.__enter__()
        self.thread_server = threading.Thread(
            target=self.strategy.start_server, kwargs=dict(data=data, request_complete_callback=self.request_complete)
        )
        self.thread_server.start()
        while not self.strategy.running:
            time.sleep(0.1)
        
        print("thread server start", flush=True)
        print(self.thread_server, flush=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL_ONE)
    def stop_server(self, data: DataProto = None):
        if not hasattr(self, "thread_server"):
            raise ValueError("server is not initialized")
        if self.thread_server is None: 
            return DataProto(meta_info={"metrics": self.server_metrics})

        self.strategy.add_request(command=GenerateRequestType.STOP, data=data)
        self.thread_server.join()
        self.thread_server = None
        self.response_call_back_fns.clear()
        self.offload_manager.__exit__(None, None, None)
        ray.get(self.response_callback_refs)
        self.response_callback_refs.clear()

        return DataProto(meta_info={"metrics": self.server_metrics})


    @register(dispatch_mode=Dispatch.ONE_TO_ALL_ONE_SECOND_HALF, execute_mode=Execute.SECOND_HALF)
    def stop_server_second_half(self, data: DataProto = None, _ranks=List[int]):
        if not hasattr(self, "thread_server"):
            raise ValueError("server is not initialized")

        print("start concrete stop_server_second_half {}".format(self.rank_info), flush=True)
        self.strategy.add_request(command=GenerateRequestType.STOP, data=data)
        self.thread_server.join()
        self.thread_server = None
        self.response_call_back_fns.clear()
        self.offload_manager.__exit__(None, None, None)
        ray.get(self.response_callback_refs)
        self.response_callback_refs.clear()
        print("finish concrete stop_server_second_half {}".format(self.rank_info), flush=True)
        return DataProto(meta_info={"metrics": self.server_metrics})
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL_ONE_FIRST_HALF, execute_mode=Execute.FIRST_HALF)
    def stop_server_first_half(self, data: DataProto = None, _ranks=List[int]):
        if not hasattr(self, "thread_server"):
            raise ValueError("server is not initialized")

        self.strategy.add_request(command=GenerateRequestType.STOP, data=None)
        self.thread_server.join()
        self.thread_server = None
        self.response_call_back_fns.clear()
        self.offload_manager.__exit__(None, None, None)
        ray.get(self.response_callback_refs)
        self.response_callback_refs.clear()

        return DataProto(meta_info={"metrics": self.server_metrics})

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def compute_log_probs(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'log_probs': output})
        """
        data = self.strategy.get_data_input(data)
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        use_dynamic_bsz = data.meta_info.get('use_dynamic_bsz', False)
        use_dynamic_bsz = False 
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_log_probs",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            data = data.to("cuda")

            if 'micro_batch_size' not in data.meta_info: 
                data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size # FIXME
            if use_dynamic_bsz: 
                ppo_max_token_len_per_gpu = 16384
                
                # micro_batches, _ = rearrange_micro_batches(batch=data.batch, max_token_len=ppo_max_token_len_per_gpu)
                micro_batches = data
                # import pdb; pdb.set_trace() 
                with torch.no_grad():
                    results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                        batch=micro_batches, forward_func=self.forward_func_log_probs
                    )
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                log_probs = log_probs[revert_indices]
            else: 
                with torch.no_grad():
                    results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                        batch=data, forward_func=self.forward_func_log_probs
                    )

            if results is None:
                return DataProto(batch=None, meta_info={"metrics": metrics})
            output = DataProto.from_dict(tensors={"log_probs": results["log_probs"], "entropy": results["entropy"]})
            output = output.to("cpu")
            data.to("cpu")
        output.meta_info = {"metrics": metrics}
        # self.strategy.offload_states(include=[OffloadStateType.model_params]) # TODO: see how deepspeed address it
        return output


    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST_SECOND_HALF, execute_mode=Execute.SECOND_HALF)
    def compute_log_probs_second_half(self, data: DataProto, _ranks=List[int]):
        """
        return DataProto.from_dict(tensors={'log_probs': output})
        """
        data = self.strategy.get_data_input(data)
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        print("start compute_log_probs_second_half step 1", flush=True)
        
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_log_probs",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            print("start offload compute_log_probs_second_half step 2", flush=True)
            data = data.to("cuda")
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
            
            with torch.no_grad():
                results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                    batch=data, forward_func=self.forward_func_log_probs
                )
            if results is None:
                return DataProto(batch=None, meta_info={"metrics": metrics})
            print("start offload compute_log_probs_second_half step 3", flush=True)
            output = DataProto.from_dict(tensors={"log_probs": results["log_probs"], "entropy": results["entropy"]})
            output = output.to("cpu")
            data.to("cpu")
        output.meta_info = {"metrics": metrics}

        print("complete compute_log_probs_second_half", flush=True)
        return output

    def forward_func_log_probs(self, data: DataProto, output_tensor: torch.Tensor):
        """
        forward func 接口定义:
            data: DataProto, 由forward_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
        )
        entropy = self.strategy.op_compute_entropy(logits=output_tensor, attention_mask=data.batch["response_mask"])
        return log_probs, {"log_probs": log_probs.clone().detach(), "entropy": entropy.clone().detach()}

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        try:

            print(f"LOSS FUNC AAAAA")

            response_mask = data.batch["response_mask"][:, 1:].long()
            ref_log_probs = data.batch["ref_log_probs"]
            old_log_probs = data.batch["old_log_probs"]
            advantages = data.batch["advantages"]

            log_probs = self.strategy.op_compute_log_probs(
                logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
            )

            print(f"LOSS FUNC BBBBB")

            ratio = (log_probs - old_log_probs).exp()

            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.pipeline_config.pg_clip, 1 + self.pipeline_config.pg_clip) * advantages
            pg_loss = -torch.min(surr1, surr2)
            if self.pipeline_config.dual_clip_loss:
                dual_clip_loss = -torch.max(-pg_loss, (1 + self.pipeline_config.pg_clip * 2) * advantages)
                pg_loss = torch.where(advantages < 0, dual_clip_loss, pg_loss)

            pg_loss = agg_loss(loss_mat=pg_loss, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode)

            kl_loss = compute_approx_kl(log_probs=log_probs, log_probs_base=ref_log_probs, action_mask=response_mask,
                                        kl_penalty="k3")
            kl_loss = agg_loss(loss_mat=kl_loss, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode)

            approxkl = compute_approx_kl(
                log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="mse"
            )
            policykl = compute_approx_kl(
                log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="kl"
            )

            print(f"LOSS FUNC CCCCC")

            clipped_low = (ratio < 1 - self.pipeline_config.pg_clip).float()
            clipped_high = (ratio > 1 + self.pipeline_config.pg_clip).float()
            clipped = (clipped_low + clipped_high).float()

            entropy = self.strategy.op_compute_entropy(logits=output_tensor, attention_mask=data.batch["response_mask"])
            entropy_loss = agg_loss(
                loss_mat=entropy,
                loss_mask=response_mask,
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
            )

            if self.pipeline_config.use_kl_loss:
                total_loss = pg_loss + kl_loss * self.pipeline_config.kl_loss_coef
            else:
                total_loss = pg_loss
            if self.pipeline_config.entropy_loss_coef > 0:
                total_loss = total_loss - entropy_loss * self.pipeline_config.entropy_loss_coef
            
            print(f"LOSS FUNC DDDDD")

            pg_metrics = {
                "actor/ppo_ratio_high_clipfrac": clipped_high.mean().detach().item(),
                "actor/ppo_ratio_low_clipfrac": clipped_low.mean().detach().item(),
                "actor/ppo_ratio_clipfrac": clipped.mean().detach().item(),
                "actor/ratio_mean": masked_mean(ratio, response_mask, dim=-1).mean().detach().item(),
                "actor/ratio_max": torch.max(ratio * response_mask).detach().item(),
                "actor/ratio_min": torch.min(ratio * response_mask + (1 - response_mask) * 1e10).detach().item(),
                "actor/clipfrac": agg_loss(loss_mat=torch.lt(surr2, surr1).float(), loss_mask=response_mask,
                                        loss_agg_mode=self.pipeline_config.loss_agg_mode).detach().item(),
                "actor/pg_loss": pg_loss.detach().item(),
                "actor/kl_loss": kl_loss.detach().item(),
                "actor/total_loss": total_loss.detach().item(),
                "actor/approxkl": agg_loss(loss_mat=approxkl, loss_mask=response_mask,
                                        loss_agg_mode=self.pipeline_config.loss_agg_mode).detach().item(),
                "actor/policykl": agg_loss(loss_mat=policykl, loss_mask=response_mask,
                                        loss_agg_mode=self.pipeline_config.loss_agg_mode).detach().item(),
            }
        except Exception as e:
            print(e)
            import traceback; traceback.print_exc()

        return total_loss, pg_metrics

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def do_checkpoint(self, global_step):
        with Timer("do_checkpoint") as total_timer:
            ckpt_id = f"checkpoint-{global_step}"

            # actor train是直接存在save dir目录下的，其他role是存在save_dir/cluster_name下的
            save_dir = os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id)
            self.logger.info(f"save checkpoint-{global_step} to {save_dir}")

            exec_metrics: Dict = self.strategy.save_checkpoint(save_dir, global_step, ckpt_id)

        metrics = {
            f"time/{self.cluster_name}/do_checkpoint/total": total_timer.last,
        }
        metric_prefix = f"time/{self.cluster_name}/do_checkpoint"
        metrics.update({f"{metric_prefix}/{k}": v for k, v in exec_metrics.items()})
        output = DataProto(meta_info={"metrics": metrics})
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_FIRST_HALF, execute_mode=Execute.FIRST_HALF, clear_cache=False)
    def add_request_first_half(self, command, data: DataProto, _ranks=List[int]):
        """
        data req meta_info里需要包含:
            request_id: str
            response_callback_fn: callable
        generation_config, 按request设置
        """
        if command == GenerateRequestType.ALIVE_CHECK:
            if self.thread_server is not None:
                if not self.thread_server.is_alive():
                    raise Exception("thread server has stopped unexpectedly. check stderr for more info.")
            output = DataProto(meta_info={"request_counts": len(self.response_call_back_fns)})
            return output
        elif command == GenerateRequestType.ADD:
            assert "response_callback_fn" in data.meta_info, "response_callback_fn is not in data.meta_info"
            is_num_return_sequences_expand = data.meta_info.get("is_num_return_sequences_expand", False)
            if "generation_config" not in data.meta_info:
                generation_config = self.worker_config.generating_args.to_dict()
                if is_num_return_sequences_expand:
                    self.worker_config.generating_args.num_return_sequences = 1
                    generation_config["num_return_sequences"] = 1
                    self.logger.info(f"is_num_return_sequences_expand is True, set num_return_sequences to 1.")
            else:
                generation_config = data.meta_info["generation_config"]
            generation_config["eos_token_id"] = [
                self.tokenizer.eos_token_id
            ] + self.tokenizer.additional_special_tokens_ids
            generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            data.meta_info["generation_config"] = generation_config
            self.response_call_back_fns[data.meta_info["request_id"]] = data.meta_info.pop("response_callback_fn")
        self.strategy.add_request(command=command, data=data)
        return DataProto(meta_info={"request_counts": len(self.response_call_back_fns)})

    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    def add_batched_request(self, command, data_list: List[DataProto]):
        if command == GenerateRequestType.ADD:
            for data in data_list:
                assert "response_callback_fn" in data.meta_info, "response_callback_fn is not in data.meta_info"
                is_num_return_sequences_expand = data.meta_info.get("is_num_return_sequences_expand", False)
                if "generation_config" not in data.meta_info:
                    generation_config = self.worker_config.generating_args.to_dict()
                    if is_num_return_sequences_expand:
                        self.worker_config.generating_args.num_return_sequences = 1
                        generation_config["num_return_sequences"] = 1
                        self.logger.info(f"is_num_return_sequences_expand is True, set num_return_sequences to 1.")
                else:
                    generation_config = data.meta_info["generation_config"]
                generation_config["eos_token_id"] = [
                    self.tokenizer.eos_token_id
                ] + self.tokenizer.additional_special_tokens_ids
                generation_config["pad_token_id"] = self.tokenizer.pad_token_id
                data.meta_info["generation_config"] = generation_config
                self.response_call_back_fns[data.meta_info["request_id"]] = data.meta_info.pop("response_callback_fn")
            self.strategy.add_batched_request(command=command, data_list=data_list)
        return None
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    def add_request(self, command, data: DataProto):
        """
        data req meta_info里需要包含:
            request_id: str
            response_callback_fn: callable
        generation_config, 按request设置
        """
        if command == GenerateRequestType.ALIVE_CHECK:
            if self.thread_server is not None:
                if not self.thread_server.is_alive():
                    raise Exception("thread server has stopped unexpectedly. check stderr for more info.")
            output = DataProto(meta_info={"request_counts": len(self.response_call_back_fns)})
            return output
        elif command in [GenerateRequestType.ADD, GenerateRequestType.RESUME]:
            if "response_callback_fn" not in data.meta_info: 
                print(f"data.meta_info is {data.meta_info}")
            assert "response_callback_fn" in data.meta_info, "response_callback_fn is not in data.meta_info"
            is_num_return_sequences_expand = data.meta_info.get("is_num_return_sequences_expand", False)
            if "generation_config" not in data.meta_info:
                generation_config = self.worker_config.generating_args.to_dict()
                if is_num_return_sequences_expand:
                    self.worker_config.generating_args.num_return_sequences = 1
                    generation_config["num_return_sequences"] = 1
                    self.logger.info(f"is_num_return_sequences_expand is True, set num_return_sequences to 1.")
            else:
                generation_config = data.meta_info["generation_config"]
            generation_config["eos_token_id"] = [
                self.tokenizer.eos_token_id
            ] + self.tokenizer.additional_special_tokens_ids
            generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            data.meta_info["generation_config"] = generation_config
            self.response_call_back_fns[data.meta_info["request_id"]] = data.meta_info.pop("response_callback_fn")
        self.strategy.add_request(command=command, data=data)
        return DataProto(meta_info={"request_counts": len(self.response_call_back_fns)})

    def request_complete(self, data: DataProto):
        data.meta_info["eos_token_id"] = self.tokenizer.eos_token_id
        data.meta_info["pad_token_id"] = self.tokenizer.pad_token_id
        response_call_back_fn = self.response_call_back_fns.pop(data.meta_info["request_id"])
        # print(f"response_call_back_fn is {response_call_back_fn} in request_complete", flush=True)
        self.response_callback_refs.append(response_call_back_fn(data))


class CriticWorker(Worker):

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        self.strategy.initialize(model_provider=default_value_model_provider)
        self.tokenizer = self.strategy.tokenizer

        if self.pipeline_config.resume_from_checkpoint:
            load_dir = os.path.join(download_model(self.pipeline_config.resume_from_checkpoint), self.cluster_name)
            self.strategy.load_checkpoint(load_dir=load_dir, tag="checkpoint")

        self.logger.info(f"{self.worker_name} initialized")

        self.strategy.offload_states()

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def compute_values(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'values': values})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_values",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            data = data.to("cuda")
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
            with torch.no_grad():
                results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                    batch=data, forward_func=self.forward_func_values
                )

            output = DataProto.from_dict(tensors={"values": results["values"]})
            data.to("cpu")
            output = output.to("cpu")

        output.meta_info = {"metrics": metrics}
        return output

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def train_step(self, data: DataProto):
        """
        return DataProto(meta_info={'metrics': metrics})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/train_step",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params, OffloadStateType.other_params]},
        ):
            data = data.to("cuda")
            per_device_train_batch_size = self.worker_config.training_args.per_device_train_batch_size
            backward_batch_size = (
                per_device_train_batch_size * self.worker_config.training_args.gradient_accumulation_steps
            )

            dataloader = data.make_iterator(
                mini_batch_size=backward_batch_size,
                epochs=1,
                seed=self.pipeline_config.seed,
                dataloader_kwargs={"shuffle": True},
            )

            for batch_idx, data in tqdm(
                enumerate(dataloader),
                desc=f"{self.worker_name} train global step {global_step}",
                total=data.batch.batch_size[0] * self.pipeline_config.ppo_epochs // backward_batch_size,
            ):
                vf_metrics = self.strategy.train_step(batch=data, loss_func=self.loss_func)
                append_to_dict(metrics, vf_metrics)

            data.to("cpu")
            metrics["critic/lr"] = self.strategy.scheduler.get_last_lr()[0]

        output = DataProto(meta_info={"metrics": metrics}).to("cpu")
        return output

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        response_mask = data.batch["response_mask"][:, 1:]
        old_values = data.batch["values"]
        returns = data.batch["returns"]

        values, _ = self.forward_func_values(data=data, output_tensor=output_tensor)

        if self.pipeline_config.value_clip is not None:
            values_clipped = torch.clip(
                values,
                old_values - self.pipeline_config.value_clip,
                old_values + self.pipeline_config.value_clip,
            )
            surr1 = (values - returns) ** 2
            surr2 = (values_clipped - returns) ** 2
            vf_clipfrac = masked_mean(torch.gt(surr2, surr1).float(), response_mask, dim=-1).mean()
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2
            vf_clipfrac = masked_mean(loss, response_mask, dim=-1).mean()

        vf_loss = 0.5 * masked_mean(loss, response_mask, dim=-1).mean()

        vf_metrics = {
            "critic/loss": vf_loss.detach().item(),
            "critic/value": (masked_mean(old_values, response_mask, dim=-1)).mean().detach().item(),
            "critic/vpred": (masked_mean(values, response_mask, dim=-1)).mean().detach().item(),
            "critic/clipfrac": vf_clipfrac.detach().item(),
            "critic/error": masked_mean((values - returns) ** 2, response_mask, dim=-1).mean().detach().item(),
        }

        return vf_loss, vf_metrics

    def forward_func_values(self, data: DataProto, output_tensor: torch.Tensor):
        values = output_tensor[:, :-1]
        values = values.squeeze(dim=-1)
        return values, {"values": values.clone().detach()}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def do_checkpoint(self, global_step):
        with Timer("do_checkpoint") as total_timer:
            ckpt_id = f"checkpoint-{global_step}"
            save_dir = os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id, self.cluster_name)
            critic_save_dir = os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id)
            self.logger.info(f"save checkpoint-{global_step} to {save_dir}")
            exec_metrics: Dict = self.strategy.save_checkpoint(save_dir, global_step, ckpt_id, local_state_path=critic_save_dir)

        metrics = {
            f"time/{self.cluster_name}/do_checkpoint/total": total_timer.last,
        }
        metric_prefix = f"time/{self.cluster_name}/do_checkpoint"
        metrics.update({f"{metric_prefix}/{k}": v for k, v in exec_metrics.items()})
        output = DataProto(meta_info={"metrics": metrics})
        return output


class RewardWorker(Worker):
    """
    Reward Model 使用 AutoModelForSequenceClassification 协议
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        self.strategy.initialize(model_provider=default_reward_model_provider)
        self.tokenizer = self.strategy.tokenizer

        self.logger.info(f"{self.worker_name} initialized")
        self.strategy.offload_states()

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'rewards': rewards})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_rewards",
            is_offload_states=is_offload_states,
        ):
            data = data.to("cuda")

            # TODO: _switch_chat_template, 异构reward model

            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
            with torch.no_grad():
                results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                    batch=data, forward_func=self.forward_func_values
                )
            token_level_rewards = results["values"]  # (bsz, input_ids.shape[1]-1)
            input_ids = data.batch["input_ids"][:, 1:]
            seq_lengths = torch.eq(input_ids, self.tokenizer.pad_token_id).int().argmax(-1) - 1
            seq_lengths = (seq_lengths % input_ids.shape[-1]).to(token_level_rewards.device)
            response_level_rewards = token_level_rewards[
                torch.arange(seq_lengths.shape[0], device=token_level_rewards.device), seq_lengths
            ]

            output = DataProto.from_dict(
                tensors={"token_level_rewards": token_level_rewards, "response_level_rewards": response_level_rewards}
            )

            data.to("cpu")
            output = output.to("cpu")

        output.meta_info = {"metrics": metrics}
        return output

    def forward_func_values(self, data: DataProto, output_tensor: torch.Tensor):
        values = output_tensor[:, 1:]
        values = values.squeeze(dim=-1)
        return values, {"values": values.clone().detach()}
