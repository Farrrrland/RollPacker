import os
from datetime import timedelta
from typing import Callable, Tuple, Dict
import numpy as np 
import random 
import torch
import torch.distributed as dist
import ray
from codetiming import Timer
from transformers import AutoModelForCausalLM, AutoConfig, AutoModelForVision2Seq
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, CPUOffload
from torch import optim

# deepspeed.init_distributed(timeout=timedelta(minutes=self.worker_config.backend_timeout))
import torch.distributed


from roll.utils.collective import collective
from tqdm import tqdm
from transformers import get_scheduler
from transformers.integrations import HfDeepSpeedConfig
from transformers import set_seed
from torch.distributed.device_mesh import init_device_mesh


from roll.datasets.collator import collate_fn_to_dict_list
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_tokenizer_provider, default_processor_provider
from roll.models.func_providers import log_probs_forward_step_func
from roll.third_party.deepspeed.offload_states_patch import bind_deepspeed_offload_states_func
from roll.utils.deepspeed_utils import get_optimizer_grouped_parameters
from roll.utils.functionals import append_to_dict, log_probs_from_logits
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType

import warnings
from roll.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from roll.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_model_to_cpu, offload_fsdp_model_gradients_to_cpu, load_fsdp_optimizer, \
    load_fsdp_model_to_gpu, load_fsdp_model_gradients_to_gpu
from roll.utils.torch_functional import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp import ShardedStateDictConfig, ShardedOptimStateDictConfig
import torch.distributed as dist 

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
from contextlib import AbstractContextManager, contextmanager, nullcontext

logger = get_logger()



# a potential improvement: https://github.com/pytorch/pytorch/blob/v2.6.0/torch/distributed/fsdp/_runtime_utils.py#L691
def all_reduce_full_grad_hook(fsdp_module, mesh_dim='ddp'):
    """
    Hook 函数：在 backward 后，对所有 flatten 参数的梯度执行 all_reduce
    """
    device_mesh = fsdp_module._device_mesh
    if device_mesh.ndim == 1 or device_mesh.shape[0] == 1: 
        return 
    if device_mesh is None or not fsdp_module._sync_gradients:
        return 
    world_size = device_mesh.size(0)
    group = device_mesh.get_group(mesh_dim)
    for param in fsdp_module.params:
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group)
            dist.barrier(group=group)



def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    else:
        device_mesh = init_device_mesh('cuda',
                                       mesh_shape=(world_size // fsdp_size, fsdp_size),
                                       mesh_dim_names=['ddp', 'fsdp'])
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy
    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


class FSDPInferStrategy(InferenceStrategy):
    strategy_name = "fsdp_infer"

    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.worker_config.strategy_args.strategy_config["train_micro_batch_size_per_gpu"] = (
            self.worker_config.training_args.per_device_train_batch_size
        )

        # deepspeed的train_batch_size是全局batch_size
        self.worker_config.strategy_args.strategy_config["train_batch_size"] = (
            self.worker_config.training_args.per_device_train_batch_size
            * self.worker_config.training_args.gradient_accumulation_steps
            * self.worker.world_size
        )
        
        self.use_remove_padding = self.worker_config.use_remove_padding 

    def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        
        fsdp_size = self.worker_config.strategy_args.strategy_config.get('fsdp_size', -1)
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.worker_config.strategy_args.strategy_config.get('context_parallel_size', 1)
        if False: 
            self.ulysses_device_mesh = None
            self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
            dp = world_size // self.ulysses_sequence_parallel_size
            if self.ulysses_sequence_parallel_size > 1:
                self.ulysses_device_mesh = init_device_mesh(
                    device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
                )

            self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)


        dp = world_size // self.ulysses_sequence_parallel_size
        # end of dist init
        
        dist.all_reduce(torch.zeros(1).cuda())
        
        rank = dist.get_rank()



        self.worker.rank_info.dp_rank = dist.get_rank()
        self.worker.rank_info.dp_size = dist.get_world_size()
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.processor = default_processor_provider(model_args=self.worker_config.model_args)

        # in transformers 4.49.0, qwen2.5-vl using fa2 with apply_rotary_pos_emb_flashatt
        # has dtype error when used with deepspeed, please refer to
        # https://github.com/huggingface/transformers/pull/36188
        # in transformers 4.50.0, shape error: shape '[8, 5120, -1, 128]' is invalid for
        # query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        # https://github.com/QwenLM/Qwen2.5-VL/issues/1032
        # patch for transformers 4.49.0 currently
        try:
            import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as modeling_qwen2_5_vl

            def apply_rotary_pos_emb_flashatt(
                q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                cos = cos.chunk(2, dim=-1)[0].contiguous()
                sin = sin.chunk(2, dim=-1)[0].contiguous()
                q_embed = modeling_qwen2_5_vl.apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
                k_embed = modeling_qwen2_5_vl.apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
                return q_embed, k_embed

            modeling_qwen2_5_vl.apply_rotary_pos_emb_flashatt = apply_rotary_pos_emb_flashatt
        except:
            # if not include qwen2_5_vl, throw exception by others
            pass

        model = model_provider(tokenizer=self.tokenizer, model_args=self.worker_config.model_args, is_trainable=False)
        trust_remote_code = self.worker_config.strategy_args.strategy_config.get('trust_remote_code', False)
        local_path = self.worker_config.model_args.model_name_or_path
        actor_model_config = AutoConfig.from_pretrained(self.worker_config.model_args.model_name_or_path, trust_remote_code=trust_remote_code)

        torch_dtype = self.worker_config.strategy_args.strategy_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
            # torch_dtype = torch.float32
        
        
        if False: 
            self.model, *_ = deepspeed.initialize(
                model=model,
                config=self.worker_config.strategy_args.strategy_config,
                dist_init_required=True,
            )

            bind_deepspeed_offload_states_func(self.model)

            logger.info(f"{self.model}")
        else: 
            pass 

            # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
            init_context = get_init_weight_context_manager(use_meta_tensor=not getattr(model.config, "tie_word_embeddings", False),
                                                        mesh=self.device_mesh)

            with init_context(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                actor_module = model 
                if False: 
                    if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                        actor_module_class = AutoModelForVision2Seq
                    else:
                        actor_module_class = AutoModelForCausalLM

                    actor_module = actor_module_class.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                    torch_dtype=torch_dtype,
                                                                    config=actor_model_config,
                                                                    attn_implementation='flash_attention_2',
                                                                    trust_remote_code=trust_remote_code)
                
                

                # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
                actor_module.to(torch_dtype)
                fsdp_config = self.worker_config.strategy_args.strategy_config

                

            # end of with context statement



            if True: 
                param_dtype = torch.bfloat16
                # param_dtype = torch.float32
                reduce_dtype = torch.float32
                buffer_dtype = torch.float32

                mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
                
                

            auto_wrap_policy = None # get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))

            fsdp_mesh = self.device_mesh
            sharding_strategy = get_sharding_strategy(fsdp_mesh)
            cpu_offload = False # CPUOffload(offload_params=True)

            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=True,
                auto_wrap_policy=auto_wrap_policy,
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False)
            # import pdb; pdb.set_trace()
            self.model = actor_module_fsdp

        dist.barrier()

    def forward_step(
        self,
        batch: DataProto,
        forward_func: Callable[[DataProto, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        batch_size = batch.batch.batch_size[0]
        micro_batch_size = batch.meta_info["micro_batch_size"]
        num_microbatches = max(batch_size // micro_batch_size, 1)
        micro_batches = batch.chunk(chunks=num_microbatches)
        losses_reduced = []
        print(f'length of micro_batches is {len(micro_batches)}', flush=True)
        use_remove_padding = self.use_remove_padding 

        for data in micro_batches:
            input_ids = data.batch["input_ids"]
            attention_mask = data.batch["attention_mask"]
            position_ids = data.batch["position_ids"]
            forward_args = data.meta_info.get("forward_args", {})
            if position_ids.dim() == 3:
                # qwen2vl mrope, maybe use a placeholder and let model generate position_ids
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
            if "multi_modal_inputs" in data.non_tensor_batch:
                multi_modal_inputs = data.non_tensor_batch["multi_modal_inputs"]
                for key in multi_modal_inputs[0].keys():
                    assert key not in forward_args
                    # DataProto.to('cuda') in upper frame not work for non_tensor_batch
                    forward_args[key] = torch.concat([inputs[key] for inputs in multi_modal_inputs], dim=0).to(
                        input_ids.device
                    )

            if use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                                                              1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if False: # self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.model(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           **forward_args)  # prevent model thinks we are generating
                # output = self.model(input_ids=input_ids_rmpad,attention_mask=None,position_ids=position_ids_rmpad)
                output.logits = pad_input(hidden_states=output.logits.squeeze(0).unsqueeze(-1), indices=indices, batch=input_ids.size(0), seqlen=input_ids.size(1)).squeeze(-1)
            else: 
                output = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **forward_args
                )
            
            loss, loss_reduced = forward_func(data, output.logits)
            losses_reduced.append(loss_reduced)
        results = collate_fn_to_dict_list(losses_reduced)
        return results

    def generate(self, batch: DataProto, generation_config):
        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **generation_config,
        )

        return output

    def unwrap_model(self):
        return self.model.module

    # 参数同步相关接口
    def broadcast_parameter(self, model_update_name, src_pp_rank, dtype, shape, parameter_name):
        comm_plan = self.model_update_comm_plan[model_update_name][src_pp_rank]
        # comm_plan = self.model_update_comm_plan[src_pp_rank]
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        collective.broadcast(tensor=weight, src_rank=0, group_name=comm_plan["group_name"])
        param = self.model.get_parameter(parameter_name)
        if not self.ds_config.is_zero3():
            param.data.copy_(weight.to("cpu"))
        else:
            with GatheredParameters([param], modifier_rank=0):
                if dist.get_rank() == 0:
                    param.data.copy_(weight)
        del weight

    def update_parameter(self, model_update_name, parameter_name, weight, ranks_in_worker):
        # import pdb; pdb.set_trace() 
        param = self.model.get_parameter(parameter_name)
        if not self.ds_config.is_zero3():
            param.data.copy_(weight)
        else:
            with GatheredParameters([param], modifier_rank=0):
                if dist.get_rank() == 0:
                    param.data.copy_(weight)
        del weight


    # offload/load 相关接口
    def load_states(self, include=None, non_blocking=False):
        if include is None:
            include = [OffloadStateType.model_params, OffloadStateType.other_params, OffloadStateType.optimizer_states]
        

        if OffloadStateType.model_params in include: 
            load_fsdp_model_to_gpu(self.model)
        
        # if OffloadStateType.other_params in include: 
        #     load_fsdp_model_gradients_to_gpu(self.model)

        if OffloadStateType.optimizer_states in include: 
            load_fsdp_optimizer(optimizer=self.optimizer, device_id=torch.cuda.current_device())
            
            

    def offload_states(self, include=None, non_blocking=False):
        if include is None:
            include = [OffloadStateType.model_params, OffloadStateType.other_params, OffloadStateType.optimizer_states]
        
        # print(f"include is {include}")

        if OffloadStateType.model_params in include:
            offload_fsdp_model_to_cpu(self.model)
        
        # if OffloadStateType.other_params in include: 
        #     offload_fsdp_model_gradients_to_cpu(model=self.model)

        if OffloadStateType.optimizer_states in include and hasattr(self, 'optimizer'):
            offload_fsdp_optimizer(optimizer=self.optimizer)
        
        torch.cuda.empty_cache()



class FSDPTrainStrategy(FSDPInferStrategy, TrainStrategy):
    strategy_name = "fsdp_train"

    def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        
        fsdp_size = self.worker_config.strategy_args.strategy_config.get('fsdp_size', -1)
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.worker_config.strategy_args.strategy_config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        # end of dist init
        

        self.worker.rank_info.dp_rank = dist.get_rank()
        self.worker.rank_info.dp_size = dist.get_world_size()

        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)

        # in transformers 4.49.0, qwen2.5-vl using fa2 with apply_rotary_pos_emb_flashatt
        # has dtype error when used with deepspeed, please refer to
        # https://github.com/huggingface/transformers/pull/36188
        # in transformers 4.50.0, shape error: shape '[8, 5120, -1, 128]' is invalid for
        # query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        # https://github.com/QwenLM/Qwen2.5-VL/issues/1032
        # patch for transformers 4.49.0 currently
        try:
            import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as modeling_qwen2_5_vl

            def apply_rotary_pos_emb_flashatt(
                q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                cos = cos.chunk(2, dim=-1)[0].contiguous()
                sin = sin.chunk(2, dim=-1)[0].contiguous()
                q_embed = modeling_qwen2_5_vl.apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
                k_embed = modeling_qwen2_5_vl.apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
                return q_embed, k_embed

            modeling_qwen2_5_vl.apply_rotary_pos_emb_flashatt = apply_rotary_pos_emb_flashatt
        except:
            # if not include qwen2_5_vl, throw exception by others
            pass

        model = model_provider(tokenizer=self.tokenizer, model_args=self.worker_config.model_args, is_trainable=True)

        torch_dtype = self.worker_config.strategy_args.strategy_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        

        if False: 
            adam_optimizer = FusedAdam
            optim_params = get_optimizer_grouped_parameters(
                model, weight_decay=self.worker_config.training_args.weight_decay
            )
            optimizer = adam_optimizer(
                optim_params,
                lr=self.worker_config.training_args.learning_rate,
                betas=(self.worker_config.training_args.adam_beta1, self.worker_config.training_args.adam_beta2),
            )

        logger.info(f"max steps pipeline {self.worker_config.training_args.max_steps}")
        self.worker_config.training_args.max_steps = (
            self.worker_config.training_args.max_steps // self.worker.rank_info.dp_size
        )
        logger.info(f"max steps worker train {self.worker_config.training_args.max_steps}")



        if False: # original deepspeed implementation 
            self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
                model_parameters=model.parameters(),
                model=model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                config=self.worker_config.strategy_args.strategy_config,
                dist_init_required=True,
            )
            bind_deepspeed_offload_states_func(self.model)

            logger.info(f"{self.model}")
        else: 
            # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
            init_context = get_init_weight_context_manager(use_meta_tensor=not getattr(model.config, "tie_word_embeddings", False),
                                                        mesh=self.device_mesh)

            with init_context(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                actor_module = model 
                

                # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
                actor_module.to(torch_dtype)
                fsdp_config = self.worker_config.strategy_args.strategy_config

            # end of with context statement


            if True: 
                param_dtype = torch.bfloat16
                # param_dtype = torch.float32
                reduce_dtype = torch.float32
                buffer_dtype = torch.float32
                mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
                
                

            auto_wrap_policy = None # get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))

            fsdp_mesh = self.device_mesh
            sharding_strategy = get_sharding_strategy(fsdp_mesh)
            cpu_offload = None # CPUOffload(offload_params=True)
            
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=True, # False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False)
            
            self.model = actor_module_fsdp

        self.optimizer = optim.AdamW(self.model.parameters(),
                        lr=self.worker_config.training_args.learning_rate,
                        betas=(self.worker_config.training_args.adam_beta1, self.worker_config.training_args.adam_beta2),
                        weight_decay=self.worker_config.training_args.weight_decay)


        self.scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
            num_warmup_steps=self.worker_config.training_args.get_warmup_steps(
                self.worker_config.training_args.max_steps
            ),
            num_training_steps=self.worker_config.training_args.max_steps)

        for idx, (name, param) in enumerate(self.model.named_parameters()):
            if param.requires_grad and idx < 1:
                print(f'register {name}')
                param.register_hook(lambda grad, name=name: print(f"--- {name}, rank {torch.distributed.get_rank()} grad: {grad}", flush=True))
        dist.barrier()

    def train_step(
        self,
        batch: DataProto,
        loss_func: Callable[[DataProto, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
        no_sync: bool=False,
        in_no_sync_train: bool=False,
    ):
        self.model.train()
        mini_batch_size = self.worker_config.training_args.per_device_train_batch_size
        data_iter = batch.make_iterator(mini_batch_size=mini_batch_size, epochs=1)
        mini_steps = batch.batch.batch_size[0] // self.worker_config.training_args.per_device_train_batch_size
        gradient_accumulation_steps = self.worker_config.training_args.gradient_accumulation_steps
        assert gradient_accumulation_steps >= mini_steps, \
            f"gradient_accumulation_steps {gradient_accumulation_steps} should be greater than or equal to mini_steps {mini_steps}"
        loss_scale = 1. # / (gradient_accumulation_steps * mini_batch_size * self.worker.rank_info.dp_size)
        metrics = {}
        # TODO: 
        # 1. gradient_accumulation, 
        # 2. loss, entropy 
        # self.gradient_accumulation = self.worker_config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
        use_remove_padding = self.use_remove_padding
        for step in range(mini_steps):
            if no_sync:
                print("run step {} {}".format(step, no_sync), flush=True)
            data: DataProto = next(data_iter)
            input_ids = data.batch["input_ids"]
            attention_mask = data.batch["attention_mask"]
            position_ids = data.batch["position_ids"]
            forward_args = data.meta_info.get("forward_args", {})
            if position_ids.dim() == 3:
                # qwen2vl mrope, maybe use a placeholder and let model generate position_ids
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
            if "multi_modal_inputs" in data.non_tensor_batch:
                multi_modal_inputs = data.non_tensor_batch["multi_modal_inputs"]
                for key in multi_modal_inputs[0].keys():
                    assert key not in forward_args
                    # DataProto.to('cuda') in upper frame not work for non_tensor_batch
                    forward_args[key] = torch.concat([inputs[key] for inputs in multi_modal_inputs], dim=0).to(
                        input_ids.device
                    )
            if no_sync:
                with self.model.no_sync():
                    if use_remove_padding:
                        input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                                attention_mask)  # input_ids_rmpad (total_nnz, ...)
                        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                        # unpad the position_ids to align the rotary
                        if position_ids.dim() == 3:
                            position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                                indices).transpose(0, 1).unsqueeze(
                                                                    1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                        else:
                            position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                                indices).transpose(0, 1)

                        # for compute the log_prob
                        input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                        input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                        # only pass input_ids and position_ids to enable flash_attn_varlen
                        output = self.model(input_ids=input_ids_rmpad,
                                                attention_mask=None,
                                                position_ids=position_ids_rmpad,
                                                **forward_args)  # prevent model thinks we are generating
                        # output = self.model(input_ids=input_ids_rmpad,attention_mask=None,position_ids=position_ids_rmpad)
                        output.logits = pad_input(hidden_states=output.logits.squeeze(0).unsqueeze(-1), indices=indices, batch=input_ids.size(0), seqlen=input_ids.size(1)).squeeze(-1)
                    else: 
                        output = self.model(
                            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **forward_args
                        )
                    loss, loss_reduced = loss_func(data, output.logits)
                    loss = loss * loss_scale
                    append_to_dict(metrics, loss_reduced)
                    loss.backward()
            else:
                if use_remove_padding: 
                    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                                attention_mask)  # input_ids_rmpad (total_nnz, ...)
                    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                    # unpad the position_ids to align the rotary
                    if position_ids.dim() == 3:
                        position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                            indices).transpose(0, 1).unsqueeze(
                                                                1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                    else:
                        position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                            indices).transpose(0, 1)

                    # for compute the log_prob
                    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                    # only pass input_ids and position_ids to enable flash_attn_varlen
                    output = self.model(input_ids=input_ids_rmpad,
                                            attention_mask=None,
                                            position_ids=position_ids_rmpad,
                                            **forward_args)  # prevent model thinks we are generating
                    # output = self.model(input_ids=input_ids_rmpad,attention_mask=None,position_ids=position_ids_rmpad)
                    output.logits = pad_input(hidden_states=output.logits.squeeze(0).unsqueeze(-1), indices=indices, batch=input_ids.size(0), seqlen=input_ids.size(1)).squeeze(-1)
                else: 
                    
                    output = self.model(
                            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **forward_args
                        )
                loss, loss_reduced = loss_func(data, output.logits)
                loss = loss * loss_scale
                append_to_dict(metrics, loss_reduced)
                loss.backward()
                all_reduce_full_grad_hook(fsdp_module=self.model)
            
            if ((step + 1) % gradient_accumulation_steps == 0 or (step + 1 == mini_steps) ) and not no_sync: 
                
                self.load_states(include=[OffloadStateType.optimizer_states])
                
                grad_norm = self.model.clip_grad_norm_(max_norm=self.worker.pipeline_config.max_grad_norm)
                metrics.update({self.worker_config.name + "/" + "grad_norm": grad_norm.item()})
                if not torch.isfinite(grad_norm):
                    print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.offload_states(include=[OffloadStateType.optimizer_states])
        torch.cuda.empty_cache()
        return metrics


    def save_checkpoint(self, save_dir, global_step, ckpt_id, tag="checkpoint", **kwargs):
        """
        save ckpt/hf model/tokenizer to local dir
        save_dir/actor_train/{hf files}
        save_dir/actor_train/checkpoint/{checkpoint files}
        """
        logger.info(f"save_dir: {save_dir}")
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)

        with Timer("load", logger=None) as load_timer:
            self.load_states()

        if False: # self.ds_config.is_zero3():
            if self.model.zero_gather_16bit_weights_on_model_save():
                state_dict = self.model._zero3_consolidated_16bit_state_dict()
            else:
                raise ValueError(
                    "Cannot get 16bit model weights because `stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False. "
                    "To save the model weights in 16bit, set `stage3_gather_16bit_weights_on_model_save` to True in DeepSpeed config file or "
                    "set `zero3_save_16bit_model` to True when using `accelerate config`. "
                    "To save the full checkpoint, run `model.save_checkpoint(save_dir)` and use `zero_to_fp32.py` to recover weights."
                )
        else:
            state_dict = self.model.state_dict()

        # save fsdp 

        if True: 
            state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
            optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                    model_state_dict = self.model.state_dict()
                    if self.optimizer is not None:
                        optimizer_state_dict = self.optimizer.state_dict()
                    else:
                        optimizer_state_dict = None
                    if self.scheduler is not None:
                        lr_scheduler_state_dict = self.scheduler.state_dict()
                    else:
                        lr_scheduler_state_dict = None

                    extra_state_dict = {
                        'lr_scheduler': lr_scheduler_state_dict,
                        'rng': self.get_rng_state(),
                    }
                    world_size = torch.distributed.get_world_size()
                    model_path = os.path.join(save_dir, f'model_world_size_{world_size}_rank_{self.worker.rank_info.dp_rank}.pt')
                    optim_path = os.path.join(save_dir, f'optim_world_size_{world_size}_rank_{self.worker.rank_info.dp_rank}.pt')
                    extra_path = os.path.join(save_dir, f'extra_state_world_size_{world_size}_rank_{self.worker.rank_info.dp_rank}.pt')
                    torch.save(model_state_dict, model_path)
                    torch.save(optimizer_state_dict, optim_path)  # TODO: address optimizer is None
                    torch.save(extra_state_dict, extra_path)
        # save huggingface pretrained model
        
        if dist.get_rank() == 0:
            # import pdb; pdb.set_trace() 
            # TODO
            self.model.module.save_pretrained(save_dir, state_dict=state_dict, safe_serialization=False)
            self.tokenizer.save_pretrained(save_dir)
            if getattr(self, "processor", None):
                self.processor.save_pretrained(save_dir)
            # save tokenizer
        # self.model.save_checkpoint(save_dir, tag=tag, **kwargs)
        # import pdb; pdb.set_trace() 

        if self.worker_config.checkpoint_config.get("async_upload", True):
            self.thread_executor.submit(self.checkpoint_manager.upload, ckpt_id=ckpt_id, local_state_path=save_dir)
        else:
            self.checkpoint_manager.upload(ckpt_id=ckpt_id, local_state_path=save_dir)

        metrics = {
            "load": load_timer.last,
        }
        return metrics

    def load_checkpoint(self, load_dir, tag="checkpoint", **kwargs):
        # every rank download its own checkpoint
        world_size = torch.distributed.get_world_size()
        rank = self.worker.rank_info.dp_rank
        local_model_path = os.path.join(load_dir, f'model_world_size_{world_size}_rank_{rank}.pt')
        local_optim_path = os.path.join(load_dir, f'optim_world_size_{world_size}_rank_{rank}.pt')
        local_extra_state_path = os.path.join(load_dir,
                                               f'extra_state_world_size_{world_size}_rank_{rank}.pt')
        
        model_state_dict = torch.load(local_model_path, weights_only=False)
        optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
        extra_state_dict = torch.load(local_extra_state_path, weights_only=False)

        lr_scheduler_state_dict = extra_state_dict['lr_scheduler']

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            self.model.load_state_dict(model_state_dict)
            if self.optimizer is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
        # recover random state
        if 'rng' in extra_state_dict:
            # 'rng' may not exist for backward compatibility
            self.load_rng_state(extra_state_dict['rng'])

        if self.scheduler is not None:
            self.scheduler.load_state_dict(lr_scheduler_state_dict)

    def model_update(self, model_update_name, tgt_workers, broadcast_tgt_devices, p2p_tgt_devices):
        comm_plan = self.model_update_comm_plan[model_update_name][self.worker.rank_info.pp_rank]
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            state_dict = self.model.state_dict()
        if "lm_head.weight" in state_dict: 
            state_dict['model.embed_tokens.weight'] = state_dict['lm_head.weight']
            state_dict.pop('lm_head.weight')
            # might consider vllm_model.config.tie_word_embeddings

        broadcast_time_cost = 0
        with Timer("model_update_total", logger=None) as timer_total:
            for param_name, param in tqdm(
                state_dict.items(), desc="weight update progress", total=len(list(state_dict.items()))
            ):
                shape = param.shape 
                if False: # not self.ds_config.is_zero3():
                    param_weight = param.data
                    refs = []
                    for p2p_tgt_device in p2p_tgt_devices:
                        p2p_tgt_worker = tgt_workers[p2p_tgt_device["rank"]]
                        ref = p2p_tgt_worker.update_parameter.remote(
                            model_update_name=model_update_name,
                            parameter_name=param_name,
                            weight=param_weight,
                            ranks_in_worker=[p2p_tgt_device["device"]["rank"]],
                        )
                        refs.append(ref)

                    if (
                        self.worker.rank_info.tp_rank == 0
                        and self.worker.rank_info.cp_rank == 0
                        and self.worker.rank_info.dp_rank == 0
                    ):
                        for worker in tgt_workers:
                            ref = worker.broadcast_parameter.remote(
                                model_update_name=model_update_name,
                                src_pp_rank=self.worker.rank_info.pp_rank,
                                dtype=param_weight.dtype,
                                shape=shape,
                                parameter_name=param_name,
                            )
                            refs.append(ref)
                    if len(broadcast_tgt_devices) > 0:
                        collective.broadcast(tensor=param_weight, src_rank=0, group_name=comm_plan["group_name"])
                    ray.get(refs)

                else:
                    if True: 
                        param_weight = param.data
                        # import pdb; pdb.set_trace() 
                        with Timer("broadcast", logger=None) as timer_broadcast:
                            refs = []
                            for p2p_tgt_device in p2p_tgt_devices:
                                p2p_tgt_worker = tgt_workers[p2p_tgt_device["rank"]]
                                ref = p2p_tgt_worker.update_parameter.remote(
                                    model_update_name=model_update_name,
                                    parameter_name=param_name,
                                    weight=param_weight,
                                    ranks_in_worker=[p2p_tgt_device["device"]["rank"]],
                                )
                                refs.append(ref)

                            if (
                                self.worker.rank_info.tp_rank == 0
                                and self.worker.rank_info.cp_rank == 0
                                and self.worker.rank_info.dp_rank == 0
                            ):
                                for worker in tgt_workers:
                                    ref = worker.broadcast_parameter.remote(
                                        model_update_name=model_update_name,
                                        src_pp_rank=self.worker.rank_info.pp_rank,
                                        dtype=param_weight.dtype,
                                        shape=shape,
                                        parameter_name=param_name,
                                    )
                                    refs.append(ref)
                            if len(broadcast_tgt_devices) > 0:
                                collective.broadcast(
                                    tensor=param_weight, src_rank=0, group_name=comm_plan["group_name"]
                                )
                            ray.get(refs)
                        broadcast_time_cost += timer_broadcast.last

        metrics = {
            "all_gather": timer_total.last - broadcast_time_cost,
            "broadcast": broadcast_time_cost,
        }
        return metrics

    @staticmethod
    def get_rng_state():
        rng_state = {
            'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(),
            'numpy': np.random.get_state(),
            'random': random.getstate(),
        }
        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        torch.set_rng_state(rng_state['cpu'])
        torch.cuda.set_rng_state(rng_state['cuda'])
        np.random.set_state(rng_state['numpy'])
        random.setstate(rng_state['random'])
