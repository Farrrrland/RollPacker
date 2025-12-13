import asyncio
import copy
import gc
import itertools
import os
import queue
from concurrent import futures
from typing import Dict, List, Optional, Union

import ray
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed
from mcore_adapter.models.converter.convert_utils import RecvBucketManager
import vllm 
from vllm import SamplingParams, RequestOutput, envs
from vllm import RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.utils import random_uuid
import vllm

from mcore_adapter.models.converter.convert_utils import RecvBucketManager
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.third_party.vllm import LLM, AsyncLLM
from roll.utils.collective import collective
from roll.utils.functionals import GenerateRequestType, concatenate_input_and_output
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType

import time

logger = get_logger()


class VllmStrategy(InferenceStrategy):
    strategy_name = "vllm"

    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.model: Union[LLM, AsyncLLM]
        self.executor: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(max_workers=1)
        self.pending_size = 1
        self.recv_manager = RecvBucketManager()
        self.command_queue: Optional[queue.Queue] = None

        self.request_metas = {}
        self.group_name = "vllm_worker_default"
        self.running = False
        self.in_migration_state = False 

    def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)
        vllm_config = copy.deepcopy(self.worker_config.strategy_args.strategy_config)
        self.vllm_config = vllm_config
        engine_mode = vllm_config.pop("engine_mode", "sync")  # sync/async
        self.pending_size = vllm_config.pop("pending_size", 1)
        self.sleep_level = vllm_config.pop("sleep_level", 1)
        self.command_queue = queue.Queue()

        if self.worker_config.model_args.dtype == "fp32":
            dtype = "float32"
        elif self.worker_config.model_args.dtype == "fp16":
            dtype = "float16"
        elif self.worker_config.model_args.dtype == "bf16":
            dtype = "bfloat16"
        else:
            dtype = "auto"
        vllm_config.update(
            {
                "model": self.worker_config.model_args.model_name_or_path,
                "dtype": dtype,
                "enforce_eager": vllm_config.get("enforce_eager", False),
                "trust_remote_code": True,
                "seed": self.worker.pipeline_config.seed,
                "disable_custom_all_reduce": vllm_config.get(
                    "disable_custom_all_reduce", True
                ),  # potentially hangs in tp>1
                "enable_prefix_caching": vllm_config.get("enable_prefix_caching", False),
                "max_num_seqs": vllm_config.get("max_num_seqs", 32),
                "load_format": vllm_config.get("load_format", "dummy"),  # use model update passed value
                "disable_log_stats": eval(str(vllm_config.get("disable_log_stats", False))),
                "compilation_config":vllm.config.CompilationConfig(),
            }
        )

        # support speculative generation
        self.other_gen_configs = dict()
        if vllm_config.get("enable_speculative") is not None:
            # remove uncessary speculative kwargs if not enabled
            # import pdb; pdb.set_trace()
            if not eval(str(vllm_config.get("enable_speculative"))):
                for key in ['speculative_model', 'speculative_method', 'num_speculative_tokens', 'ngram_prompt_lookup_max', 'prompt_lookup_max', 'prompt_lookup_min', 'method', 'skip_first_k_speculative_tokens', 'enable_suffix_decoding', 'additional_temperature']:
                    if key in vllm_config:
                        vllm_config.pop(key, None)
            else:
                if "0.8.4" in vllm.__version__ or '0.8.5.dev' in vllm.__version__:
                    speculative_config = {
                        "method": vllm_config.pop("method", None), 
                        "num_speculative_tokens": vllm_config.pop("num_speculative_tokens", 3),
                        "prompt_lookup_max": vllm_config.pop("prompt_lookup_max", 3),
                        "prompt_lookup_min": vllm_config.pop("prompt_lookup_min", 3),
                    }
                    
                    if skip_first_k_speculative_tokens := vllm_config.pop("skip_first_k_speculative_tokens", None): 
                        self.other_gen_configs['skip_first_k_speculative_tokens'] = skip_first_k_speculative_tokens
                    if additional_temperature := vllm_config.pop("additional_temperature", None):
                        self.other_gen_configs["additional_temperature"] = additional_temperature
                    
                    if enable_suffix_decoding := eval(str(vllm_config.get("enable_suffix_decoding", False))):
                        speculative_config["enable_suffix_decoding"] = enable_suffix_decoding
                        vllm_config.pop("enable_suffix_decoding")
                    vllm_config['speculative_config'] = speculative_config
                    # import pdb; pdb.set_trace()
                else:
                    raise NotImplementedError(
                        "vllm_strategy does not support speculative generation, please set enable_speculative to False"
                    )
            vllm_config.pop("enable_speculative", None)


        self.is_lora = self.worker_config.model_args.lora_target is not None
        if self.is_lora:
            lora_kwargs = {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": self.worker_config.model_args.lora_rank,
            }
            vllm_config.update(lora_kwargs)
            vllm_config["load_format"] = "auto"  # enables vLLM to load the base model for add_lora
        logger.info(f"vllm_config: {vllm_config}")
        assert not dist.is_initialized()

        # set VLLM_PORT to avoid port conflict applied by vllm
        vllm_port = self.worker.get_free_port()
        os.environ["VLLM_PORT"] = str(vllm_port)

        if engine_mode == "sync":
            self.model = LLM(resource_placement_groups=self.worker_config.resource_placement_groups, **vllm_config)
            self.tokenizer = self.model.get_tokenizer()
        else:
            self.model = AsyncLLM(
                resource_placement_groups=self.worker_config.resource_placement_groups, **vllm_config
            )
            loop = asyncio.get_event_loop()
            self.tokenizer = loop.run_until_complete(self.model.get_tokenizer())
        additional_special_tokens = self.tokenizer.additional_special_tokens
        special_tokens = [
            add_token
            for add_token in self.tokenizer.added_tokens_decoder.values()
            if add_token.special and add_token.content not in additional_special_tokens
        ]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}, replace_additional_special_tokens=False
        )
        logger.info(f"add {special_tokens} to additional_special_tokens: {self.tokenizer.additional_special_tokens}")

        self.worker.rank_info.dp_rank = self.worker.rank
        self.worker.rank_info.dp_size = self.worker.world_size
        collective.init_collective_group(
            world_size=self.worker.world_size,
            rank=self.worker.rank,
            group_name=self.group_name,
            master_addr=self.worker.master_addr,
            master_port=self.worker.master_port,
        )

    def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        vllm实现compute log probs在这里实现即可
        """
        pass

    def generate(self, batch: DataProto, generation_config) -> torch.Tensor:
        sampling_params = create_sampling_params_for_vllm(gen_kwargs=generation_config)

        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask

        vllm_input_args = {}
        if "multi_modal_data" in batch.non_tensor_batch:
            vllm_input_args["prompts"] = batch.non_tensor_batch["multi_modal_data"]
        else:
            vllm_input_args["prompt_token_ids"] = gather_unpadded_input_ids(
                input_ids=input_ids, attention_mask=attention_mask
            )

        lora_requests = None
        if self.is_lora:
            batch_size = len(input_ids)
            lora_int_ids = list(self.model.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(
                        lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="dummy_lora_path"
                    )
                ] * batch_size
        vllm_outputs = self.model.generate(
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=lora_requests,
            **vllm_input_args,
        )

        # (bs * num_return_sequences, max_response_len)
        output_ids = gather_outputs_to_pad_tensor(
            request_outputs=vllm_outputs,
            pad_token_id=self.tokenizer.pad_token_id,
            device=input_ids.device,
        )

        # (bs * num_return_sequences, input_len + max_response_len)
        output = concatenate_input_and_output(
            input_ids=input_ids, output_ids=output_ids, num_return_sequences=sampling_params.n
        )

        return output

    def process_vllm_output(self, vllm_outputs: List[RequestOutput], request_complete_callback):
        # 转成response id, request_complete_callback
        for request_output in vllm_outputs:
            output_token_ids = []
            request_id = request_output.request_id
            if request_id not in self.request_metas:
                continue
            for completion_output in request_output.outputs:
                output_token_ids.append(completion_output.token_ids)
            output_data = DataProto(meta_info=self.request_metas[request_id])
            output_data.meta_info["output_token_ids"] = output_token_ids
            request_complete_callback(data=output_data)
            # TODO: add total output tokens by @gw
            
    def reset_set_max_num_seqs(self, estimated_length):
        block_size = self.model.llm_engine.model_executor.cache_engine.block_manager.block_size
        total_num_blocks = self.model.llm_engine.model_executor.cache_engine.block_manager.num_gpu_blocks
        max_num_seqs = min(int(total_num_blocks * block_size / estimated_length / 1.2), self.vllm_config['max_num_seqs'])
        rank = self.worker.rank_info.dp_rank
        for sched in self.model.llm_engine.scheduler: 
            print(f'reset rank {rank} vllm\'s from {sched.scheduler_config.max_num_seqs} to {max_num_seqs}', flush=True)
            sched.scheduler_config.max_num_seqs = max_num_seqs

    def start_server(self, data: DataProto, request_complete_callback):
        self.running = True
        self.in_migration_state = False 
        rank = self.worker.rank_info.dp_rank
        with open(f"output/vllm_strategy_{rank}.txt", 'w') as f: 
            pass 

        while True:
            while not self.command_queue.empty():
                command, batch = self.command_queue.get_nowait()
                if self.in_migration_state: 
                    assert command in GenerateRequestType.STOP

                if command == GenerateRequestType.ADD:
                    input_ids = batch.batch["input_ids"]
                    attention_mask = batch.batch["attention_mask"]
                    request_id = batch.meta_info["request_id"]
                    self.request_metas[request_id] = batch.meta_info
                    generation_config = batch.meta_info.get("generation_config")
                    generation_config.update(self.other_gen_configs)                    
                    max_new_tokens = batch.meta_info.get("max_new_tokens", generation_config["max_new_tokens"])
                    max_new_tokens = min(max_new_tokens, generation_config["max_new_tokens"])
                    min_new_tokens = generation_config["min_new_tokens"]
                    if 'skip_first_k_speculative_tokens' in generation_config:
                        sampling_params = create_sampling_params_for_vllm_for_spec(
                            gen_kwargs={**generation_config, "max_new_tokens": max_new_tokens}
                        )
                        
                    else: 
                        domains = batch.non_tensor_batch.get("domain", [])
                        sampling_params = []
                        for i, domain in enumerate(domains):
                            gen_kwargs = {**generation_config, "max_new_tokens": max_new_tokens}
                            if domain == "code_sandbox":
                                gen_kwargs["min_new_tokens"] = 200
                            else:
                                gen_kwargs["min_new_tokens"] = min_new_tokens
                            
                            if os.environ.get("NO_MIN_LENGTH", "0") == "1":
                                gen_kwargs["min_new_tokens"] = min_new_tokens
                            
                            sampling_params.append(
                                create_sampling_params_for_vllm(gen_kwargs=gen_kwargs)
                            )
                        
                    
                    if "multi_modal_data" in batch.non_tensor_batch:
                        prompt_token_ids = [
                            batch.non_tensor_batch["multi_modal_data"][0]
                            ["prompt_token_ids"]
                        ]
                        multi_modal_data = [
                            batch.non_tensor_batch["multi_modal_data"][0]
                            ["multi_modal_data"]
                        ]
                    else:
                        prompt_token_ids = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
                        multi_modal_data = None
                    lora_requests = None
                    if self.is_lora:
                        batch_size = len(prompt_token_ids)
                        lora_int_ids = list(self.model.llm_engine.list_loras())
                        if len(lora_int_ids) > 0:
                            lora_int_id = lora_int_ids[0]
                            lora_requests = [
                                LoRARequest(
                                    lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="dummy_lora_path"
                                )
                            ] * batch_size
                    print(f"[VLLM STRATEGY] request {request_id} added into vllm engine!")
                    self.model.add_requests(
                        request_ids=[request_id],
                        prompt_token_ids=prompt_token_ids,
                        sampling_params=sampling_params,
                        multi_modal_data=multi_modal_data,
                        lora_requests=lora_requests,
                    )
                    # print(f"request {request_id} added to vllm_server, {time.time()}")
                elif command == GenerateRequestType.ABORT:
                    request_id = batch.meta_info["request_id"]
                    print(f"VLLM STRATEGY ABORT {request_id}")
                    self.model.abort_request(request_id=request_id)

                elif command == GenerateRequestType.MIGRATE: 
                    request_id = batch.meta_info["request_id"]
                    migrate_seqs = self.model.abort_request_for_migrate(request_id)
                    if envs.VLLM_USE_V1:
                        migrate_seqs = post_process_request_states(migrate_seqs)
                    else: 
                        migrate_seqs = post_process_seq_groups(migrate_seqs)
                    # print(f"partial_req_infos == {partial_req_infos}")
                    if migrate_seqs: 
                        migrate_callback_fn = batch.meta_info["migrate_callback_fn"]
                        migrate_callback_fn(data=migrate_seqs, request_id=request_id, dp_rank=batch.meta_info['dp_rank'])

                elif command == GenerateRequestType.MIGRATE_ALL:
                    migrate_seqs = self.model.abort_request_for_migrate(request_id=list(self.request_metas.keys()))
                    if envs.VLLM_USE_V1:
                        migrate_seqs = post_process_request_states(migrate_seqs)
                    else: 
                        migrate_seqs = post_process_seq_groups(migrate_seqs)
                    self.request_metas.clear()
                    if True:
                        while not self.command_queue.empty():
                            command, _batch = self.command_queue.get_nowait()
                            if command == GenerateRequestType.ADD: 
                                migrate_seqs.append(_batch)
                            elif command == GenerateRequestType.ABORT: 
                                request_id = _batch.meta_info["request_id"]
                                self.model.abort_request(request_id=request_id)
                            else:
                                print(f"command is {command} ")
                                raise NotImplementedError
                    if migrate_seqs: 
                        migrate_callback_fn = batch.meta_info["migrate_callback_fn"]
                        migrate_callback_fn(data=migrate_seqs, request_id=batch.meta_info['request_id'], dp_rank=batch.meta_info['dp_rank'])
                        continue 
                    

                elif command == GenerateRequestType.RESUME: 
                    input_ids = batch.batch["input_ids"]
                    request_id = batch.meta_info["request_id"]
                    self.request_metas[request_id] = batch.meta_info
                    generation_config = batch.meta_info.get("generation_config")
                    max_new_tokens = batch.meta_info.get("max_new_tokens", generation_config["max_new_tokens"])
                    max_new_tokens = min(max_new_tokens, generation_config["max_new_tokens"])
                    sampling_params = create_sampling_params_for_vllm(
                        gen_kwargs={**generation_config, "max_new_tokens": max_new_tokens}
                    )
                    rank = self.worker.rank_info.dp_rank
                    with open(f"output/vllm_strategy_{rank}.txt", 'a+') as f: 
                        f.write(f"request id resume in {request_id}, {rank}, {sampling_params}, {input_ids.shape}\n")
                    # print(f"request id resume in {request_id}, {rank}", flush=True)
                    
                    # print(f"request id resume in {request_id}, {self.worker.rank_info.dp_rank}", flush=True)
                    prompt_token_ids = [ids.tolist() for ids in input_ids]

                    if "multi_modal_data" in batch.non_tensor_batch:
                        multi_modal_data = [
                            batch.non_tensor_batch["multi_modal_data"][0]
                            ["multi_modal_data"]
                        ]
                    else:
                        multi_modal_data = None

                    lora_requests = None
                    if self.is_lora:
                        batch_size = len(prompt_token_ids)
                        lora_int_ids = list(self.model.llm_engine.list_loras())
                        if len(lora_int_ids) > 0:
                            lora_int_id = lora_int_ids[0]
                            lora_requests = [
                                LoRARequest(
                                    lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="dummy_lora_path"
                                )
                            ] * batch_size

                    self.model.add_requests(
                        request_ids=[request_id], 
                        prompt_token_ids=prompt_token_ids, 
                        sampling_params=sampling_params, 
                        multi_modal_data=multi_modal_data, # not support in default
                        lora_requests=lora_requests,
                    )
                elif command == GenerateRequestType.STOP:
                    if False: # self.vllm_config.get("speculative_config", None):
                        actual_num_accepted_tokens = self.model.llm_engine.model_executor.driver_worker.worker.spec_decode_sampler.actual_num_accepted_tokens
                        num_draft_tokens = self.model.llm_engine.model_executor.driver_worker.worker.spec_decode_sampler.num_draft_tokens
                        acceptance_rate = actual_num_accepted_tokens / num_draft_tokens
                        
                        # total_num_output_tokens = None 
                        # acceptance_rate_per_output = actual_num_accepted_tokens / total_num_output_tokens
                        
                        with open('output/generation.txt', 'a') as f: 
                            f.write(f'actual_num_accepted_tokens: {actual_num_accepted_tokens}, num_draft_tokens: {num_draft_tokens}\n')
                            f.write(f'actual_acceptance_rate: {acceptance_rate}\n\n')
                        
                        stat_loggers = getattr(self.model.llm_engine, "stat_loggers", None)
                        if False: # stat_loggers: deprecated 
                            stat_logger = stat_loggers["prometheus"]
                            acceptance_rate = (stat_logger.metrics.gauge_spec_decode_draft_acceptance_rate.labels(**stat_logger.labels)._value.get())
                            draft_tokens = stat_logger.metrics.counter_spec_decode_num_draft_tokens.labels(**stat_logger.labels)._value.get()
                            accepted_tokens = stat_logger.metrics.counter_spec_decode_num_accepted_tokens.labels(**stat_logger.labels)._value.get()
                            emitted_tokens = stat_logger.metrics.counter_spec_decode_num_emitted_tokens.labels(**stat_logger.labels)._value.get()
                            with open('output/generation.txt', 'a') as f: 
                                f.write(f"acceptance_rate: {acceptance_rate}, draft_tokens: {draft_tokens}, accepted_tokens: {accepted_tokens}, emitted_tokens: {emitted_tokens}\n")
                            logger.info('acceptance_rate %f, draft_tokens %d, accepted_tokens %d, emitted_tokens %d', acceptance_rate, draft_tokens, accepted_tokens, emitted_tokens)
                            # assert True, "speculative generation not supported in vllm_strategy"
                    if self.in_migration_state: 
                        migrate_seqs = self.model.abort_request_for_migrate(request_id=list(self.request_metas.keys()))
                        assert len(migrate_seqs) == 0, 'vllm strategy should clean seq groups before clean all'
                    else: 
                        self.model.abort_request(request_id=list(self.request_metas.keys()))
                    
                    self.request_metas.clear()
                    while not self.command_queue.empty():
                        self.command_queue.get_nowait()
                    # Run llm_engine again to consume all out standing requests and
                    # stop model execute loop, otherwise collective_rpc will stuck by
                    # model execute loop or there will be garbage output at next step.
                    self.model.clear_unfinished_requests()
                    self.model.llm_engine.reset_iteration_stats()
                    self.running = False
                    self.in_migration_state = False 
                    return

            vllm_outputs: List[RequestOutput] = self.model.fetch_output()
            self.process_vllm_output(vllm_outputs=vllm_outputs, request_complete_callback=request_complete_callback)

    def add_request(self, command, data: DataProto):
        self.command_queue.put((command, data))

    
    def add_batched_request(self, command, data_list: List[DataProto]):
        for data in data_list:
            self.command_queue.put((command, data))

    async def async_generate(self, batch: DataProto, generation_config: Dict) -> torch.Tensor:
        # TODO: refactor async_generate interface. not supported now!
        raise NotImplementedError()
        from vllm.inputs.data import TokensPrompt

        sampling_params = create_sampling_params_for_vllm(gen_kwargs=generation_config)

        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask
        assert input_ids.size(0) == 1, f"async_generate: batch['input_ids'] must have exactly one batch dimension"

        prompt_token_ids = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)

        # TODO meaningful request id?
        #   async_generate如何实现abort_request
        request_id = random_uuid()
        result_generator = self.model.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_token_ids[0]),
            sampling_params=sampling_params,
            request_id=request_id,
        )
        vllm_output: Optional[RequestOutput] = None
        async for request_output in result_generator:
            vllm_output = request_output
        assert vllm_output is not None

        # (bs * num_return_sequences, max_response_len)
        output_ids = gather_outputs_to_pad_tensor(
            request_outputs=[vllm_output], pad_token_id=self.tokenizer.pad_token_id, device=input_ids.device
        )
        # (bs * num_return_sequences, input_len + max_response_len)
        output = concatenate_input_and_output(
            input_ids=input_ids, output_ids=output_ids, num_return_sequences=sampling_params.n
        )
        return output

    # offload/reload 接口
    def load_states(self, *args, **kwargs):
        self.model.load_states()

    def offload_states(self, include=None, non_blocking=False):
        if include is None or OffloadStateType.model_params in include:
            self.model.offload_states(self.sleep_level)
        self.recv_manager.clear()
        gc.collect()
        torch.cuda.empty_cache()

    # 参数同步相关接口
    def setup_collective_group(self, model_update_name, comm_plan, backend="nccl"):
        self.model.setup_collective_group(comm_plan=comm_plan, backend=backend, rank_in_cluster=self.worker.rank)

    def broadcast_parameter(self, model_update_name, src_pp_rank, dtype, shape, parameter_name, is_lora=False):
        self.model.broadcast_parameter(src_pp_rank, dtype, shape, parameter_name, is_lora)

    def broadcast_bucket(self, model_update_name, src_pp_rank, meta_infos, bucket_size):
        self.model.broadcast_bucket(src_pp_rank, meta_infos, bucket_size)

    def update_parameter(self, model_update_name, parameter_name, weight, ranks_in_worker, is_lora=False):
        self.model.update_parameter(parameter_name, weight, ranks_in_worker, is_lora)

    def update_parameter_in_bucket(self, model_update_name, meta_infos, buffer, ranks_in_worker):
        self.model.update_parameter_in_bucket(meta_infos, buffer, ranks_in_worker)

    def add_lora(self, peft_config):
        self.model.add_lora(peft_config)


def gather_unpadded_input_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    gathered_input_ids = [ids[mask.bool()].tolist() for ids, mask in zip(input_ids, attention_mask)]
    return gathered_input_ids


def gather_outputs_to_pad_tensor(request_outputs: List["RequestOutput"], pad_token_id, device="cuda") -> torch.Tensor:
    token_ids_list_of_lists = [
        torch.tensor(completion_output.token_ids, device=device)
        for request_output in request_outputs
        for completion_output in request_output.outputs
    ]
    output_tensor = pad_sequence(token_ids_list_of_lists, batch_first=True, padding_value=pad_token_id)
    return output_tensor


def create_sampling_params_for_vllm_for_spec(gen_kwargs):
    if gen_kwargs["num_beams"] > 1:
        return SamplingParams(
            max_tokens=gen_kwargs["max_new_tokens"],
            stop_token_ids=gen_kwargs["eos_token_id"],
            repetition_penalty=gen_kwargs["repetition_penalty"],
            n=gen_kwargs["num_return_sequences"],
            best_of=gen_kwargs["num_beams"],
            use_beam_search=True,
            skip_first_k_speculative_tokens=gen_kwargs["skip_first_k_speculative_tokens"],
            additional_temperature=gen_kwargs["additional_temperature"],
            logprobs=0,

        )
    return SamplingParams(
        max_tokens=gen_kwargs["max_new_tokens"],
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        top_k=gen_kwargs["top_k"],
        stop_token_ids=gen_kwargs["eos_token_id"],
        repetition_penalty=gen_kwargs["repetition_penalty"],
        n=gen_kwargs["num_return_sequences"],
        skip_first_k_speculative_tokens=gen_kwargs["skip_first_k_speculative_tokens"],
        additional_temperature=gen_kwargs["additional_temperature"],
        logprobs=0,
    )


def create_sampling_params_for_vllm(gen_kwargs):
    if gen_kwargs["num_beams"] > 1:
        return SamplingParams(
            max_tokens=gen_kwargs["max_new_tokens"],
            # NOTE: debug for fixed size generation
            min_tokens=gen_kwargs["min_new_tokens"],
            stop_token_ids=gen_kwargs["eos_token_id"],
            repetition_penalty=gen_kwargs["repetition_penalty"],
            n=gen_kwargs["num_return_sequences"],
            best_of=gen_kwargs["num_beams"],
            use_beam_search=True,
            logprobs=0,
        )
    return SamplingParams(
        max_tokens=gen_kwargs["max_new_tokens"],
        # NOTE: debug for fixed size generation
        min_tokens=gen_kwargs["min_new_tokens"],
        # min_tokens=200,
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        top_k=gen_kwargs["top_k"],
        stop_token_ids=gen_kwargs["eos_token_id"],
        repetition_penalty=gen_kwargs["repetition_penalty"],
        n=gen_kwargs["num_return_sequences"],
        logprobs=0,
    )


def compare_sampling_params(params1: SamplingParams, params2: SamplingParams) -> bool:
    # 只比较采样参数的配置
    param_attrs = [
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "n",
        "stop_token_ids",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "min_p",
        "best_of",
        "stop",
        "ignore_eos",
        "use_beam_search",
        "best_of",
        "use_beam_search",
    ]

    # 比较每个采样参数
    for attr in param_attrs:
        if hasattr(params1, attr) and hasattr(params2, attr):
            val1 = getattr(params1, attr)
            val2 = getattr(params2, attr)
            if val1 != val2:
                print(f"采样参数 {attr} 不同: {val1} != {val2}")
                return False
    return True

def post_process_seq_groups(migrate_seq_groups):
    partial_req_infos = list() 
    for seq_group in migrate_seq_groups: 
        sequences = seq_group.get_seqs()
        for seq in sequences: 
            prompt_token_ids = seq.prompt_token_ids
            generated_token_ids = seq.get_token_ids()
            partial_req_infos.append({
                "request_id": seq_group.request_id, 
                "prompt_length": len(prompt_token_ids),
                "prompt_token_ids": prompt_token_ids, 
                "token_ids": generated_token_ids,
            })
    return partial_req_infos


from vllm.v1.engine.output_processor import RequestState

def post_process_request_states(migrate_request_states):
    partial_req_infos = list() 
    for request_state in migrate_request_states: 
        prompt_token_ids = request_state.prompt_token_ids
        generated_token_ids = request_state.detokenizer.output_token_ids
        partial_req_infos.append({
            "request_id": request_state.request_id, 
            "prompt_length": len(prompt_token_ids),
            "prompt_token_ids": prompt_token_ids, 
            "token_ids": generated_token_ids,
            })
    return partial_req_infos
