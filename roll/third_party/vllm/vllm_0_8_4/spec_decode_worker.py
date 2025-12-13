import gc
from typing import Optional
import copy

import torch
from vllm.spec_decode.spec_decode_worker import SpecDecodeWorker

# from roll.third_party.vllm.worker_helper import WorkerHelper
from roll.third_party.vllm.spec_decode_worker_helper import SpecDecodeWorkerHelper
from roll.utils.logging import get_logger

logger = get_logger()


from vllm.config import ParallelConfig, SpeculativeConfig, VllmConfig
from vllm.spec_decode.target_model_runner import TargetModelRunner
from vllm.utils import resolve_obj_by_qualname


def create_spec_worker_new(*args, **kwargs) -> "SpecDecodeWorker084":
    """Helper method that is the entrypoint for Executors which use
    WorkerWrapper. It constructs a SpecDecodeWorker from the speculative config.
    """
    vllm_config: VllmConfig = kwargs.get("vllm_config")
    speculative_config: SpeculativeConfig = vllm_config.speculative_config
    assert speculative_config is not None

    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        raise NotImplementedError("Speculative decoding is currently "
                                  "incompatible with pipeline parallelism")

    draft_worker_kwargs = kwargs.copy()

    kwargs["model_runner_cls"] = TargetModelRunner
    target_worker_config = copy.deepcopy(vllm_config)
    target_worker_config.parallel_config.worker_cls =\
        target_worker_config.parallel_config.sd_worker_cls
    cls = resolve_obj_by_qualname(
        target_worker_config.parallel_config.worker_cls)
    target_worker = cls(*args, **kwargs)
    # Set the disable_logprobs variable in the TargetModelRunner instance
    # as per its value specified in the SpeculativeConfig.
    target_worker.model_runner.disable_logprobs =\
         speculative_config.disable_logprobs

    draft_worker_config = copy.deepcopy(vllm_config)
    draft_worker_config.model_config = speculative_config.draft_model_config
    draft_worker_config.quant_config = VllmConfig._get_quantization_config(
        draft_worker_config.model_config,
        vllm_config.load_config,
    )
    speculative_config.draft_parallel_config.worker_cls =\
        draft_worker_config.parallel_config.sd_worker_cls
    draft_worker_config.parallel_config = speculative_config.draft_parallel_config  # noqa
    # TODO allow draft-model specific load config.

    # Override draft-model specific worker args.
    draft_worker_kwargs.update(
        vllm_config=draft_worker_config,
        ngram_prompt_lookup_max=speculative_config.prompt_lookup_max,
        ngram_prompt_lookup_min=speculative_config.prompt_lookup_min,
    )

    spec_decode_worker = SpecDecodeWorker084.create_worker(
        scorer_worker=target_worker,
        draft_worker_kwargs=draft_worker_kwargs,
        disable_mqa_scorer=speculative_config.disable_mqa_scorer,
        disable_by_batch_size=speculative_config.disable_by_batch_size,
        draft_token_acceptance_method=speculative_config.acceptance_method,
        typical_acceptance_sampler_posterior_threshold=speculative_config.
        posterior_threshold,
        typical_acceptance_sampler_posterior_alpha=speculative_config.
        posterior_alpha,
        disable_logprobs=speculative_config.disable_logprobs,
        disable_log_stats=speculative_config.disable_log_stats,
        num_speculative_tokens=speculative_config.num_speculative_tokens,
        skip_first_k_tokens_speculation=speculative_config.skip_first_k_tokens_speculation,
    )

    return spec_decode_worker

from vllm.platforms import current_platform
from vllm.worker.worker_base import LoRANotSupportedWorkerBase, WorkerBase
from vllm.spec_decode.ngram_worker import NGramWorker
from typing import Any, Dict, List, Optional, Set, Tuple, Type


if current_platform.is_cuda_alike():
    from vllm.spec_decode.draft_model_runner import TP1DraftModelRunner

from vllm.spec_decode.medusa_worker import MedusaWorker
from vllm.spec_decode.mlp_speculator_worker import MLPSpeculatorWorker
# from vllm.spec_decode.mqa_scorer import MQAScorer
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.ngram_worker import NGramWorker
from vllm.spec_decode.smaller_tp_proposer_worker import SmallerTpProposerWorker
from vllm.spec_decode.target_model_runner import TargetModelRunner
from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeBaseSampler, SpecDecodeStochasticBaseSampler)
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
# from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.layers.typical_acceptance_sampler import (
    TypicalAcceptanceSampler)

class SpecDecodeWorker084(SpecDecodeWorker, SpecDecodeWorkerHelper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # import pdb; pdb.set_trace()
        self.weight_loaded : bool = True
        self.kv_cache_loaded : bool = True


    @classmethod
    def create_worker(
        cls,
        scorer_worker: WorkerBase,
        draft_worker_kwargs: Dict[str, Any],
        disable_mqa_scorer: bool,
        disable_by_batch_size: Optional[int],
        draft_token_acceptance_method: str,
        typical_acceptance_sampler_posterior_threshold: float,
        typical_acceptance_sampler_posterior_alpha: float,
        disable_logprobs: bool,
        disable_log_stats: bool,
        num_speculative_tokens: int,
        skip_first_k_tokens_speculation: Optional[int],
    ) -> "SpecDecodeWorker":

        allow_zero_draft_token_step = True
        enable_lm_head_weight_load = False
        num_spec_prefill_steps = 1
        ngram_prompt_lookup_max = (
            draft_worker_kwargs.pop("ngram_prompt_lookup_max"))
        ngram_prompt_lookup_min = (
            draft_worker_kwargs.pop("ngram_prompt_lookup_min"))
        draft_model_config = draft_worker_kwargs["vllm_config"].model_config
        draft_parallel_config: ParallelConfig = draft_worker_kwargs[
            'vllm_config'].parallel_config
        if ngram_prompt_lookup_max > 0:
            draft_worker_kwargs[
                "device_type"] = scorer_worker.device_config.device.type
            proposer_worker = NGramWorker(**draft_worker_kwargs)
            proposer_worker.set_ngram_window_size(ngram_prompt_lookup_min,
                                                  ngram_prompt_lookup_max)
        else:
            draft_tp = draft_parallel_config.tensor_parallel_size
            target_tp = scorer_worker.parallel_config.tensor_parallel_size

            if draft_model_config.hf_config.model_type == "mlp_speculator":
                proposer_worker = MLPSpeculatorWorker(**draft_worker_kwargs)
            elif draft_model_config.hf_config.model_type == "medusa":
                proposer_worker = MedusaWorker(**draft_worker_kwargs)
            else:
                if draft_tp == 1:
                    if current_platform.is_cuda_alike():
                        draft_worker_kwargs[
                            "model_runner_cls"] = TP1DraftModelRunner
                else:
                    if draft_model_config.hf_config.model_type == "eagle":
                        raise NotImplementedError(
                            f"{draft_model_config.hf_config.model_type} "
                            "does not support TP > 1 yet")

                    allow_zero_draft_token_step = False

                # Load lm_head weight for eagle in init_device
                if draft_model_config.hf_config.model_type == "eagle":
                    enable_lm_head_weight_load = True

                proposer_worker = MultiStepWorker(**draft_worker_kwargs)
                if draft_model_config.hf_config.model_type == "deepseek_mtp":
                    num_spec_prefill_steps = \
                        draft_model_config.hf_config.n_predict

            proposer_worker = SmallerTpProposerWorker.maybe_wrap_worker(
                proposer_worker, draft_tp, target_tp)

        logger.info("Configuring SpecDecodeWorker with proposer=%s",
                    type(proposer_worker))

        spec_decode_sampler: SpecDecodeBaseSampler = None
        if draft_token_acceptance_method == "rejection_sampler":
            spec_decode_sampler = RejectionSampler()
        elif draft_token_acceptance_method == "typical_acceptance_sampler":
            spec_decode_sampler = TypicalAcceptanceSampler(
                posterior_threshold=\
                    typical_acceptance_sampler_posterior_threshold,
                posterior_alpha=typical_acceptance_sampler_posterior_alpha,
            )
        logger.info(
            "[Speculative Decoding] Configuring"
            " SpecDecodeWorker with sampler=%s", type(spec_decode_sampler))

        if not disable_mqa_scorer:
            if scorer_worker.model_runner.attn_backend.get_name(
            ) != "FLASH_ATTN":
                disable_mqa_scorer = True
                logger.info(
                    "[Speculative Decoding] Disabling MQA scorer as the "
                    "MQA is only available with flash attn backend.")

            if draft_model_config and \
                draft_model_config.max_model_len < \
                    scorer_worker.model_config.max_model_len:
                disable_mqa_scorer = True
                logger.info(
                    "[Speculative Decoding] Disabling MQA scorer as the "
                    "draft model max_model_len is smaller than the target "
                    "model max_model_len.")

            if not scorer_worker.model_runner.model_config.enforce_eager:
                disable_mqa_scorer = True
                logger.info(
                    "[Speculative Decoding] Disabling MQA scorer as the "
                    "target model is not running in eager mode.")

        return SpecDecodeWorker084(
            proposer_worker,
            scorer_worker,
            disable_mqa_scorer=disable_mqa_scorer,
            disable_logprobs=disable_logprobs,
            disable_log_stats=disable_log_stats,
            disable_by_batch_size=disable_by_batch_size,
            spec_decode_sampler=spec_decode_sampler,
            allow_zero_draft_token_step=allow_zero_draft_token_step,
            enable_lm_head_weight_load=enable_lm_head_weight_load,
            num_spec_prefill_steps=num_spec_prefill_steps,
            skip_first_k_tokens_speculation=skip_first_k_tokens_speculation)


    def reload_model(self):
        if not self.weight_loaded:
            self.scorer_worker.wake_up(["weights"])
            if self.proposer_worker is not None:
                try:
                    self.proposer_worker.wake_up(["weights"])
                except Exception as e:
                    pass
            self.weight_loaded = True

    def load_states(self):
        self.reload_model()
        if not self.kv_cache_loaded:
            self.scorer_worker.wake_up(["kv_cache"])
            if self.proposer_worker is not None:
                try:
                    self.proposer_worker.wake_up(["kv_cache"])
                except Exception as e:
                    pass

            self.kv_cache_loaded = True


    def offload_states(self):
        assert (self.weight_loaded and self.kv_cache_loaded) or (not self.weight_loaded and not self.kv_cache_loaded)
        if not self.weight_loaded:
            return
        self.scorer_worker.sleep()
        if self.proposer_worker is not None:
            try: 
                self.proposer_worker.sleep()
            except Exception as e:
                pass 
        self.weight_loaded = False
        self.kv_cache_loaded = False
        if hasattr(self, 'recv_manager'):
            self.recv_manager.clear()
        gc.collect()
        torch.cuda.empty_cache()
