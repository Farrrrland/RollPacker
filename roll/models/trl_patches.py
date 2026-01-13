from typing import Dict, Any, Optional

import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput

from trl import PreTrainedModelWrapper

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import logging


def value_head_load_state_dict(self: PreTrainedModelWrapper, state_dict: Dict[str, Any], strict=False) -> None:
    for name in list(state_dict.keys()):
        if name.startswith("v_head."):
            state_dict[name] = state_dict.pop(name)
        else:
            state_dict[name.replace("pretrained_model.", "")] = state_dict.pop(name)
    pretrained_model = getattr(self, "pretrained_model", None)
    if pretrained_model is not None:
        pretrained_model.load_state_dict(state_dict, strict=False)
        v_head: nn.Module = getattr(self, "v_head", None)
        if v_head is not None:
            for k in list(state_dict.keys()):
                if "v_head." in k:
                    state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
            v_head.load_state_dict(state_dict, strict=False)
    else:
        self.load_state_dict(state_dict, strict=False)


def token_classifier_forward(
    self: PreTrainedModelWrapper,
    input_ids=None,
    past_key_values=None,
    attention_mask=None,
    return_past_key_values=False,
    **kwargs,
) -> TokenClassifierOutput:
    kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
    kwargs["past_key_values"] = past_key_values

    if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
        kwargs.pop("past_key_values")

    base_model_output = self.pretrained_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **kwargs,
    )
    last_hidden_state = base_model_output.hidden_states[-1]
    if last_hidden_state.device != self.v_head.summary.weight.device:
        last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)

    value = self.v_head(last_hidden_state)

    return TokenClassifierOutput(
        loss=None,
        logits=value,
        hidden_states=base_model_output.hidden_states,
        attentions=base_model_output.attentions,
    )


def no_set_device_hook_post_init(self, state_dict):
    r"""
    We add the state dictionary of the value head to the state dictionary of the wrapped model
    by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
    keys of the value head state dictionary.
    """
    for k in list(state_dict.keys()):
        if "v_head." in k:
            state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
    self.v_head.load_state_dict(state_dict, strict=False)
    del state_dict

    if hasattr(self.pretrained_model, "hf_device_map"):
        if (
            "cpu" in self.pretrained_model.hf_device_map.values()
            or "disk" in self.pretrained_model.hf_device_map.values()
        ):
            raise ValueError(
                "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models."
            )

        # get the lm_head device
        for name, module in self.pretrained_model.named_modules():
            if any(attribute in name for attribute in self.lm_head_namings):
                lm_head_device = module.weight.device
                break

        # put v_head on the same device as the lm_head to avoid issues
        self.v_head = self.v_head.to(lm_head_device)

        def set_device_hook(module, input, outputs):
            r"""
            A hook that sets the device of the output of the model to the device of the first
            parameter of the model.

            Args:
                module (`nn.Module`):
                    The module to which the hook is attached.
                input (`tuple`):
                    The input to the module.
                outputs (`tuple`):
                    The output of the module.
            """
            if isinstance(outputs, dict):
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        outputs[k] = v.to(lm_head_device)
                new_output = outputs
            else:
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(lm_head_device),)
                    else:
                        new_output += (output,)
            return new_output

        self.register_forward_hook(set_device_hook)
        self.is_sequential_parallel = True


class LayerWise(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.cpu_state_dict = {}
        self.initialize_cpu_tensors()

    def initialize_cpu_tensors(self):
        for name, param in self.layer.named_parameters():
            self.cpu_state_dict[name] = param.data.detach().cpu().pin_memory()

    def to_gpu(self):
        for name, param in self.layer.named_parameters():
            param.data = self.cpu_state_dict[name].to("cuda")

    def to_cpu(self):
        for name, param in self.layer.named_parameters():
            param.data = self.cpu_state_dict[name]


def layerwise_forward_qwen2(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> BaseModelOutputWithPast:
    import os
    import math

    if not hasattr(self, "_upload_stream"):
        self._upload_stream = torch.cuda.Stream()
    upload_stream = self._upload_stream

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

    if inputs_embeds is None:
        inputs_embeds = self.optimized_embed_tokens.layer(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    position_embeddings = self.optimized_rotary_emb.layer(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    layer_by_layer_ratio = float(os.environ.get("LAYER_BY_LAYER_CACHE_RATIO", "0.0"))
    total_layers = len(self.optimized_layers)
    num_gpu_layers = int(math.ceil(total_layers * layer_by_layer_ratio))

    for i, optimized_layer in enumerate(self.optimized_layers):
        decoder_layer = optimized_layer.layer

        if i < num_gpu_layers:
            pass
        else:
            torch.cuda.current_stream().wait_stream(upload_stream)
            with torch.cuda.stream(upload_stream):
                optimized_layer.to_gpu()

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            raise NotImplementedError("Gradient checkpointing not supported in layerwise mode.")

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **flash_attn_kwargs,
        )
        hidden_states = layer_outputs[0]

        if i < num_gpu_layers:
            pass
        else:
            optimized_layer.to_cpu()

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.optimized_norm.layer(hidden_states)
    
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def layerwise_forward_qwen2_nopipeline(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> BaseModelOutputWithPast:
    import os
    import math

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

    if inputs_embeds is None:
        inputs_embeds = self.optimized_embed_tokens.layer(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    position_embeddings = self.optimized_rotary_emb.layer(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    layer_by_layer_ratio = float(os.environ.get("LAYER_BY_LAYER_CACHE_RATIO", "0.0"))
    total_layers = len(self.optimized_layers)
    num_gpu_layers = int(math.ceil(total_layers * layer_by_layer_ratio))

    for i, optimized_layer in enumerate(self.optimized_layers):
        decoder_layer = optimized_layer.layer

        if i < num_gpu_layers:
            pass
        else:
            optimized_layer.to_gpu()

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            raise NotImplementedError("Gradient checkpointing not supported in layerwise mode.")

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **flash_attn_kwargs,
        )
        hidden_states = layer_outputs[0]

        if i < num_gpu_layers:
            pass
        else:
            optimized_layer.to_cpu()

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.optimized_norm.layer(hidden_states)
    
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )



def layerwise_forward_qwen3(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> BaseModelOutputWithPast:
    upload_stream = torch.cuda.Stream()
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

    if inputs_embeds is None:
        inputs_embeds = self.optimized_embed_tokens.layer(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.optimized_rotary_emb.layer(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for i, optimized_layer in enumerate(self.optimized_layers):
        decoder_layer = optimized_layer.layer
        torch.cuda.current_stream().wait_stream(upload_stream)
        with torch.cuda.stream(upload_stream):
            optimized_layer.to_gpu()

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **flash_attn_kwargs,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.optimized_norm.layer(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
