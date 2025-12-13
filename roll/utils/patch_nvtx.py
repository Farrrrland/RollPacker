# model.py

import torch
import torch.nn as nn
from typing import Optional, Tuple
import torch.distributed as dist
from vllm.model_executor.models.qwen2 import Qwen2Attention, Qwen2MLP, Qwen2Model

from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer, WeightsMapper,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

# === NVTX Profiling Utilities ===
# --- Global flag to control when NVTX recording starts ---
ENABLE_NVTX_PROFILING = False

def enable_profiling():
    global ENABLE_NVTX_PROFILING
    ENABLE_NVTX_PROFILING = True

def disable_profiling():
    global ENABLE_NVTX_PROFILING
    ENABLE_NVTX_PROFILING = False

def maybe_push(name: str):
    global ENABLE_NVTX_PROFILING
    if ENABLE_NVTX_PROFILING:
        torch.cuda.nvtx.range_push(name)

def maybe_pop():
    global ENABLE_NVTX_PROFILING
    if ENABLE_NVTX_PROFILING:
        torch.cuda.nvtx.range_pop()
        

def patch_qwen2_attention_with_nvtx(attn: "Qwen2Attention", layer_id: int):
    original_forward = attn.forward

    def nvtx_forward(
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        layer_name = f"start/layer{layer_id}/attn"
        
        maybe_push(layer_name)
        
        # QKV Projection
        maybe_push(f"{layer_name}/qkv_proj")
        qkv, _ = attn.qkv_proj(hidden_states)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        maybe_pop()

        # Rotary Embedding
        maybe_push(f"{layer_name}/rotary")
        q, k = attn.rotary_emb(positions, q, k)
        maybe_pop()

        # Attention (PagedAttention / FlashAttention)
        maybe_push(f"{layer_name}/fmha")
        attn_output = attn.attn(q, k, v)
        maybe_pop()

        # O Projection
        maybe_push(f"{layer_name}/o_proj")
        output, _ = attn.o_proj(attn_output)
        maybe_pop()

        maybe_pop()  # layerX/attn
        return output

    attn.forward = nvtx_forward
    return attn


def patch_qwen2_mlp_with_nvtx(mlp: "Qwen2MLP", layer_id: int):
    original_forward = mlp.forward

    def nvtx_forward(x: torch.Tensor):
        layer_name = f"layer{layer_id}/mlp"
        
        maybe_push(layer_name)

        # Gate & Up Proj
        maybe_push(f"{layer_name}/gate_up_proj")
        gate_up, _ = mlp.gate_up_proj(x)
        maybe_pop()

        # Activation (SiLU)
        maybe_push(f"{layer_name}/silu")
        x = mlp.act_fn(gate_up)
        maybe_pop()

        # Down Proj
        maybe_push(f"{layer_name}/down_proj")
        x, _ = mlp.down_proj(x)
        maybe_pop()

        maybe_pop()  # layerX/mlp
        return x

    mlp.forward = nvtx_forward
    return mlp



# --- In patch_nvtx.py ---
from vllm.sequence import IntermediateTensors
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from typing import Iterable, Optional, Set, Tuple, Union


def patch_qwen2_model_with_nvtx(model: "Qwen2Model"):
    """
    Patch Qwen2Model.forward to add an NVTX range only around
    the loop over transformer layers (i.e., the main compute body).
    """
    original_forward = model.forward

    def nvtx_forward(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # === Input handling: no NVTX (embedding, residual setup) ===
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = model.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # === 🔥 PROFILING TARGET: Transformer layer loop ===
        maybe_push("start/layers")  # ← Only this part will be profiled

        for layer in model.layers[model.start_layer:model.end_layer]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        maybe_pop()  # end: start/layers
        # === End of main compute loop ===

        # === Final norm and return (no NVTX) ===
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = model.norm(hidden_states, residual)
        return hidden_states

    # Replace forward
    model.forward = nvtx_forward
    return model