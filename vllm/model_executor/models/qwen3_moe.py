# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only Qwen3MoE model with FORCED separated architecture for GBA quantization."""
from collections.abc import Iterable
from typing import Any, Optional, Union
import logging

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear,
                                               ColumnParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (AutoWeightsLoader, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

logger = init_logger(__name__)


class Qwen3MoeMLP(nn.Module):
    """Traditional MLP for non-MoE layers - FORCED separated architecture"""

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            quant_config: Optional[QuantizationConfig] = None,
            reduce_results: bool = True,
            prefix: str = "",
    ) -> None:
        super().__init__()

        # FORCE separated architecture - NO fusion
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj")

        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj")

        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj")

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")

    def forward(self, x):
        # Manual SiLU(gate(x)) * up(x) implementation
        gate_output, _ = self.gate_proj(x)
        up_output, _ = self.up_proj(x)

        # Apply SiLU activation
        gate_output = torch.nn.functional.silu(gate_output)

        # Element-wise multiplication
        intermediate = gate_output * up_output

        # Down projection
        output, _ = self.down_proj(intermediate)
        return output


class Qwen3MoeExpert(nn.Module):
    """Individual MoE expert - FORCED separated architecture for GBA compatibility"""

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            quant_config: Optional[QuantizationConfig] = None,
            expert_id: int = 0,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.expert_id = expert_id

        # FORCE separated architecture - NO fusion
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj"
        )

        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj"
        )

        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj"
        )

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")

        logger.debug(f"Created SEPARATED expert {expert_id} at {prefix}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Manual SiLU(gate(x)) * up(x) implementation
        gate_output, _ = self.gate_proj(x)
        up_output, _ = self.up_proj(x)

        # Apply SiLU activation to gate output
        gate_output = torch.nn.functional.silu(gate_output)

        # Element-wise multiplication
        intermediate = gate_output * up_output

        # Down projection
        output, _ = self.down_proj(intermediate)
        return output


class Qwen3MoeSparseMoeBlock(nn.Module):
    """
    GBA-compatible Sparse MoE block using FORCED separated experts
    NO weight fusion allowed for GBA quantization compatibility
    """

    def __init__(
            self,
            config: PretrainedConfig,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.quant_config = quant_config

        if self.tp_size > self.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.num_experts}.")

        # Create individual experts - FORCED separated architecture
        self.experts = nn.ModuleList([
            Qwen3MoeExpert(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                expert_id=i,
                prefix=f"{prefix}.experts.{i}"
            )
            for i in range(self.num_experts)
        ])

        # Gate layer - also separated
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate"
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        GBA-compatible forward pass using ONLY separated experts
        Token-by-token processing for maximum quantization compatibility
        """
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Router computation (保持不变)
        router_logits, _ = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        routing_weights = routing_weights.to(hidden_states.dtype)

        expert_batch_inputs = {}
        expert_batch_info = {}

        # 第一遍：收集所有expert的输入
        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts == expert_idx)

            if not expert_mask.any():
                continue

            token_indices, expert_positions = torch.where(expert_mask)

            if len(token_indices) == 0:
                continue

            expert_inputs = hidden_states[token_indices]
            expert_weights = routing_weights[token_indices, expert_positions]

            # 批量存储该expert的所有输入
            expert_batch_inputs[expert_idx] = expert_inputs  # [N, hidden_dim]
            expert_batch_info[expert_idx] = {
                'token_indices': token_indices,
                'expert_positions': expert_positions,
                'weights': expert_weights
            }

        if not expert_batch_inputs:
            return torch.zeros_like(hidden_states).view(orig_shape)

        final_hidden_states = torch.zeros_like(hidden_states)

        # 第二遍：批量调用每个expert（大幅减少调用次数）
        for expert_idx, batch_inputs in expert_batch_inputs.items():
            info = expert_batch_info[expert_idx]

            # 关键：一次性处理该expert的所有输入（而非逐token）
            # 输入大小从 [1, hidden_dim] 变为 [N, hidden_dim]
            batch_outputs = self.experts[expert_idx](batch_inputs)  # 单次GBA调用处理N个token

            # 应用权重并累加到结果
            weighted_outputs = batch_outputs * info['weights'].unsqueeze(-1)

            # 分发回原始位置
            final_hidden_states.index_add_(0, info['token_indices'], weighted_outputs)

        return final_hidden_states.view(orig_shape)


class Qwen3MoeAttention(nn.Module):
    """FORCED separated attention architecture for GBA compatibility"""

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            rope_theta: float = 10000,
            rope_scaling: Optional[dict[str, Any]] = None,
            max_position_embeddings: int = 8192,
            head_dim: Optional[int] = None,
            rms_norm_eps: float = 1e-06,
            qkv_bias: bool = False,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # FORCE separated QKV - NO fusion allowed
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj")

        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj")

        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj")

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj")

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Use separated projections
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # Apply qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3MoeDecoderLayer(nn.Module):
    """Decoder layer with FORCED separated architecture"""

    def __init__(
            self,
            config: PretrainedConfig,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        # FORCED separated attention
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # Determine MLP type
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers)

        use_moe = (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0)

        if use_moe:
            self.mlp = Qwen3MoeSparseMoeBlock(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp")
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp")

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Qwen3MoeModel(nn.Module):
    """Model with FORCED separated architecture"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens")

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3MoeDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """SEPARATED architecture weight loading - NO fusion allowed"""

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        skipped_params: set[str] = set()

        # Collect weight information first
        all_weights = list(weights)
        weight_names = [name for name, _ in all_weights]

        # Check if weights are already separated
        has_separated_qkv = any('q_proj.qweight' in name for name in weight_names)
        has_separated_mlp = any('gate_proj.qweight' in name for name in weight_names)

        # Load weights directly - NO mapping/fusion
        for name, loaded_weight in all_weights:
            logger.debug(f"Processing weight: {name}, shape: {loaded_weight.shape}")

            # Skip bias for quantized models
            if name.endswith(".bias") and name not in params_dict:
                logger.debug(f"Skipping bias: {name}")
                continue

            # Skip PP missing parameters
            if is_pp_missing_parameter(name, self):
                logger.debug(f"Skipping PP missing: {name}")
                continue

            # Handle FP8 kv-scale remapping
            if name.endswith("kv_scale"):
                remapped_name = name.replace(".kv_scale", ".attn.kv_scale")
                if remapped_name in params_dict:
                    name = remapped_name
                else:
                    logger.warning(f"FP8 kv-scale not found: {name}")
                    continue

            # Direct loading - no fusion
            if name not in params_dict:
                # Check if it's a quantization-related parameter
                gba_suffixes = ['qweight', 'scales', 'zeros', 'q_perm', 'channel_scale', 'q_groups']
                is_quant_param = any(suffix in name for suffix in gba_suffixes)

                if is_quant_param:
                    logger.debug(f"Skipping quantization parameter: {name}")
                    skipped_params.add(name)
                    continue
                else:
                    logger.warning(f"Parameter not found in model: {name}")
                    skipped_params.add(name)
                    continue

            # Load the parameter
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)

            try:
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
                logger.debug(f"Successfully loaded: {name}")
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
                if any(suffix in name for suffix in gba_suffixes):
                    logger.warning(f"Continuing despite quantization error for {name}")
                    continue
                else:
                    raise

        if skipped_params:
            logger.debug(f"Skipped parameters: {sorted(list(skipped_params))}")

        return loaded_params


class Qwen3MoeForCausalLM(nn.Module, SupportsPP):
    """
    Main model class with FORCED separated architecture
    NO packed_modules_mapping to prevent fusion
    """

    # CRITICAL: Empty mapping to prevent any fusion attempts
    packed_modules_mapping = {}

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        # Force detection of separated architecture
        if quant_config and quant_config.get_name() == "gba":
            # Ensure no fusion is attempted
            self.__class__.packed_modules_mapping = {}

        self.model = Qwen3MoeModel(vllm_config=vllm_config,
                                   prefix=maybe_prefix(prefix, "model"))

        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config)

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Use AutoWeightsLoader with separated architecture"""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)