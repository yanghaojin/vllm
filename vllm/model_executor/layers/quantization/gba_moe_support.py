import json
import os
from typing import Dict, Any, Optional, List
from vllm.logger import init_logger

from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

import torch.nn as nn

logger = init_logger(__name__)


def detect_moe_model_type(config) -> Dict[str, Any]:
    """
    Detect MoE model type and return model information
    Enhanced version for vLLM compatibility
    """
    model_info = {
        'type': 'standard',
        'needs_patch': False,
        'experts_count': 0,
        'has_shared_experts': False,
        'routed_experts': 0,
        'shared_experts': 0,
        'use_individual_experts': True  # vLLM使用独立专家而非FusedMoE用于GBA
    }

    if not hasattr(config, 'model_type'):
        return model_info

    model_type = config.model_type.lower()

    # Check Qwen3 MoE - vLLM compatible
    if 'qwen3' in model_type:
        if getattr(config, 'num_experts', 0) > 0:
            model_info.update({
                'type': 'qwen3_moe',
                'needs_patch': False,  # 使用独立专家，不需要patch
                'experts_count': config.num_experts,
                'routed_experts': config.num_experts,
                'has_shared_experts': getattr(config, 'num_shared_experts', 0) > 0,
                'shared_experts': getattr(config, 'num_shared_experts', 0),
                'decoder_sparse_step': getattr(config, 'decoder_sparse_step', 1)
            })

    # Check DeepSeek V3 MoE - vLLM compatible
    elif 'deepseek' in model_type:
        # DeepSeek V3 has different attribute names
        routed_experts = getattr(config, 'n_routed_experts', 0)
        shared_experts = getattr(config, 'n_shared_experts', 0)

        if routed_experts > 0:
            model_info.update({
                'type': 'deepseek_v3_moe',
                'needs_patch': False,  # 使用独立专家
                'experts_count': routed_experts,
                'routed_experts': routed_experts,
                'has_shared_experts': shared_experts > 0,
                'shared_experts': shared_experts
            })

    # Check other MoE models (Mixtral, etc.) - vLLM compatible
    elif hasattr(config, 'num_local_experts') and config.num_local_experts > 0:
        model_info.update({
            'type': 'mixtral_moe',
            'needs_patch': False,  # vLLM handles Mixtral natively
            'experts_count': config.num_local_experts,
            'routed_experts': config.num_local_experts,
            'has_shared_experts': False,
            'shared_experts': 0
        })

    return model_info


def apply_moe_quant_strategy(name_attr: str, quant_strategy: Dict, moe_info: Dict[str, Any]) -> Optional[Dict]:
    """
    Apply quantization strategy for MoE models
    Enhanced version for vLLM with individual expert support
    """
    strategy = None

    # DeepSeek V2/V3 style attention projections (decomposed Q/K/V)
    deepseek_attention_mapping = {
        'q_a_proj': 'q_a_proj',
        'q_b_proj': 'q_b_proj',
        'kv_a_proj_with_mqa': 'kv_a_proj_with_mqa',
        'kv_b_proj': 'kv_b_proj'
    }

    for layer_name, strategy_key in deepseek_attention_mapping.items():
        if layer_name in name_attr:
            try:
                strategy = quant_strategy[strategy_key]
                return strategy
            except KeyError:
                pass

    # Standard attention projections (for backward compatibility)
    standard_attention_keys = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    for key in standard_attention_keys:
        if key in name_attr and not any(prefix in name_attr for prefix in ['q_a_', 'q_b_', 'kv_a_', 'kv_b_']):
            try:
                strategy = quant_strategy[key]
                return strategy
            except KeyError:
                pass

    # vLLM individual MoE gate layer handling
    # Pattern: mlp.gate.weight (vLLM uses ReplicatedLinear for gates)
    if ('mlp.gate.' in name_attr or name_attr.endswith('mlp.gate')) and 'experts' not in name_attr:
        try:
            strategy = quant_strategy['moe_gate']
            return strategy
        except KeyError:
            # Handle special case where moe_gate might not be quantized
            strategy = quant_strategy.get('moe_gate')
            if isinstance(strategy, dict) and 'desc' in strategy and 'Not quantized' in strategy['desc']:
                logger.info(f"Skipping quantization for MoE gate {name_attr} (marked as not quantized)")
                return None
            pass

    # vLLM individual MoE shared expert layers (DeepSeek V2/V3 specific)
    if 'mlp.shared_experts.' in name_attr or 'shared_experts.' in name_attr:
        if '.gate_proj' in name_attr or 'gate_proj' in name_attr:
            try:
                strategy = quant_strategy['moe_shared_expert_gate_proj']
                return strategy
            except KeyError:
                # Fallback to standard gate_proj
                try:
                    strategy = quant_strategy['gate_proj']
                    return strategy
                except KeyError:
                    pass
        elif '.up_proj' in name_attr or 'up_proj' in name_attr:
            try:
                strategy = quant_strategy['moe_shared_expert_up_proj']
                return strategy
            except KeyError:
                try:
                    strategy = quant_strategy['up_proj']
                    return strategy
                except KeyError:
                    pass
        elif '.down_proj' in name_attr or 'down_proj' in name_attr:
            try:
                strategy = quant_strategy['moe_shared_expert_down_proj']
                return strategy
            except KeyError:
                try:
                    strategy = quant_strategy['down_proj']
                    return strategy
                except KeyError:
                    pass

    # vLLM individual MoE expert layers - pattern: experts.X.gate_proj, experts.X.up_proj, experts.X.down_proj
    if 'experts.' in name_attr:
        if '.gate_proj' in name_attr:
            try:
                strategy = quant_strategy['moe_expert_gate_proj']
                return strategy
            except KeyError:
                # Fallback to standard gate_proj
                try:
                    strategy = quant_strategy['gate_proj']
                    return strategy
                except KeyError:
                    pass
        elif '.up_proj' in name_attr:
            try:
                strategy = quant_strategy['moe_expert_up_proj']
                return strategy
            except KeyError:
                try:
                    strategy = quant_strategy['up_proj']
                    return strategy
                except KeyError:
                    pass
        elif '.down_proj' in name_attr:
            try:
                strategy = quant_strategy['moe_expert_down_proj']
                return strategy
            except KeyError:
                try:
                    strategy = quant_strategy['down_proj']
                    return strategy
                except KeyError:
                    pass

    # Fallback to standard FFN layers (non-MoE layers)
    standard_ffn_keys = ['gate_proj', 'up_proj', 'down_proj']
    for key in standard_ffn_keys:
        if key in name_attr and 'experts' not in name_attr and 'shared_experts' not in name_attr:
            try:
                strategy = quant_strategy[key]
                return strategy
            except KeyError:
                pass

    # Additional fallback for other layer types
    fallback_keys = ['qkv_proj', 'gate_up_proj']
    for key in fallback_keys:
        if key in name_attr:
            try:
                strategy = quant_strategy[key]
                return strategy
            except KeyError:
                pass

    return strategy


def should_quantize_moe_layer(layer_name: str, layer_module, moe_info: Dict[str, Any]) -> bool:
    """
    Determine if a MoE layer should be quantized
    Enhanced for vLLM compatibility
    """

    # Only quantize Linear layers
    if not isinstance(layer_module, nn.Linear):
        return False

    # Skip embedding and normalization layers
    skip_patterns = [
        "embed_tokens", "embed_positions", "layernorm", "norm", "lm_head"
    ]

    layer_name_lower = layer_name.lower()
    if any(pattern in layer_name_lower for pattern in skip_patterns):
        return False

    # For vLLM MoE models, be more selective
    if moe_info['type'] != 'standard':
        # Skip very small layers
        if hasattr(layer_module, 'in_features') and hasattr(layer_module, 'out_features'):
            if layer_module.in_features < 32 or layer_module.out_features < 32:
                return False

        # For models with many experts, might want to skip some layers based on strategy
        if moe_info['experts_count'] > 50:  # Large expert count like DeepSeek V3
            # Could implement more selective quantization here based on strategy
            pass

    return True


def apply_moe_patches(moe_info: Dict[str, Any]) -> List[str]:
    """
    Apply patches for MoE models based on type.
    For vLLM with individual experts, patches are typically not needed.
    """
    applied_patches = []

    # 由于vLLM使用独立专家架构，通常不需要应用patches
    if not moe_info['needs_patch']:
        logger.info(f"No MoE patches needed for {moe_info['type']} (using individual experts)")
        return applied_patches

    # 如果确实需要patches，可以在这里添加
    logger.info(f"MoE patches not implemented for vLLM individual expert architecture")
    return applied_patches


def restore_moe_patches(applied_patches: List[str]):
    """Restore/cleanup applied MoE patches."""
    for patch_name in applied_patches:
        try:
            logger.info(f"Restored patch {patch_name} (no-op for vLLM individual experts)")
        except Exception as e:
            logger.warning(f"Failed to restore patch {patch_name}: {e}")