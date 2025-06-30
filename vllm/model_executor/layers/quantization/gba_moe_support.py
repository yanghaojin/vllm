import json
import os
from typing import Dict, Any, Optional, List
from vllm.logger import init_logger

from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

import torch.nn as nn

logger = init_logger(__name__)


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