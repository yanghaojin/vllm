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
        'use_vllm_fused_moe': True  # vLLM使用FusedMoE
    }

    # Check Qwen3 MoE - vLLM compatible
    if hasattr(config, 'model_type') and 'qwen3' in config.model_type.lower():
        if getattr(config, 'num_experts', 0) > 0:
            model_info.update({
                'type': 'qwen3_moe',
                'needs_patch': True,
                'experts_count': config.num_experts,
                'routed_experts': config.num_experts,
                'has_shared_experts': getattr(config, 'num_shared_experts', 0) > 0,
                'shared_experts': getattr(config, 'num_shared_experts', 0),
                'decoder_sparse_step': getattr(config, 'decoder_sparse_step', 1)
            })

    # Check DeepSeek V3 MoE - vLLM compatible
    elif hasattr(config, 'model_type') and 'deepseek' in config.model_type.lower():
        # DeepSeek V3 has different attribute names
        routed_experts = getattr(config, 'n_routed_experts', 0)
        shared_experts = getattr(config, 'n_shared_experts', 0)

        if routed_experts > 0:
            model_info.update({
                'type': 'deepseek_v3_moe',
                'needs_patch': True,
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


def get_disable_bias_for_moe(name_attr: str, model_type: str, moe_info: Dict[str, Any]) -> bool:
    """
    Enhanced bias handling for MoE models
    Compatible with vLLM's FusedMoE architecture
    """
    # Original Qwen2 exception handling
    MODEL_TYPE_QWEN2 = "qwen2"
    for key in ['q_proj', 'k_proj', 'v_proj']:
        if key in name_attr and model_type == MODEL_TYPE_QWEN2:
            return False

    # DeepSeek V3 decomposed attention layers
    if moe_info['type'] == 'deepseek_v3_moe':
        deepseek_attention_layers = ['q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'kv_b_proj']
        for key in deepseek_attention_layers:
            if key in name_attr:
                # For DeepSeek models, these layers typically don't use bias
                return True

    # vLLM FusedMoE expert layers
    if 'experts.' in name_attr and any(proj in name_attr for proj in ['gate_proj', 'up_proj', 'down_proj']):
        return True

    # Default behavior
    return True


def apply_moe_quant_strategy(name_attr: str, quant_strategy: Dict, moe_info: Dict[str, Any]) -> Optional[Dict]:
    """
    Apply quantization strategy for MoE models
    Enhanced version for vLLM with FusedMoE support
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

    # vLLM FusedMoE gate layer handling
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

    # vLLM FusedMoE shared expert layers (DeepSeek V2/V3 specific)
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

    # vLLM FusedMoE expert layers - pattern: experts.X.gate_proj, experts.X.up_proj, experts.X.down_proj
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
    Enhanced for vLLM compatibility
    """
    applied_patches = []

    if not moe_info['needs_patch']:
        return applied_patches

    try:
        if moe_info['type'] == 'qwen3_moe':
            from vllm.model_executor.layers.quantization.patches.qwen3_moe_patch import apply_qwen3_moe_patch

            # Choose strategy based on expert count and vLLM FusedMoE
            strategy = 'adaptive'  # Adaptive works well with vLLM FusedMoE

            if apply_qwen3_moe_patch(strategy):
                applied_patches.append('qwen3_moe')
                logger.info(f"Applied Qwen3 MoE patch with {strategy} strategy for vLLM")

        elif moe_info['type'] == 'deepseek_v3_moe':
            from vllm.model_executor.layers.quantization.patches.deepseek_v3_moe_patch import \
                apply_deepseek_v3_moe_patch

            # Use hybrid strategy for DeepSeek V3 due to large number of experts
            strategy = 'hybrid' if moe_info['experts_count'] > 50 else 'conservative'

            if apply_deepseek_v3_moe_patch(strategy):
                applied_patches.append('deepseek_v3_moe')
                logger.info(f"Applied DeepSeek V3 MoE patch with {strategy} strategy for vLLM")

    except Exception as e:
        logger.warning(f"Failed to apply MoE patches: {e}")
        restore_moe_patches(applied_patches)
        applied_patches = []

    return applied_patches


def restore_moe_patches(applied_patches: List[str]):
    """Restore/cleanup applied MoE patches."""
    for patch_name in applied_patches:
        try:
            if patch_name == 'qwen3_moe':
                from vllm.model_executor.layers.quantization.patches.qwen3_moe_patch import restore_qwen3_moe_patch
                restore_qwen3_moe_patch()
            elif patch_name == 'deepseek_v3_moe':
                from vllm.model_executor.layers.quantization.patches.deepseek_v3_moe_patch import \
                    restore_deepseek_v3_moe_patch
                restore_deepseek_v3_moe_patch()
        except Exception as e:
            logger.warning(f"Failed to restore patch {patch_name}: {e}")


def get_moe_layer_strategy_mapping(moe_info: Dict[str, Any]) -> Dict[str, str]:
    """
    Get layer strategy mapping for different MoE architectures
    Enhanced for vLLM compatibility
    """
    base_mapping = {
        "q_proj": "q_proj",
        "k_proj": "k_proj",
        "v_proj": "v_proj",
        "o_proj": "o_proj",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
    }

    if moe_info['type'] == 'deepseek_v3_moe':
        # Add DeepSeek V3 specific mappings
        deepseek_mapping = {
            "q_a_proj": "q_a_proj",
            "q_b_proj": "q_b_proj",
            "kv_a_proj_with_mqa": "kv_a_proj_with_mqa",
            "kv_b_proj": "kv_b_proj",
            "moe_shared_expert_gate_proj": "moe_shared_expert_gate_proj",
            "moe_shared_expert_up_proj": "moe_shared_expert_up_proj",
            "moe_shared_expert_down_proj": "moe_shared_expert_down_proj",
        }
        base_mapping.update(deepseek_mapping)

    elif moe_info['type'] == 'qwen3_moe':
        # Add Qwen3 MoE specific mappings for vLLM
        qwen3_mapping = {
            "moe_gate": "moe_gate",
            "moe_expert_gate_proj": "moe_expert_gate_proj",
            "moe_expert_up_proj": "moe_expert_up_proj",
            "moe_expert_down_proj": "moe_expert_down_proj",
        }
        base_mapping.update(qwen3_mapping)

    return base_mapping


# vLLM integration utilities
def is_vllm_fused_moe_layer(layer_name: str) -> bool:
    """Check if this is a vLLM FusedMoE layer"""
    return 'experts.' in layer_name and any(proj in layer_name for proj in ['gate_proj', 'up_proj', 'down_proj'])


def get_vllm_expert_id_from_name(layer_name: str) -> Optional[int]:
    """Extract expert ID from vLLM layer name like 'experts.0.gate_proj'"""
    import re
    match = re.search(r'experts\.(\d+)\.', layer_name)
    return int(match.group(1)) if match else None


def fix_qwen3_rope_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix the RoPE configuration issue of the Qwen3 model
    Direct port from green-bit-llm
    """
    if 'rope_scaling' in config_dict:
        rope_scaling = config_dict['rope_scaling']

        if rope_scaling is None:
            return config_dict

        if isinstance(rope_scaling, dict):
            # Handle missing rope_type
            if 'rope_type' not in rope_scaling:
                if 'type' in rope_scaling:
                    # If there is a 'type' field, rename it to 'rope_type'
                    rope_scaling['rope_type'] = rope_scaling.pop('type')
                else:
                    # If none, add a default rope_type
                    rope_scaling['rope_type'] = 'default'

            # Make sure the rope_scaling dictionary is structured correctly
            config_dict['rope_scaling'] = rope_scaling
        else:
            # If rope_scaling is not a dictionary, set it to None
            config_dict['rope_scaling'] = None

    return config_dict


def load_config_with_rope_fix(model_path: str):
    """
    Safely load configurations and automatically fix RoPE configuration issues
    Direct port from green-bit-llm
    """

    try:
        # Try to load normally first
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return config
    except KeyError as e:
        if "rope_scaling" in str(e) and "rope_type" in str(e):
            logger.info(f"RoPE configuration problem detected, trying to repair...")

            # Read and repair the configuration file
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            # Fix RoPE configuration
            config_dict = fix_qwen3_rope_config(config_dict)

            # Create a config from the fixed dictionary
            model_type = config_dict.get('model_type', 'qwen3')

            if model_type in CONFIG_MAPPING:
                config_class = CONFIG_MAPPING[model_type]
                config = config_class.from_dict(config_dict)
            else:
                # If the model type is not in the mapping, try the generic method
                config = AutoConfig.from_dict(config_dict)

            logger.info(f"Successfully repaired RoPE configuration")
            return config
        else:
            raise e