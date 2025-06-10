"""
Patches for MoE models to support GBA quantization.
"""

from .qwen3_moe_patch import apply_qwen3_moe_patch, restore_qwen3_moe_patch
from .deepseek_v3_moe_patch import apply_deepseek_v3_moe_patch, restore_deepseek_v3_moe_patch

__all__ = [
    "apply_qwen3_moe_patch",
    "restore_qwen3_moe_patch",
    "apply_deepseek_v3_moe_patch",
    "restore_deepseek_v3_moe_patch"
]