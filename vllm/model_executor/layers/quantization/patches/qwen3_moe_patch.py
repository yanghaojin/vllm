import torch
import torch.nn.functional as F
from typing import Tuple

from vllm.logger import init_logger

logger = init_logger(__name__)

# Import Qwen3 MoE components - try to import, handle if not available
try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    QWEN3_MOE_AVAILABLE = True
except ImportError:
    Qwen3MoeSparseMoeBlock = None
    QWEN3_MOE_AVAILABLE = False


class QuantizedQwen3MoeSparseMoeBlock:
    """
    Qwen3MoeSparseMoeBlock forward method optimized for quantized models.
    Provides vectorized and micro-batched processing strategies.
    """

    @staticmethod
    def forward_vectorized(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantization-friendly MoE forward implementation using vectorized strategy.
        Best for medium to large batch sizes.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        total_tokens = batch_size * sequence_length

        # Flatten for processing
        hidden_states_flat = hidden_states.reshape(total_tokens, hidden_dim)

        # 1. Route calculation
        router_logits = self.gate(hidden_states_flat)
        if len(router_logits.shape) > 2:
            router_logits = router_logits.reshape(total_tokens, -1)

        # 2. Calculate routing weights
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # 3. Select top experts and weights
        routing_weights_topk, indices_topk = torch.topk(routing_weights, self.top_k, dim=1)

        # 4. Normalize top weights if required
        if self.norm_topk_prob:
            routing_weights_topk /= routing_weights_topk.sum(dim=1, keepdim=True)
        routing_weights_topk = routing_weights_topk.to(hidden_states.dtype)

        # 5. Pre-allocate expert output storage
        # Shape: [total_tokens, top_k, hidden_dim]
        expert_outputs = torch.zeros(
            total_tokens, self.top_k, hidden_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        # 6. Batch processing by experts - vectorized approach
        for expert_idx in range(self.num_experts):
            # Create expert mask [total_tokens, top_k]
            expert_mask = (indices_topk == expert_idx)

            if not expert_mask.any():
                continue

            # Find positions using current expert
            token_idx, topk_idx = torch.where(expert_mask)

            if len(token_idx) == 0:
                continue

            # Batch processing - key for quantization efficiency
            expert_inputs = hidden_states_flat[token_idx]
            expert_result = self.experts[expert_idx](expert_inputs)

            # Store results
            expert_outputs[token_idx, topk_idx] = expert_result

        # 7. Apply weights and sum
        # Expand weight dimension: [total_tokens, top_k, 1]
        weights_expanded = routing_weights_topk.unsqueeze(-1)

        # Weighted sum: [total_tokens, hidden_dim]
        final_hidden_states = (expert_outputs * weights_expanded).sum(dim=1)

        # 8. Reshape back to original shape
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits

    @staticmethod
    def forward_micro_batched(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantization-friendly MoE forward implementation using micro-batched strategy.
        Better for small batch sizes or memory-constrained scenarios.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        total_tokens = batch_size * sequence_length

        hidden_states_flat = hidden_states.reshape(total_tokens, hidden_dim)

        # Route calculation
        router_logits = self.gate(hidden_states_flat)
        if len(router_logits.shape) > 2:
            router_logits = router_logits.reshape(total_tokens, -1)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights_topk, indices_topk = torch.topk(routing_weights, self.top_k, dim=1)

        if self.norm_topk_prob:
            routing_weights_topk /= routing_weights_topk.sum(dim=1, keepdim=True)
        routing_weights_topk = routing_weights_topk.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states_flat)

        # Micro-batch processing - quantization friendly
        micro_batch_size = min(16, total_tokens)  # Small fixed batch size

        for start_idx in range(0, total_tokens, micro_batch_size):
            end_idx = min(start_idx + micro_batch_size, total_tokens)

            # Process token by token in micro batch for maximum quantization compatibility
            for token_idx in range(start_idx, end_idx):
                token_input = hidden_states_flat[token_idx:token_idx + 1]  # [1, hidden_dim]
                token_output = torch.zeros_like(token_input)

                # Process each selected expert for this token
                for expert_pos in range(self.top_k):
                    expert_idx = indices_topk[token_idx, expert_pos].item()
                    expert_weight = routing_weights_topk[token_idx, expert_pos].item()

                    # Skip very small weights for performance
                    if expert_weight < 1e-6:
                        continue

                    # Call expert - single token input for quantization stability
                    expert_output = self.experts[expert_idx](token_input)
                    token_output = token_output + expert_output * expert_weight

                final_hidden_states[token_idx] = token_output[0]

        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    @staticmethod
    def forward_adaptive(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive forward method that chooses between vectorized and micro-batched
        based on input characteristics.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        total_tokens = batch_size * sequence_length

        # Decision criteria for strategy selection
        if total_tokens >= 32:  # Use vectorized for larger batches
            return QuantizedQwen3MoeSparseMoeBlock.forward_vectorized(self, hidden_states)
        else:  # Use micro-batched for smaller batches
            return QuantizedQwen3MoeSparseMoeBlock.forward_micro_batched(self, hidden_states)


def apply_qwen3_moe_patch(strategy: str = "adaptive") -> bool:
    """
    Apply the quantization-friendly patch to Qwen3MoeSparseMoeBlock.

    Args:
        strategy: One of "vectorized", "micro_batched", or "adaptive"

    Returns:
        bool: True if patch applied successfully, False otherwise
    """
    if not QWEN3_MOE_AVAILABLE:
        logger.warning("Qwen3 MoE models not available in current transformers installation")
        return False

    try:
        # Save the original method (in case we need to restore it)
        if not hasattr(Qwen3MoeSparseMoeBlock, '_original_forward'):
            Qwen3MoeSparseMoeBlock._original_forward = Qwen3MoeSparseMoeBlock.forward

        # Select and apply the appropriate strategy
        if strategy == "vectorized":
            Qwen3MoeSparseMoeBlock.forward = QuantizedQwen3MoeSparseMoeBlock.forward_vectorized
            logger.info("Applied Qwen3 MoE vectorized quantization patch")
        elif strategy == "micro_batched":
            Qwen3MoeSparseMoeBlock.forward = QuantizedQwen3MoeSparseMoeBlock.forward_micro_batched
            logger.info("Applied Qwen3 MoE micro-batched quantization patch")
        elif strategy == "adaptive":
            Qwen3MoeSparseMoeBlock.forward = QuantizedQwen3MoeSparseMoeBlock.forward_adaptive
            logger.info("Applied Qwen3 MoE adaptive quantization patch")
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Must be one of: vectorized, micro_batched, adaptive")

        return True

    except Exception as e:
        logger.error(f"Failed to apply Qwen3 MoE quantization patch: {e}")
        return False


def restore_qwen3_moe_patch() -> bool:
    """
    Restore the original Qwen3MoeSparseMoeBlock forward method.

    Returns:
        bool: True if restoration successful, False otherwise
    """
    if not QWEN3_MOE_AVAILABLE:
        return False

    try:
        if hasattr(Qwen3MoeSparseMoeBlock, '_original_forward'):
            Qwen3MoeSparseMoeBlock.forward = Qwen3MoeSparseMoeBlock._original_forward
            delattr(Qwen3MoeSparseMoeBlock, '_original_forward')
            logger.info("Restored original Qwen3 MoE forward method")
            return True
        else:
            logger.warning("No original Qwen3 MoE forward method found to restore")
            return False

    except Exception as e:
        logger.error(f"Failed to restore Qwen3 MoE forward method: {e}")
        return False


def is_qwen3_moe_patched() -> bool:
    """
    Check if Qwen3 MoE patch is currently applied.

    Returns:
        bool: True if patch is applied, False otherwise
    """
    if not QWEN3_MOE_AVAILABLE:
        return False

    return hasattr(Qwen3MoeSparseMoeBlock, '_original_forward')


def get_qwen3_moe_info() -> dict:
    """
    Get information about Qwen3 MoE patch status.

    Returns:
        dict: Information about patch availability and status
    """
    return {
        "available": QWEN3_MOE_AVAILABLE,
        "patched": is_qwen3_moe_patched(),
        "strategies": ["vectorized", "micro_batched", "adaptive"]
    }