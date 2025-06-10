import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Import DeepSeek V3 components - try to import, handle if not available
try:
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
    DEEPSEEK_V3_AVAILABLE = True
except ImportError:
    DeepseekV3MoE = None
    DEEPSEEK_V3_AVAILABLE = False


class QuantizedDeepSeekV3MoE:
    """
    DeepSeekV3MoE forward method optimized for quantized models.
    Provides multiple processing strategies for different scenarios.
    """

    @staticmethod
    def moe_token_by_token(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        """
        Quantization-friendly MoE implementation using token-by-token processing.
        Maximum compatibility with quantized models.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        total_tokens = batch_size * sequence_length

        # Flatten processing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Initialize output with correct dtype
        final_hidden_states = torch.zeros_like(hidden_states_flat, dtype=topk_weights.dtype)

        # Pre-convert to CPU to reduce GPU-CPU transfers if needed
        topk_indices_cpu = topk_indices.cpu().numpy()
        topk_weights_cpu = topk_weights.cpu().numpy()
        top_k = topk_indices.shape[-1]

        # Process token by token for maximum quantization compatibility
        for token_idx in range(total_tokens):
            # Get single token input - fixed batch size (1) for quantization stability
            token_input = hidden_states_flat[token_idx:token_idx + 1]  # [1, hidden_dim]
            token_output = torch.zeros_like(token_input, dtype=topk_weights.dtype)

            # Get expert indices and weights for this token
            token_experts = topk_indices_cpu[token_idx]  # [top_k]
            token_weights = topk_weights_cpu[token_idx]  # [top_k]

            # Process selected experts
            for expert_pos in range(top_k):
                expert_idx = int(token_experts[expert_pos])
                expert_weight = float(token_weights[expert_pos])

                # Skip very small weights for performance
                if expert_weight < 1e-6:
                    continue

                # Call expert network - fixed batch size, quantization friendly
                expert_layer = self.experts[expert_idx]
                expert_output = expert_layer(token_input)

                # Weighted accumulation - simple scalar operations
                token_output = token_output + expert_output * expert_weight

            # Store result
            final_hidden_states[token_idx] = token_output[0]

        # Convert back to original dtype and shape
        return final_hidden_states.type(hidden_states.dtype).view(batch_size, sequence_length, hidden_dim)

    @staticmethod
    def moe_vectorized(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        """
        Vectorized MoE implementation adapted for quantized models.
        Better performance for medium to large batch sizes.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        total_tokens = batch_size * sequence_length

        # Flatten processing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        top_k = topk_indices.shape[-1]

        # Pre-allocate expert output storage - key vectorization optimization
        expert_outputs = torch.zeros(
            total_tokens, top_k, hidden_dim,
            dtype=topk_weights.dtype,
            device=hidden_states.device
        )

        # Process experts in batches - adapted for DeepSeek V3's 256 experts
        for expert_idx in range(len(self.experts)):
            # Create expert mask [total_tokens, top_k]
            expert_mask = (topk_indices == expert_idx)

            if not expert_mask.any():
                continue

            # Find positions using current expert
            token_idx, topk_idx = torch.where(expert_mask)

            if len(token_idx) == 0:
                continue

            # Batch processing - key performance improvement
            expert_inputs = hidden_states_flat[token_idx]
            expert_result = self.experts[expert_idx](expert_inputs)

            # Store results
            expert_outputs[token_idx, topk_idx] = expert_result

        # Apply weights and sum - vectorized operations
        weights_expanded = topk_weights.unsqueeze(-1)
        final_hidden_states = (expert_outputs * weights_expanded).sum(dim=1)

        # Reshape back to original shape
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)

        # Convert back to original dtype
        return final_hidden_states.type(hidden_states.dtype)

    @staticmethod
    def should_use_vectorized(self, hidden_states):
        """
        Determine whether to use vectorized method based on batch size and memory efficiency.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        total_tokens = batch_size * seq_len
        top_k = getattr(self, 'top_k', 6)  # DeepSeek V3 default

        # Estimate memory requirement for expert_outputs tensor
        estimated_memory_mb = total_tokens * top_k * hidden_dim * 2 / (1024 * 1024)  # fp16

        # Primary criterion: batch size efficiency
        if total_tokens < 64:
            # Too small batch, vectorization has no advantage
            return False, estimated_memory_mb

        # Memory safety check for extreme cases
        try:
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated()
                available_memory_mb = (total_memory - allocated_memory) / (1024 * 1024)

                # Only fallback if memory is really insufficient (< 20% available)
                memory_threshold = available_memory_mb * 0.2

                if estimated_memory_mb > memory_threshold:
                    return False, estimated_memory_mb

        except Exception:
            # If cannot get memory info, don't restrict based on memory
            pass

        return True, estimated_memory_mb

    @staticmethod
    def forward_hybrid(self, hidden_states):
        """
        Hybrid strategy forward method for DeepSeek V3 MoE.
        Automatically chooses between vectorized and token-by-token processing.
        """
        # Save for residual connection and shared experts
        residuals = hidden_states

        # Route calculation - maintain DeepSeek V3's complex routing logic
        topk_indices, topk_weights = self.gate(hidden_states)

        # Choose strategy based on memory and batch size
        use_vectorized, estimated_mb = QuantizedDeepSeekV3MoE.should_use_vectorized(self, hidden_states)

        if use_vectorized:
            moe_output = QuantizedDeepSeekV3MoE.moe_vectorized(
                self, hidden_states, topk_indices, topk_weights
            )
        else:
            moe_output = QuantizedDeepSeekV3MoE.moe_token_by_token(
                self, hidden_states, topk_indices, topk_weights
            )

        # Add shared expert output - DeepSeek V3 specific feature
        if hasattr(self, 'shared_experts'):
            shared_expert_output = self.shared_experts(residuals)
            final_output = moe_output + shared_expert_output
        else:
            final_output = moe_output

        return final_output

    @staticmethod
    def forward_conservative(self, hidden_states):
        """
        Conservative forward method using only token-by-token processing.
        Maximum compatibility with quantized models.
        """
        residuals = hidden_states

        # Route calculation
        topk_indices, topk_weights = self.gate(hidden_states)

        # Use token-by-token method for maximum compatibility
        moe_output = QuantizedDeepSeekV3MoE.moe_token_by_token(
            self, hidden_states, topk_indices, topk_weights
        )

        # Add shared expert output if available
        if hasattr(self, 'shared_experts'):
            shared_expert_output = self.shared_experts(residuals)
            moe_output = moe_output + shared_expert_output

        return moe_output

    @staticmethod
    def forward_vectorized(self, hidden_states):
        """
        Vectorized forward method for maximum performance.
        Best for larger batch sizes with sufficient memory.
        """
        residuals = hidden_states

        # Route calculation
        topk_indices, topk_weights = self.gate(hidden_states)

        # Use vectorized method
        moe_output = QuantizedDeepSeekV3MoE.moe_vectorized(
            self, hidden_states, topk_indices, topk_weights
        )

        # Add shared expert output if available
        if hasattr(self, 'shared_experts'):
            shared_expert_output = self.shared_experts(residuals)
            moe_output = moe_output + shared_expert_output

        return moe_output


def apply_deepseek_v3_moe_patch(strategy: str = 'hybrid') -> bool:
    """
    Apply DeepSeek V3 MoE quantization patch.

    Args:
        strategy: One of 'conservative', 'vectorized', or 'hybrid'

    Returns:
        bool: True if patch applied successfully, False otherwise
    """
    if not DEEPSEEK_V3_AVAILABLE:
        logger.warning("DeepSeek V3 models not available in current transformers installation")
        return False

    try:
        # Save original method
        if not hasattr(DeepseekV3MoE, '_original_forward'):
            DeepseekV3MoE._original_forward = DeepseekV3MoE.forward

        # Apply strategy
        if strategy == 'conservative':
            DeepseekV3MoE.forward = QuantizedDeepSeekV3MoE.forward_conservative
            logger.info("Applied DeepSeek V3 conservative MoE quantization patch")
        elif strategy == 'vectorized':
            DeepseekV3MoE.forward = QuantizedDeepSeekV3MoE.forward_vectorized
            logger.info("Applied DeepSeek V3 vectorized MoE quantization patch")
        elif strategy == 'hybrid':
            DeepseekV3MoE.forward = QuantizedDeepSeekV3MoE.forward_hybrid
            logger.info("Applied DeepSeek V3 hybrid MoE quantization patch")
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Must be one of: conservative, vectorized, hybrid")

        return True

    except Exception as e:
        logger.error(f"Failed to apply DeepSeek V3 MoE quantization patch: {e}")
        return False


def restore_deepseek_v3_moe_patch() -> bool:
    """
    Restore original DeepSeek V3 MoE forward method.

    Returns:
        bool: True if restoration successful, False otherwise
    """
    if not DEEPSEEK_V3_AVAILABLE:
        return False

    try:
        if hasattr(DeepseekV3MoE, '_original_forward'):
            DeepseekV3MoE.forward = DeepseekV3MoE._original_forward
            delattr(DeepseekV3MoE, '_original_forward')
            logger.info("Restored original DeepSeek V3 MoE forward method")
            return True
        else:
            logger.warning("No original DeepSeek V3 MoE forward method found to restore")
            return False

    except Exception as e:
        logger.error(f"Failed to restore DeepSeek V3 MoE forward method: {e}")
        return False


def is_deepseek_v3_moe_patched() -> bool:
    """
    Check if DeepSeek V3 MoE patch is currently applied.

    Returns:
        bool: True if patch is applied, False otherwise
    """
    if not DEEPSEEK_V3_AVAILABLE:
        return False

    return hasattr(DeepseekV3MoE, '_original_forward')


def detect_deepseek_v3_moe_model(config) -> bool:
    """
    Detect if model is DeepSeek V3 MoE.

    Args:
        config: Model configuration object

    Returns:
        bool: True if this is a DeepSeek V3 MoE model
    """
    return (
        hasattr(config, 'model_type') and
        'deepseek' in config.model_type.lower() and
        (
            getattr(config, 'n_routed_experts', 0) > 0 or
            getattr(config, 'num_experts', 0) > 0
        )
    )


def get_deepseek_v3_moe_info() -> dict:
    """
    Get information about DeepSeek V3 MoE patch status.

    Returns:
        dict: Information about patch availability and status
    """
    return {
        "available": DEEPSEEK_V3_AVAILABLE,
        "patched": is_deepseek_v3_moe_patched(),
        "strategies": ["conservative", "vectorized", "hybrid"]
    }