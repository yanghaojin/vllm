import re
from typing import Any, Dict, List, Optional, Union, Callable
import torch

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs
from vllm.logger import init_logger
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.gba_moe_support import apply_moe_quant_strategy
from vllm.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank


logger = init_logger(__name__)


class GBAConfig(QuantizationConfig):
    """GBA quantization configuration class"""

    def __init__(
            self,
            weight_bits: int = 4,
            group_size: int = 128,
            use_mbw: bool = False,
            strategy: Optional[Dict] = None,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.use_mbw = use_mbw
        self.strategy = strategy or {}

        self._moe_info = None
        self._moe_detected = False

        # Validate parameters
        if self.weight_bits not in [2, 3, 4, 5, 6, 8]:
            raise ValueError(f"Unsupported weight_bits: {self.weight_bits}")

        if self.group_size not in [-1, 32, 64, 128, 256]:
            raise ValueError(f"Unsupported group_size: {self.group_size}")

    @classmethod
    def get_name(cls) -> str:
        return "gba"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quant_strategy.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GBAConfig":
        """Create GBAConfig from configuration dictionary"""

        weight_bits = config.get("weight_bits", config.get("bits", 4))
        group_size = config.get("group_size", 128)
        use_mbw = config.get("use_mbw", False)
        strategy = config.get("strategy", {})

        # Handle model name based detection
        if "_name_or_path" in config:
            model_name = config["_name_or_path"]
            if "channel-mix" in model_name:
                use_mbw = True

            # Parse parameters from model name
            bpw_match = re.search(r'bpw-(\d+\.?\d*)', model_name)
            if bpw_match:
                weight_bits = int(float(bpw_match.group(1)))

            groupsize_match = re.search(r'groupsize(\d+)', model_name)
            if groupsize_match:
                group_size = int(groupsize_match.group(1))

        instance = cls(
            weight_bits=weight_bits,
            group_size=group_size,
            use_mbw=use_mbw,
            strategy=strategy,
        )

        # Add MoE info if present and apply patches
        if 'moe_info' in config:
            instance.moe_info = config['moe_info']
            instance._config_dict = config
            logger.debug(f"MoE info detected: {instance.moe_info}")

        return instance

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional[Union["GBALinearMethod"]]:
        """return quantization method"""

        # normal Linear
        if isinstance(layer, LinearBase):
            logger.debug(f"Creating GBA linear method for layer: {prefix}")
            return GBALinearMethod(self)

        # Handle MoE-related layers if MoE support is available
        moe_patterns = [
            'experts', 'gate_proj', 'up_proj', 'down_proj', 'mlp.gate',
            'shared_experts', 'moe_gate'
        ]

        if any(pattern in prefix for pattern in moe_patterns):
            logger.debug(f"Creating GBA linear method for MoE-related layer: {prefix}")
            return GBALinearMethod(self)

        logger.debug(f"No quantization method for layer: {prefix} (type: {type(layer).__name__})")
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


def flatten_x(x: torch.Tensor):
    """
    Flattens a 3D tensor into a 2D tensor by combining the first two dimensions.

    Args:
        x (torch.Tensor): A 3D tensor with shape [batch_size, seq_length, hidden_size].

    Returns:
        tuple[torch.Tensor, list]: A tuple containing the flattened 2D tensor with shape
        [batch_size * seq_length, hidden_size] and the original shape as a list
        [batch_size, seq_length] for later unflattening.
    """
    # shape of x in BERT/Transformer：[batch_size, seq_length, hidden_size]
    # flatten x to 2D tensor : [batch_size * seq_length, hidden_size]
    shape = list(x.size()[:-1])
    x = x.view(-1, x.size(-1))
    return x, shape


def unflatten_x(x: torch.Tensor, shape: list):
    """
    Unflattens a 2D tensor back into a 3D tensor using the original shape.

    Args:
        x (torch.Tensor): A 2D tensor with shape [batch_size * seq_length, output_size].
        shape (list): The original shape of the tensor before flattening,
        as a list [batch_size, seq_length].

    Returns:
        torch.Tensor: The unflattened 3D tensor with shape [batch_size, seq_length, output_size].
    """
    # from [batch_size * seq_length, output_size] to [batch_size, seq_length, output_size]
    x = x.view(shape + [x.size(-1)])
    return x


class GBALinearMethod(LinearMethodBase):
    """GBA quantization linear layer method"""

    def __init__(self, quant_config: GBAConfig):
        self.quant_config = quant_config
        logger.debug(
            f"Initialized GBA linear method with config: weight_bits={quant_config.weight_bits}, "
            f"group_size={quant_config.group_size}")

    def create_weights(
            self,
            layer: torch.nn.Module,
            input_size_per_partition: int,
            output_partition_sizes: List[int],
            input_size: int,
            output_size: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs,
    ) -> None:
        """Create GBA quantized weight parameters"""

        output_size_per_partition = sum(output_partition_sizes)

        gba_weight_loader = self._get_gba_weight_loader()

        # layer specific config
        layer_prefix = extra_weight_attrs.get("prefix", "")

        is_moe_expert = 'mlp.experts.' in layer_prefix and any(
            proj in layer_prefix for proj in ['gate_proj', 'up_proj', 'down_proj'])

        is_moe_gate = (layer_prefix.endswith('mlp.gate') or
                       'mlp.gate.' in layer_prefix) and 'experts' not in layer_prefix

        layer_config = self._get_layer_config(layer_prefix)

        weight_bits = layer_config.get("weight_bits", self.quant_config.weight_bits)
        group_size = layer_config.get("group_size", self.quant_config.group_size)

        if group_size == -1:
            group_size = input_size_per_partition

        num_groups = input_size_per_partition // group_size

        if not self.quant_config.use_mbw:
            # Standard quantization mode
            packed_rows = input_size_per_partition * weight_bits // 32
            qweight_shape = (packed_rows, output_size_per_partition)
        else:
            # Mixed bit-width mode
            qweight_shape = (input_size_per_partition // 32, output_size_per_partition)

        # Create quantization scales and zero points
        scale_zero_shape = (num_groups, output_size_per_partition)

        # Create quantized weights
        qweight = torch.nn.Parameter(
            torch.empty(qweight_shape, dtype=torch.int32, device="cuda"),
            requires_grad=False,
        )
        qweight._param_name = "qweight"
        set_weight_attrs(qweight, {
            "input_dim": 0,
            "output_dim": 1,
            "weight_loader": gba_weight_loader
        })

        scales = torch.nn.Parameter(
            torch.empty(scale_zero_shape, dtype=params_dtype, device="cuda"),
            requires_grad=False,
        )
        scales._param_name = "scales"
        set_weight_attrs(scales, {
            "input_dim": 0,
            "output_dim": 1,
            "weight_loader": gba_weight_loader
        })

        zeros = torch.nn.Parameter(
            torch.empty(scale_zero_shape, dtype=params_dtype, device="cuda"),
            requires_grad=False,
        )
        zeros._param_name = "zeros"
        set_weight_attrs(zeros, {
            "input_dim": 0,
            "output_dim": 1,
            "weight_loader": gba_weight_loader
        })

        # Create permutation indices
        q_perm = torch.nn.Parameter(
            torch.empty(input_size_per_partition, dtype=torch.int16, device="cuda"),
            requires_grad=False,
        )
        q_perm._param_name = "q_perm"
        set_weight_attrs(q_perm, {
            "input_dim": 0,
            "output_dim": -1,
            "weight_loader": gba_weight_loader
        })

        channel_scale = torch.nn.Parameter(
            torch.ones((1, 1, input_size_per_partition), dtype=params_dtype, device="cuda"),
            requires_grad=False,
        )
        channel_scale._param_name = "channel_scale"
        set_weight_attrs(channel_scale, {
            "input_dim": 2,
            "output_dim": -1,
            "weight_loader": gba_weight_loader
        })
        layer.register_parameter("channel_scale", channel_scale)

        # Mixed bit-width mode requires additional parameters
        if self.quant_config.use_mbw:
            # Group information
            q_groups = torch.nn.Parameter(
                torch.empty(num_groups * 2, dtype=torch.int16, device="cuda"),
                requires_grad=False,
            )
            q_groups._param_name = "q_groups"
            set_weight_attrs(q_groups, {
                "input_dim": -1,
                "output_dim": -1,
                "weight_loader": gba_weight_loader
            })
            layer.register_parameter("q_groups", q_groups)

        # Group mapping and row information (created in prepare_weights)
        layer.register_buffer("q_group_map", torch.empty(0, dtype=torch.int32))
        layer.register_buffer("rows_info", torch.empty(0, dtype=torch.int32))

        # Register all parameters
        layer.register_parameter("qweight", qweight)
        layer.register_parameter("scales", scales)
        layer.register_parameter("zeros", zeros)
        layer.register_parameter("q_perm", q_perm)

        # Store configuration
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.group_size = group_size
        layer.weight_bits = weight_bits

        layer.is_moe_expert = is_moe_expert
        layer.is_moe_gate = is_moe_gate

        layer._gba_weights_initialized = True
        logger.debug(f"Successfully created GBA weights for {layer_prefix}")

    def _get_gba_weight_loader(self):
        """Get GBA-specific weight loader function """

        def gba_weight_loader_wrapper(param: torch.nn.Parameter, loaded_weight: torch.Tensor, *args, **kwargs):
            param_name = self._infer_param_name(param, args[0] if args else None)

            logger.debug(f"GBA weight loader: {param_name} - param: {param.shape}, loaded: {loaded_weight.shape}")

            return self._gba_weight_loader(param, loaded_weight, param_name)

        return gba_weight_loader_wrapper

    def _infer_param_name(self, param: torch.nn.Parameter, shard_id=None) -> str:

        if hasattr(param, '_param_name'):
            return param._param_name

        param_shape = param.shape
        param_dtype = param.dtype
        param_dim = param.dim()

        # (int32, 2D)
        if param_dtype == torch.int32 and param_dim == 2:
            return "qweight"

        # (int16, 1D)
        elif param_dtype == torch.int16:
            if param_dim == 1:
                return "q_perm"
            else:
                return "q_groups"

        # (float, 3D)
        elif param_dim == 3 and param_shape[0] == 1 and param_shape[1] == 1:
            return "channel_scale"

        # scales和zeros (float, 2D)
        elif param_dtype in [torch.float16, torch.bfloat16, torch.float32] and param_dim == 2:
            if not hasattr(self, '_scale_zero_counter'):
                self._scale_zero_counter = 0

            if self._scale_zero_counter % 2 == 0:
                self._scale_zero_counter += 1
                return "scales"
            else:
                self._scale_zero_counter += 1
                return "zeros"

        return f"unknown_{param_dim}d_{param_dtype}"

    def _gba_weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor, param_name: str) -> None:
        """GBA quantized weight loader implementation with debugging"""

        logger.debug(f"Loading GBA weight: {param_name}")
        logger.debug(f"  Param shape: {param.shape}, Loaded shape: {loaded_weight.shape}")

        def ensure_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            if tensor.dtype != dtype:
                logger.debug(f"Converting dtype from {tensor.dtype} to {dtype}")
                converted = tensor.to(dtype)

                if 'scale' in param_name.lower():
                    after_conversion_invalid = torch.any(converted <= 0)
                    before_conversion_invalid = torch.any(tensor <= 0)
                    if after_conversion_invalid and not before_conversion_invalid:
                        logger.error(f"Dtype conversion introduced invalid values in {param_name}!")
                        logger.error(f"Before: min={torch.min(tensor)}, max={torch.max(tensor)}")
                        logger.error(f"After: min={torch.min(converted)}, max={torch.max(converted)}")

                return converted
            return tensor

        def ensure_shape(tensor: torch.Tensor, target_shape: torch.Size, param_name: str) -> torch.Tensor:
            if tensor.shape == target_shape:
                return tensor

            if tensor.numel() == target_shape.numel():
                logger.debug(f"Reshaping {param_name}: {tensor.shape} -> {target_shape}")
                reshaped = tensor.view(target_shape)

                # check if reshape causing problem
                if 'scale' in param_name.lower():
                    after_reshape_invalid = torch.any(reshaped <= 0)
                    before_reshape_invalid = torch.any(tensor <= 0)
                    if after_reshape_invalid and not before_reshape_invalid:
                        logger.error(f"Reshape introduced invalid values in {param_name}!")

                return reshaped

            if param_name in ["qweight", "scales", "zeros"] and len(tensor.shape) == 2 and len(target_shape) == 2:
                tensor_h, tensor_w = tensor.shape
                target_h, target_w = target_shape

                if tensor_h == target_h and tensor_w != target_w:
                    tp_size = get_tensor_model_parallel_world_size()

                    if tensor_w * tp_size == target_w:
                        result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
                        tp_rank = get_tensor_model_parallel_rank()
                        start_idx = tp_rank * tensor_w
                        end_idx = start_idx + tensor_w
                        result[:, start_idx:end_idx] = tensor
                        return result
                    elif target_w * tp_size == tensor_w:
                        tp_rank = get_tensor_model_parallel_rank()
                        start_idx = tp_rank * target_w
                        end_idx = start_idx + target_w
                        result = tensor[:, start_idx:end_idx].contiguous()
                        return result

            logger.error(f"Cannot handle shape mismatch for {param_name}")
            logger.error(f"  Param shape: {target_shape} ({target_shape.numel()} elements)")
            logger.error(f"  Loaded shape: {tensor.shape} ({tensor.numel()} elements)")
            logger.error(f"  TP size: {get_tensor_model_parallel_world_size()}")
            logger.error(f"  TP rank: {get_tensor_model_parallel_rank()}")

            raise ValueError(
                f"Shape mismatch for {param_name}:\n"
                f"  Expected: {target_shape} ({target_shape.numel()} elements)\n"
                f"  Got: {tensor.shape} ({tensor.numel()} elements)\n"
                f"  This indicates a mismatch between model architecture and weight file."
            )

        try:
            loaded_weight = ensure_dtype(loaded_weight, param.dtype)
            shaped_weight = ensure_shape(loaded_weight, param.shape, param_name)

            param.data.copy_(shaped_weight)
            logger.debug(f"Successfully loaded {param_name}")

        except Exception as e:
            logger.error(f"Failed to load weight {param_name}: {e}")
            raise e

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Processing after weight loading"""

        # Check if weights have been initialized
        if not hasattr(layer, '_gba_weights_initialized'):
            logger.warning("GBA weights not properly initialized for layer")
            return

        # Check if processing has already been done
        if hasattr(layer, '_gba_weights_processed'):
            return

            # Check if necessary weights have been loaded
            required_weights = ["qweight", "scales", "zeros", "q_perm"]
            # channel_scale只对非MoE层是必需的
            if not layer.is_moe_expert and not layer.is_moe_gate:
                required_weights.append("channel_scale")

            for weight_name in required_weights:
                if not hasattr(layer, weight_name):
                    raise ValueError(f"Missing required weight: {weight_name}")

        logger.debug(f"Processing GBA weights for layer with input_size={layer.input_size_per_partition}")

        # Convert weight layout
        if self.quant_config.use_mbw:
            # Mixed bit-width mode
            if not hasattr(layer, "q_groups"):
                raise ValueError("Mixed bitwidth mode requires q_groups")

            qweight, rows = ops.gba_trans_qweight(
                layer.qweight,
                layer.q_groups,
                True,  # use_mbw=True
                layer.input_size_per_partition,
                layer.scales.size(0),
                self.quant_config.weight_bits,
            )
            layer.qweight.data = qweight
            if isinstance(rows, list) and len(rows) > 0:
                layer.rows_info = rows
            else:
                layer.rows_info = torch.empty(0, dtype=torch.int32, device=layer.qweight.device)

            # Create group mapping
            layer.q_group_map = ops.make_group_map(layer.q_groups, layer.qweight.size(0))

            logger.debug(f"Applied mixed bitwidth quantization: {len(rows)} row groups")

        else:
            # Standard quantization mode
            qweight, rows = ops.gba_trans_qweight(
                layer.qweight,
                torch.empty(1, dtype=torch.int16, device=layer.qweight.device),
                False,  # use_mbw=False
                layer.input_size_per_partition,
                layer.scales.size(0),
                self.quant_config.weight_bits,
            )
            layer.qweight.data = qweight

            if isinstance(rows, list) and len(rows) > 0:
                layer.rows_info = torch.tensor(rows, dtype=torch.int32, device=layer.qweight.device)
            else:
                layer.rows_info = torch.empty(0, dtype=torch.int32, device=layer.qweight.device)

            logger.debug("Applied standard quantization")

        # Mark as processed
        layer._gba_weights_processed = True

    def _get_layer_config(self, layer_prefix: str) -> Dict[str, Any]:

        if not hasattr(self.quant_config, 'strategy') or not self.quant_config.strategy:
            return {}

        strategy = self.quant_config.strategy

        moe_info = getattr(self.quant_config, 'moe_info', {'type': 'standard'})

        layer_strategy = apply_moe_quant_strategy(layer_prefix, strategy, moe_info)

        if layer_strategy:
            config = {}

            # extract weight_bits
            if 'bits' in layer_strategy and layer_strategy['bits']:
                if isinstance(layer_strategy['bits'], list):
                    config['weight_bits'] = layer_strategy['bits'][0]
                else:
                    config['weight_bits'] = layer_strategy['bits']

            # get group_size
            if 'group_size' in layer_strategy:
                group_size_config = layer_strategy['group_size']
                if isinstance(group_size_config, dict):
                    weight_bits_str = str(config.get('weight_bits', 4))
                    if weight_bits_str in group_size_config:
                        config['group_size'] = group_size_config[weight_bits_str]
                    else:
                        config['group_size'] = next(iter(group_size_config.values()))
                else:
                    config['group_size'] = group_size_config

            return config

        # fallback
        return self._get_layer_config_fallback(layer_prefix)

    def _get_layer_config_fallback(self, layer_prefix: str) -> Dict[str, Any]:

        if not hasattr(self.quant_config, 'strategy') or not self.quant_config.strategy:
            return {}

        strategy = self.quant_config.strategy

        parts = layer_prefix.split('.')
        layer_num = None
        proj_type = None

        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_num = int(parts[i + 1])
                    break
                except ValueError:
                    continue

        for part in parts:
            if part in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
                proj_type = part
                break

        if layer_num is not None and proj_type is not None:
            layer_key = f"model.layers.{layer_num}"

            if layer_key in strategy:
                layer_config = strategy[layer_key]

                if proj_type in layer_config:
                    proj_config = layer_config[proj_type]

                    config = {}

                    if 'bits' in proj_config and proj_config['bits']:
                        if isinstance(proj_config['bits'], list):
                            config['weight_bits'] = proj_config['bits'][0]
                        else:
                            config['weight_bits'] = proj_config['bits']

                    if 'group_size' in proj_config:
                        group_size_config = proj_config['group_size']
                        if isinstance(group_size_config, dict):
                            weight_bits_str = str(config.get('weight_bits', 4))
                            if weight_bits_str in group_size_config:
                                config['group_size'] = group_size_config[weight_bits_str]
                            else:
                                config['group_size'] = next(iter(group_size_config.values()))
                        else:
                            config['group_size'] = group_size_config

                    return config

        return {}

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Execute forward propagation with robust channel_scale handling"""

        # Ensure weights are processed
        if not hasattr(layer, '_gba_weights_processed'):
            self.process_weights_after_loading(layer)

        original_dtype = x.dtype
        original_shape = x.shape

        if x.dim() == 3:
            # 3D input [batch, seq_len, hidden_size]
            # Apply channel_scale (original method)
            if hasattr(layer, 'channel_scale') and layer.channel_scale is not None:
                channel_scale = layer.channel_scale
                if channel_scale.dtype != torch.float16:
                    channel_scale = channel_scale.to(torch.float16)
                x = x.mul(channel_scale)

            # Flatten 为 2D
            x, shape = flatten_x(x)

        elif x.dim() == 2:
            # 2D input [batch*seq_len, hidden_size]
            # Modify the shape of channel_scale to accommodate 2D input
            if hasattr(layer, 'channel_scale') and layer.channel_scale is not None:
                channel_scale = layer.channel_scale  # 原始形状 [1, 1, hidden_size]
                if channel_scale.dtype != torch.float16:
                    channel_scale = channel_scale.to(torch.float16)

                # 将 channel_scale 从 [1, 1, hidden_size] 转换为 [hidden_size]
                if channel_scale.dim() == 3:
                    channel_scale_2d = channel_scale.squeeze(0).squeeze(0)  # [hidden_size]
                elif channel_scale.dim() == 2:
                    channel_scale_2d = channel_scale.squeeze(0)  # [1, hidden_size] -> [hidden_size]
                else:
                    channel_scale_2d = channel_scale

                x = x.mul(channel_scale_2d)  # Broadcasting: [batch*seq_len, hidden_size] * [hidden_size]

            # For 2D input, set a fake shape for subsequent processing
            shape = [x.size(0)]

        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}, shape: {x.shape}")

        # Ensure data type
        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        # Prepare parameters
        q_group_map = getattr(layer, "q_group_map", None)
        rows_info = getattr(layer, "rows_info", None)
        rows_list = rows_info.tolist() if rows_info is not None and rows_info.numel() > 0 else []

        # Make sure the weight parameter type is correct
        scales = layer.scales.to(torch.float16) if layer.scales.dtype != torch.float16 else layer.scales
        zeros = layer.zeros.to(torch.float16) if layer.zeros.dtype != torch.float16 else layer.zeros

        # Verify shape matches
        expected_input_features = layer.qweight.size(0) * (32 // layer.weight_bits)
        actual_input_features = x.size(1)

        if expected_input_features != actual_input_features:
            raise RuntimeError(
                f"Shape mismatch in GBA layer: expected input features {expected_input_features}, got {actual_input_features}")

        # Call CUDA forward propagation
        output = ops.gba_linear_forward(
            x,
            layer.qweight,
            scales,
            zeros,
            layer.q_perm,
            layer.group_size,
            layer.weight_bits,
            self.quant_config.use_mbw,
            q_group_map,
            rows_list,
        )

        # Add bias
        if bias is not None:
            if bias.dtype != output.dtype:
                bias = bias.to(output.dtype)
            output = output + bias

        # Restore output shape
        if len(original_shape) == 3:
            # Original input is 3D, restore to 3D
            output = unflatten_x(output, shape)
        elif len(original_shape) == 2:
            # Original input is 2D, keep it 2D
            pass

        # Convert back to original data type
        if original_dtype != torch.float16:
            output = output.to(original_dtype)

        return output


class GBALinear(LinearBase):
    """GBA quantized linear layer"""

    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = True,
            skip_bias_add: bool = False,
            params_dtype: Optional[torch.dtype] = None,
            quant_config: Optional[GBAConfig] = None,
            prefix: str = "",
    ) -> None:

        if quant_config is None:
            quant_config = GBAConfig()

        super().__init__(input_size, output_size, bias, skip_bias_add, params_dtype)

        self.quant_config = quant_config
        self.quant_method = GBALinearMethod(quant_config)

    def create_weights(self, dtype: torch.dtype) -> None:
        self.quant_method.create_weights(
            self,
            self.input_size,
            [self.output_size],
            self.input_size,
            self.output_size,
            dtype,
        )

    def process_weights_after_loading(self) -> None:
        """Processing after weight loading"""
        self.quant_method.process_weights_after_loading(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = getattr(self, "bias", None)
        output = self.quant_method.apply(self, x, bias)

        if self.skip_bias_add and bias is not None:
            return output, bias
        return output