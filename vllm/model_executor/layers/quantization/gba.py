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

        return instance

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional[Union["GBALinearMethod"]]:
        """return quantization method"""

        # normal Linear
        if isinstance(layer, LinearBase):
            return GBALinearMethod(self)

        # Handle MoE-related layers if MoE support is available
        moe_patterns = [
            'experts', 'gate_proj', 'up_proj', 'down_proj', 'mlp.gate',
            'shared_experts', 'moe_gate'
        ]

        if any(pattern in prefix for pattern in moe_patterns):
            return GBALinearMethod(self)

        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class GBALinearMethod(LinearMethodBase):
    """GBA quantization linear layer method"""

    def __init__(self, quant_config: GBAConfig):
        self.quant_config = quant_config

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
        """Create GBA quantized weight parameters with TP support"""

        output_size_per_partition = sum(output_partition_sizes)
        gba_weight_loader = self._get_gba_weight_loader()
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
            packed_rows = input_size_per_partition * weight_bits // 32
            qweight_shape = (packed_rows, output_size_per_partition)
        else:
            qweight_shape = (input_size_per_partition // 32, output_size_per_partition)

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

        # 🔧 q_perm: 确保大小正确对应 input_size_per_partition
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

        # 🔧 channel_scale: 确保大小正确对应 input_size_per_partition
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
            # 🔧 q_groups: 也可能需要按 TP 分割
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

        # Register buffers and parameters
        layer.register_buffer("q_group_map", torch.empty(0, dtype=torch.int32))
        layer.register_buffer("rows_info", torch.empty(0, dtype=torch.int32))

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

    def _get_gba_weight_loader(self):
        """Get GBA-specific weight loader function """

        def gba_weight_loader_wrapper(param: torch.nn.Parameter, loaded_weight: torch.Tensor, *args, **kwargs):
            param_name = self._infer_param_name(param, args[0] if args else None)
            return self._gba_weight_loader(param, loaded_weight, param_name)

        return gba_weight_loader_wrapper

    def _infer_param_name(self, param: torch.nn.Parameter, shard_id=None) -> str:
        """Improved parameter name inference for GBA"""

        if hasattr(param, '_param_name'):
            return param._param_name

        param_shape = param.shape
        param_dtype = param.dtype
        param_dim = param.dim()

        # Use parameter properties for reliable identification
        if param_dtype == torch.int32 and param_dim == 2:
            return "qweight"
        elif param_dtype == torch.int16:
            if param_dim == 1:
                # Distinguish between q_perm and q_groups by size
                if param_shape[0] % 2 == 0:
                    return "q_groups"  # Usually even numbers for group info
                else:
                    return "q_perm"  # Usually corresponds to input features
            else:
                return "q_groups"
        elif param_dim == 3 and param_shape[0] == 1 and param_shape[1] == 1:
            return "channel_scale"
        elif param_dtype in [torch.float16, torch.bfloat16, torch.float32] and param_dim == 2:
            # Use parameter address to distinguish scales vs zeros consistently
            param_id = hash(str(param.data.data_ptr()))  # More stable than id()
            if param_id % 2 == 0:
                return "scales"
            else:
                return "zeros"

        return f"unknown_{param_dim}d_{param_dtype}_{param_shape}"

    def _gba_weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor, param_name: str) -> None:
        """GBA quantized weight loader implementation with debugging"""

        def ensure_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            if tensor.dtype != dtype:
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

        # 在 gba.py 的 ensure_shape 函数中添加完整的 GBA tensor parallel 支持
        def ensure_shape(tensor: torch.Tensor, target_shape: torch.Size, param_name: str) -> torch.Tensor:
            """Corrected ensure_shape with proper GBA tensor parallel support"""

            if tensor.shape == target_shape:
                return tensor

            if tensor.numel() == target_shape.numel():
                logger.debug(f"Reshaping {param_name}: {tensor.shape} -> {target_shape}")
                reshaped = tensor.view(target_shape)

                # Check for numerical stability in scale parameters
                if 'scale' in param_name.lower():
                    after_reshape_invalid = torch.any(reshaped <= 0)
                    before_reshape_invalid = torch.any(tensor <= 0)
                    if after_reshape_invalid and not before_reshape_invalid:
                        logger.error(f"Reshape introduced invalid values in {param_name}!")

                return reshaped

            # Get tensor parallel info
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()

            logger.debug(
                f"Attempting TP split for {param_name}: {tensor.shape} -> {target_shape} (TP size: {tp_size}, rank: {tp_rank})")

            # === 1. qweight: 按第一个维度分割 ===
            if param_name == "qweight" and len(tensor.shape) == 2 and len(target_shape) == 2:
                tensor_h, tensor_w = tensor.shape
                target_h, target_w = target_shape

                # 输入维度分割: tensor_h = target_h * tp_size
                if tensor_h == target_h * tp_size and tensor_w == target_w:
                    start_idx = tp_rank * target_h
                    end_idx = start_idx + target_h
                    result = tensor[start_idx:end_idx, :].contiguous()
                    logger.debug(f"TP split qweight by rows: rank {tp_rank} gets [{start_idx}:{end_idx}, :]")
                    return result

                # 输出维度分割: tensor_w = target_w * tp_size
                elif tensor_h == target_h and tensor_w == target_w * tp_size:
                    start_idx = tp_rank * target_w
                    end_idx = start_idx + target_w
                    result = tensor[:, start_idx:end_idx].contiguous()
                    logger.debug(f"TP split qweight by cols: rank {tp_rank} gets [:, {start_idx}:{end_idx}]")
                    return result

                # 需要广播: target_w = tensor_w * tp_size
                elif tensor_h == target_h and target_w == tensor_w * tp_size:
                    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
                    start_idx = tp_rank * tensor_w
                    end_idx = start_idx + tensor_w
                    result[:, start_idx:end_idx] = tensor
                    logger.debug(f"TP broadcast qweight: rank {tp_rank} fills [:, {start_idx}:{end_idx}]")
                    return result

            # === 2. scales 和 zeros: 按第一个维度分割 (修正!) ===
            elif param_name in ["scales", "zeros"] and len(tensor.shape) == 2 and len(target_shape) == 2:
                tensor_h, tensor_w = tensor.shape
                target_h, target_w = target_shape

                # 🔧 修正: 按第一个维度分割 (group 维度)
                if tensor_h == target_h * tp_size and tensor_w == target_w:
                    start_idx = tp_rank * target_h
                    end_idx = start_idx + target_h
                    result = tensor[start_idx:end_idx, :].contiguous()
                    logger.debug(f"TP split {param_name} by rows: rank {tp_rank} gets [{start_idx}:{end_idx}, :]")
                    return result

                # 输出维度分割: tensor_w = target_w * tp_size
                elif tensor_h == target_h and tensor_w == target_w * tp_size:
                    start_idx = tp_rank * target_w
                    end_idx = start_idx + target_w
                    result = tensor[:, start_idx:end_idx].contiguous()
                    logger.debug(f"TP split {param_name} by cols: rank {tp_rank} gets [:, {start_idx}:{end_idx}]")
                    return result

                # 需要广播: target_w = tensor_w * tp_size
                elif tensor_h == target_h and target_w == tensor_w * tp_size:
                    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
                    start_idx = tp_rank * tensor_w
                    end_idx = start_idx + tensor_w
                    result[:, start_idx:end_idx] = tensor
                    logger.debug(f"TP broadcast {param_name}: rank {tp_rank} fills [:, {start_idx}:{end_idx}]")
                    return result

            # === 3. channel_scale: 按最后一个维度分割 ===
            elif param_name == "channel_scale" and len(tensor.shape) == 3 and len(target_shape) == 3:
                tensor_h, tensor_w, tensor_d = tensor.shape
                target_h, target_w, target_d = target_shape

                if tensor_h == target_h and tensor_w == target_w and tensor_d == target_d * tp_size:
                    start_idx = tp_rank * target_d
                    end_idx = start_idx + target_d
                    result = tensor[:, :, start_idx:end_idx].contiguous()
                    logger.debug(f"TP split channel_scale: rank {tp_rank} gets [:, :, {start_idx}:{end_idx}]")
                    return result

            # === 4. q_perm: 按唯一维度分割 ===
            elif param_name == "q_perm" and len(tensor.shape) == 1 and len(target_shape) == 1:
                tensor_size = tensor.shape[0]
                target_size = target_shape[0]

                if tensor_size == target_size * tp_size:
                    start_idx = tp_rank * target_size
                    end_idx = start_idx + target_size
                    result = tensor[start_idx:end_idx].contiguous()
                    logger.debug(f"TP split q_perm: rank {tp_rank} gets [{start_idx}:{end_idx}]")
                    return result

            # === 5. q_groups: 用于 mixed bitwidth 模式 ===
            elif param_name == "q_groups" and len(tensor.shape) == 1 and len(target_shape) == 1:
                tensor_size = tensor.shape[0]
                target_size = target_shape[0]

                if tensor_size == target_size * tp_size:
                    start_idx = tp_rank * target_size
                    end_idx = start_idx + target_size
                    result = tensor[start_idx:end_idx].contiguous()
                    logger.debug(f"TP split q_groups: rank {tp_rank} gets [{start_idx}:{end_idx}]")
                    return result

            # 如果都不匹配，详细报错
            logger.error(f"Cannot handle shape mismatch for {param_name}")
            logger.error(f"  Param shape: {target_shape} ({target_shape.numel()} elements)")
            logger.error(f"  Loaded shape: {tensor.shape} ({tensor.numel()} elements)")
            logger.error(f"  TP size: {tp_size}, TP rank: {tp_rank}")

            # 分析可能的分割方案
            logger.error(f"  Analysis:")
            for dim in range(len(tensor.shape)):
                if len(target_shape) > dim:
                    ratio = tensor.shape[dim] / target_shape[dim] if target_shape[dim] != 0 else float('inf')
                    logger.error(f"    Dim {dim}: {tensor.shape[dim]} -> {target_shape[dim]} (ratio: {ratio:.2f})")

                    # 检查是否是有效的TP分割
                    if abs(ratio - tp_size) < 0.1:
                        logger.error(f"      ✅ This dimension should be split by TP!")
                    elif abs(ratio - 1.0) < 0.1:
                        logger.error(f"      ✅ This dimension should remain the same")
                    else:
                        logger.error(f"      ❌ Unexpected ratio for TP splitting")

            raise ValueError(
                f"Shape mismatch for {param_name}:\n"
                f"  Expected: {target_shape} ({target_shape.numel()} elements)\n"
                f"  Got: {tensor.shape} ({tensor.numel()} elements)\n"
                f"  TP size: {tp_size}, TP rank: {tp_rank}\n"
                f"  This indicates GBA tensor parallel support needs more cases."
            )

        try:
            loaded_weight = ensure_dtype(loaded_weight, param.dtype)
            shaped_weight = ensure_shape(loaded_weight, param.shape, param_name)

            param.data.copy_(shaped_weight)

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

    def _prepare_layer_cache(self, layer: torch.nn.Module):
        """Prepare and cache all layer parameters for optimal performance"""

        # Ensure weights are processed
        if not hasattr(layer, '_gba_weights_processed'):
            self.process_weights_after_loading(layer)

        # Cache channel scale with optimized handling
        if hasattr(layer, 'channel_scale') and layer.channel_scale is not None:
            layer._has_channel_scale = True
            channel_scale = layer.channel_scale

            # Pre-convert and cache the properly shaped channel scale
            if channel_scale.dtype != torch.float16:
                channel_scale = channel_scale.to(torch.float16)

            # Pre-compute the right shape for both 2D and 3D inputs
            if channel_scale.dim() == 3:
                layer._channel_scale_cached = channel_scale.squeeze(0).squeeze(0)
            elif channel_scale.dim() == 2:
                layer._channel_scale_cached = channel_scale.squeeze(0)
            else:
                layer._channel_scale_cached = channel_scale
        else:
            layer._has_channel_scale = False
            layer._channel_scale_cached = None

        # Cache weight parameters with proper dtype
        layer._scales_cached = (layer.scales.to(torch.float16)
                                if layer.scales.dtype != torch.float16
                                else layer.scales)
        layer._zeros_cached = (layer.zeros.to(torch.float16)
                               if layer.zeros.dtype != torch.float16
                               else layer.zeros)

        # Cache group map and rows list
        layer._q_group_map_cached = getattr(layer, "q_group_map", None)

        rows_info = getattr(layer, "rows_info", None)
        layer._rows_list_cached = (rows_info.tolist()
                                   if rows_info is not None and rows_info.numel() > 0
                                   else [])

        # Pre-compute expected input features for validation
        layer._expected_input_features = layer.qweight.size(0) * (32 // layer.weight_bits)

        if hasattr(layer, 'bias') and layer.bias is not None:
            layer._bias_cached = layer.bias
            layer._has_bias = True
        else:
            layer._bias_cached = None
            layer._has_bias = False

        # Mark as ready
        layer._gba_ready = True

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Execute forward propagation with robust channel_scale handling"""
        if not hasattr(layer, '_gba_ready'):
            self._prepare_layer_cache(layer)

        input_shape = x.shape
        input_dim = len(input_shape)

        if input_dim == 3:
            x = x.view(-1, input_shape[-1])
            needs_reshape = True
            output_shape = input_shape[:-1] + (-1,)
        else:
            needs_reshape = False
            output_shape = None

        if layer._has_channel_scale:
            x = x * layer._channel_scale_cached

        output = ops.gba_linear_forward(
            x,
            layer.qweight,
            layer._scales_cached,
            layer._zeros_cached,
            layer.q_perm,
            layer.group_size,
            layer.weight_bits,
            self.quant_config.use_mbw,
            layer._q_group_map_cached,
            layer._rows_list_cached,
        )

        if bias is not None:
            output = output + bias

        if needs_reshape:
            output = output.view(output_shape)

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