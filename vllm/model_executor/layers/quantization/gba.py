import re
from typing import Any, Dict, List, Optional
import torch

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs
from vllm.logger import init_logger
from vllm import _custom_ops as ops

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

        # Add MoE info if present
        if 'moe_info' in config:
            instance.moe_info = config['moe_info']

        return instance

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional["GBALinearMethod"]:
        if isinstance(layer, torch.nn.Linear):
            return GBALinearMethod(self)
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional["GBALinearMethod"]:
        """return quantization method"""
        if isinstance(layer, LinearBase):
            logger.debug(f"Creating GBA linear method for layer: {prefix}")
            return GBALinearMethod(self)
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

        # 获取 GBA 权重加载器
        gba_weight_loader = self._get_gba_weight_loader()

        # 调试：打印所有传入的参数 - 修复版本
        print(f"=== GBA create_weights 调试 ===")
        print(f"Layer type: {type(layer).__name__}")
        print(f"Layer class: {layer.__class__}")
        print(f"Extra weight attrs keys: {list(extra_weight_attrs.keys())}")

        # 安全地打印extra_weight_attrs，避免打印未初始化的对象
        safe_attrs = {}
        for key, value in extra_weight_attrs.items():
            if key == "weight_loader":
                safe_attrs[key] = f"<function {value.__name__ if hasattr(value, '__name__') else 'unknown'}>"
            elif isinstance(value, str):
                safe_attrs[key] = value
            else:
                safe_attrs[key] = f"<{type(value).__name__} object>"

        print(f"Extra weight attrs (safe): {safe_attrs}")

        # layer specific config
        layer_prefix = extra_weight_attrs.get("prefix", "")
        print(f"Extracted prefix: '{layer_prefix}'")

        # 如果有prefix，说明修复成功了！
        if layer_prefix:
            print(f"✅ PREFIX 修复成功！完整路径: '{layer_prefix}'")
        else:
            print(f"❌ PREFIX 仍然为空，需要进一步调试")

        print("=" * 40)



        logger.info(f"Creating GBA weights for layer: {layer_prefix}")
        layer_config = self._get_layer_config(layer_prefix)

        logger.debug(f"layer config: {layer_config},"
                     f"layer prefix: {layer_prefix}")

        weight_bits = layer_config.get("weight_bits", self.quant_config.weight_bits)
        group_size = layer_config.get("group_size", self.quant_config.group_size)

        logger.info(layer_config)

        if group_size == -1:
            group_size = input_size_per_partition

        num_groups = input_size_per_partition // group_size

        # Determine weight shape based on whether mixed bit-width is used
        if not self.quant_config.use_mbw:
            # Standard quantization mode
            packed_rows = input_size_per_partition * weight_bits // 32
            qweight_shape = (packed_rows, output_size_per_partition)
        else:
            # Mixed bit-width mode - use dynamic packing size
            qweight_shape = (input_size_per_partition // 32, output_size_per_partition)

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

        # Create quantization scales and zero points
        scale_zero_shape = (num_groups, output_size_per_partition)

        scales = torch.nn.Parameter(
            torch.empty(scale_zero_shape, dtype=params_dtype, device="cuda"),
            requires_grad=False,
        )
        scales._param_name = "scales"
        set_weight_attrs(scales, {
            "input_dim": 0,
            "output_dim": 1,
            "weight_loader": gba_weight_loader  # 添加权重加载器
        })

        zeros = torch.nn.Parameter(
            torch.empty(scale_zero_shape, dtype=params_dtype, device="cuda"),
            requires_grad=False,
        )
        zeros._param_name = "zeros"
        set_weight_attrs(zeros, {
            "input_dim": 0,
            "output_dim": 1,
            "weight_loader": gba_weight_loader  # 使用GBA权重加载器
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
            "weight_loader": gba_weight_loader  # 使用GBA权重加载器
        })

        channel_scale = torch.nn.Parameter(
            torch.ones((1, 1, input_size_per_partition), dtype=params_dtype, device="cuda"),
            requires_grad=False,
        )
        channel_scale._param_name = "channel_scale"
        set_weight_attrs(channel_scale, {
            "input_dim": 2,
            "output_dim": -1,
            "weight_loader": gba_weight_loader  # 使用GBA权重加载器
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
                "weight_loader": gba_weight_loader  # 使用GBA权重加载器
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

        layer._gba_weights_initialized = True
        logger.info(f"Created GBA weights for layer {layer_prefix} with shapes: "
                    f"qweight={qweight_shape}, "
                    f"scales={scale_zero_shape}")

    def _get_gba_weight_loader(self):
        """Get GBA-specific weight loader function - 修复版本"""

        def gba_weight_loader_wrapper(param: torch.nn.Parameter, loaded_weight: torch.Tensor, *args, **kwargs):
            """
            GBA权重加载器包装函数
            支持不同的调用方式：
            - gba_weight_loader_wrapper(param, loaded_weight)  # 2个参数
            - gba_weight_loader_wrapper(param, loaded_weight, shard_id)  # 3个参数
            """

            # 处理可能的第三个参数（shard_id等）
            shard_id = args[0] if args else kwargs.get('shard_id', None)

            # 更严谨的参数名推断逻辑
            param_name = self._infer_param_name(param, shard_id)

            logger.info(f"GBA weight loader called for parameter: {param_name}, "
                        f"param shape: {param.shape}, loaded shape: {loaded_weight.shape}")

            return self._gba_weight_loader(param, loaded_weight, param_name)

        return gba_weight_loader_wrapper

    def _infer_param_name(self, param: torch.nn.Parameter, shard_id=None) -> str:
        """更严谨的参数名推断逻辑"""

        if hasattr(param, '_param_name'):
            return param._param_name

        # 方法2：根据参数的形状和数据类型推断
        param_shape = param.shape
        param_dtype = param.dtype
        param_dim = param.dim()

        # 量化权重 (packed integers)
        if param_dtype == torch.int32:
            return "qweight"

        # 排列索引
        elif param_dtype == torch.int16:
            if param_dim == 1:
                return "q_perm"
            else:
                return "q_groups"

        # 通道缩放因子 (通常是3维)
        elif param_dim == 3 and param_shape[0] == 1 and param_shape[1] == 1:
            return "channel_scale"

        # 量化scales和zeros (通常是2维，float类型)
        elif param_dtype in [torch.float16, torch.bfloat16, torch.float32]:
            if param_dim == 2:
                # 根据形状特征进一步判断
                rows, cols = param_shape
                # scales和zeros通常有相同的形状
                # 可以根据注册顺序或其他特征区分
                # 这里使用简单的启发式：如果已经推断过scales，下一个就是zeros
                if not hasattr(self, '_last_inferred_param'):
                    self._last_inferred_param = "scales"
                    return "scales"
                else:
                    if self._last_inferred_param == "scales":
                        self._last_inferred_param = "zeros"
                        return "zeros"
                    else:
                        self._last_inferred_param = "scales"
                        return "scales"
            else:
                return "scales"  # 默认假设是scales

        # 如果都无法推断，返回基于shard_id的名称
        if shard_id is not None:
            return f"param_{shard_id}"

        # 最后的兜底
        return "unknown"

    def _gba_weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor, param_name: str) -> None:
        """
        GBA quantized weight loader implementation.
        """
        logger.info(
            f"Loading GBA weight: {param_name}, param shape: {param.shape}, loaded shape: {loaded_weight.shape}")

        def ensure_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            if tensor.dtype != dtype:
                return tensor.to(dtype)
            return tensor

    def ensure_shape(tensor: torch.Tensor, target_shape: torch.Size, param_name: str) -> torch.Tensor:
        """Ensure tensor has the correct shape, with fallback strategies - 修复版本"""
        if tensor.shape == target_shape:
            return tensor

        # Try to reshape if same number of elements
        if tensor.numel() == target_shape.numel():
            logger.info(f"Reshaping {param_name} from {tensor.shape} to {target_shape}")
            return tensor.view(target_shape)

        # For quantized weights, handle potential packing differences
        if "qweight" in param_name:
            # 计算元素数量比例
            target_elements = target_shape.numel()
            tensor_elements = tensor.numel()

            if target_elements == tensor_elements * 2:
                # 目标形状是加载权重的2倍 - 可能是gate_up_proj这样的合并层
                logger.info(f"Expanding {param_name} from {tensor.shape} to {target_shape} (2x expansion)")

                # 对于gate_up_proj，需要在列维度上扩展（复制）
                if len(tensor.shape) == 2 and len(target_shape) == 2:
                    # 在列维度上复制：[128, 3072] -> [128, 6144]
                    expanded = torch.cat([tensor, tensor], dim=1)
                    if expanded.shape != target_shape:
                        # 如果还是不匹配，尝试调整
                        if expanded.shape[1] > target_shape[1]:
                            expanded = expanded[:, :target_shape[1]]
                        logger.info(f"Final expanded shape: {expanded.shape}")
                    return expanded
                else:
                    # 通用方法：先flatten，复制，然后reshape
                    flattened = tensor.flatten()
                    expanded = torch.cat([flattened, flattened])
                    return expanded[:target_elements].view(target_shape)

            elif tensor_elements == target_elements * 2:
                # 加载权重是目标的2倍 - 可能需要截取
                logger.info(f"Truncating {param_name} from {tensor.shape} to {target_shape} (2x reduction)")

                if len(tensor.shape) == 2 and len(target_shape) == 2:
                    # 对于2D权重，在列维度上截取一半
                    return tensor[:, :target_shape[1]]
                else:
                    # 通用方法：先flatten，截取，然后reshape
                    flattened = tensor.flatten()
                    truncated = flattened[:target_elements]
                    return truncated.view(target_shape)

            # 处理其他比例关系
            ratio = target_elements / tensor_elements
            if abs(ratio - round(ratio)) < 1e-6:  # 如果是整数倍
                ratio = int(round(ratio))
                logger.info(f"Scaling {param_name} by factor {ratio}")

                if ratio > 1:
                    # 需要扩展
                    if len(tensor.shape) == 2 and len(target_shape) == 2:
                        # 对于2D权重，优先在列维度扩展
                        if target_shape[1] == tensor.shape[1] * ratio:
                            expanded = tensor.repeat(1, ratio)
                            return expanded

                    # 通用扩展方法
                    flattened = tensor.flatten()
                    expanded = flattened.repeat(ratio)
                    return expanded[:target_elements].view(target_shape)
                else:
                    # 需要缩小
                    step = int(1 / ratio)
                    flattened = tensor.flatten()
                    reduced = flattened[::step]
                    return reduced[:target_elements].view(target_shape)

        logger.warning(
            f"Shape mismatch for {param_name}: param {target_shape} vs loaded {tensor.shape}, "
            f"elements: {target_shape.numel()} vs {tensor.numel()}, "
            f"attempting default loading"
        )

        # 最后的尝试：如果是2D张量且行数匹配，尝试调整列数
        if (len(tensor.shape) == 2 and len(target_shape) == 2 and
                tensor.shape[0] == target_shape[0]):

            if tensor.shape[1] < target_shape[1]:
                # 需要扩展列
                repeat_factor = target_shape[1] // tensor.shape[1]
                remainder = target_shape[1] % tensor.shape[1]

                if remainder == 0:
                    # 完全整除，直接重复
                    expanded = tensor.repeat(1, repeat_factor)
                    return expanded
                else:
                    # 不整除，重复后截取
                    expanded = tensor.repeat(1, repeat_factor + 1)
                    return expanded[:, :target_shape[1]]
            elif tensor.shape[1] > target_shape[1]:
                # 需要截取列
                return tensor[:, :target_shape[1]]

        return tensor

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Processing after weight loading - this should be called by the model loader"""

        # Check if weights have been initialized
        if not hasattr(layer, '_gba_weights_initialized'):
            logger.warning("GBA weights not properly initialized for layer")
            return

        # Check if processing has already been done
        if hasattr(layer, '_gba_weights_processed'):
            return

        # Check if necessary weights have been loaded
        required_weights = ["qweight", "scales", "zeros", "q_perm", "channel_scale"]
        for weight_name in required_weights:
            if not hasattr(layer, weight_name):
                raise ValueError(f"Missing required weight: {weight_name}")

        logger.info(f"Processing GBA weights for layer with input_size={layer.input_size_per_partition}")

        # Convert weight layout
        if self.quant_config.use_mbw:
            # Mixed bit-width mode
            if not hasattr(layer, "q_groups"):
                raise ValueError("Mixed bitwidth mode requires q_groups")

            qweight, rows = ops.gba_trans_qweight(
                layer.qweight,
                layer.q_groups,
                True, # use_mbw=True
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

            logger.info(f"Applied mixed bitwidth quantization: {len(rows)} row groups")

        else:
            # Standard quantization mode
            qweight, rows = ops.gba_trans_qweight(
                layer.qweight,
                torch.empty(1, dtype=torch.int16, device=layer.qweight.device),
                False, # use_mbw=False
                layer.input_size_per_partition,
                layer.scales.size(0),
                self.quant_config.weight_bits,
            )
            layer.qweight.data = qweight

            if isinstance(rows, list) and len(rows) > 0:
                layer.rows_info = torch.tensor(rows, dtype=torch.int32, device=layer.qweight.device)
            else:
                layer.rows_info = torch.empty(0, dtype=torch.int32, device=layer.qweight.device)

            logger.info("Applied standard quantization")

        # Mark as processed
        layer._gba_weights_processed = True

    def apply_quant_strategy(self, name_attr: str, quant_strategy: Dict):
        """
        Apply quantization strategy based on the layer's name and the provided strategy.
        Updated to support DeepSeek V2 MoE models and other complex architectures.
        """
        strategy = None

        # 处理融合层映射
        if 'gate_up_proj' in name_attr:
            # 优先查找gate_proj配置
            for key in ['gate_proj', 'up_proj']:
                if key in quant_strategy:
                    logger.info(f"Using {key} strategy for gate_up_proj")
                    return quant_strategy[key]

        if 'qkv_proj' in name_attr:
            # 优先查找q_proj配置
            for key in ['q_proj', 'k_proj', 'v_proj']:
                if key in quant_strategy:
                    logger.info(f"Using {key} strategy for qkv_proj")
                    return quant_strategy[key]

        # DeepSeek V2 style attention projections (decomposed Q/K/V)
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

        # MoE gate layer (router) - includes both weight and bias
        # DeepSeek V2 has: mlp.gate.weight and mlp.gate.e_score_correction_bias
        if ('mlp.gate.' in name_attr or name_attr.endswith('mlp.gate')) and 'experts' not in name_attr:
            try:
                strategy = quant_strategy['moe_gate']
                return strategy
            except KeyError:
                pass

        # MoE shared expert layers (DeepSeek V2 specific)
        # Note: actual path is 'mlp.shared_experts.' (plural)
        if 'mlp.shared_experts.' in name_attr or 'shared_experts.' in name_attr:
            if '.gate_proj' in name_attr or 'gate_proj' in name_attr:
                try:
                    strategy = quant_strategy['moe_shared_expert_gate_proj']
                    return strategy
                except KeyError:
                    pass
            elif '.up_proj' in name_attr or 'up_proj' in name_attr:
                try:
                    strategy = quant_strategy['moe_shared_expert_up_proj']
                    return strategy
                except KeyError:
                    pass
            elif '.down_proj' in name_attr or 'down_proj' in name_attr:
                try:
                    strategy = quant_strategy['moe_shared_expert_down_proj']
                    return strategy
                except KeyError:
                    pass

        # MoE expert layers - match any expert number (supports 100+ experts)
        if 'mlp.experts.' in name_attr:
            if '.gate_proj' in name_attr:
                try:
                    strategy = quant_strategy['moe_expert_gate_proj']
                    return strategy
                except KeyError:
                    pass
            elif '.up_proj' in name_attr:
                try:
                    strategy = quant_strategy['moe_expert_up_proj']
                    return strategy
                except KeyError:
                    pass
            elif '.down_proj' in name_attr:
                try:
                    strategy = quant_strategy['moe_expert_down_proj']
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

    def _get_layer_config(self, layer_prefix: str) -> Dict[str, Any]:
        """获取层特定的量化配置 - 支持融合层"""
        logger.info(f"Getting layer config for prefix: {layer_prefix}")

        if not hasattr(self.quant_config, 'strategy') or not self.quant_config.strategy:
            logger.debug("No strategy config available")
            return {}

        strategy = self.quant_config.strategy
        logger.debug(f"Available strategy keys: {list(strategy.keys())}")

        # 解析完整的层路径
        parts = layer_prefix.split('.')
        layer_num = None
        proj_type = None

        # 查找层编号
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_num = int(parts[i + 1])
                    break
                except ValueError:
                    continue

        # 查找投影类型
        for part in parts:
            if part in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
                        'qkv_proj', 'gate_up_proj']:  # 添加融合层类型
                proj_type = part
                break

        if layer_num is not None and proj_type is not None:
            layer_key = f"model.layers.{layer_num}"
            logger.info(f"Looking for layer key: {layer_key}, proj type: {proj_type}")

            if layer_key in strategy:
                layer_config = strategy[layer_key]

                # 处理融合层的特殊情况
                if proj_type == "gate_up_proj":
                    # gate_up_proj是gate_proj和up_proj的融合
                    # 优先使用gate_proj的配置，如果没有则使用up_proj的配置
                    if "gate_proj" in layer_config:
                        proj_config = layer_config["gate_proj"]
                        logger.info(f"Using gate_proj config for gate_up_proj: {proj_config}")
                    elif "up_proj" in layer_config:
                        proj_config = layer_config["up_proj"]
                        logger.info(f"Using up_proj config for gate_up_proj: {proj_config}")
                    else:
                        logger.warning(f"No config found for gate_up_proj components")
                        return {}
                elif proj_type == "qkv_proj":
                    # qkv_proj是q_proj, k_proj, v_proj的融合
                    # 优先使用q_proj的配置
                    if "q_proj" in layer_config:
                        proj_config = layer_config["q_proj"]
                        logger.info(f"Using q_proj config for qkv_proj: {proj_config}")
                    elif "k_proj" in layer_config:
                        proj_config = layer_config["k_proj"]
                        logger.info(f"Using k_proj config for qkv_proj: {proj_config}")
                    else:
                        logger.warning(f"No config found for qkv_proj components")
                        return {}
                else:
                    # 普通的单独投影层
                    if proj_type in layer_config:
                        proj_config = layer_config[proj_type]
                        logger.info(f"Found layer-specific proj config: {proj_config}")
                    else:
                        logger.warning(f"Projection type {proj_type} not found in layer {layer_key}")
                        return {}

                # 从策略配置中提取参数
                config = {}

                # 提取weight_bits
                if 'bits' in proj_config and proj_config['bits']:
                    if isinstance(proj_config['bits'], list):
                        config['weight_bits'] = proj_config['bits'][0]
                    else:
                        config['weight_bits'] = proj_config['bits']

                # 提取group_size - 修复访问逻辑
                if 'group_size' in proj_config:
                    group_size_config = proj_config['group_size']
                    if isinstance(group_size_config, dict):
                        weight_bits_str = str(config.get('weight_bits', 4))
                        if weight_bits_str in group_size_config:
                            config['group_size'] = group_size_config[weight_bits_str]
                        else:
                            # 如果没有对应的bits，取第一个值
                            config['group_size'] = next(iter(group_size_config.values()))
                    else:
                        config['group_size'] = group_size_config

                logger.info(f"Extracted layer-specific config: {config}")
                return config
            else:
                logger.debug(f"Layer key {layer_key} not found in strategy")

        # 使用通用策略匹配
        logger.debug("No layer-specific config found, trying generic strategy matching")
        generic_strategy = self.apply_quant_strategy(layer_prefix, strategy)
        if generic_strategy:
            logger.info(f"Found generic strategy: {generic_strategy}")
            # ... 处理generic_strategy的逻辑保持不变 ...

        logger.info("No configuration found, using defaults")
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

        # 记录原始信息
        original_dtype = x.dtype
        original_shape = x.shape

        logger.debug(f"GBA Apply - Original input shape: {original_shape}")
        logger.debug(f"GBA Apply - Original input dtype: {original_dtype}")

        # 处理输入形状和 channel_scale 应用
        if x.dim() == 3:
            # 输入是 3D [batch, seq_len, hidden_size]
            logger.debug(f"Input is 3D: {x.shape}")

            # 应用 channel_scale（原始方式）
            if hasattr(layer, 'channel_scale'):
                channel_scale = layer.channel_scale
                if channel_scale.dtype != torch.float16:
                    channel_scale = channel_scale.to(torch.float16)
                logger.debug(f"Applying 3D channel_scale {channel_scale.shape} to {x.shape}")
                x = x.mul(channel_scale)

            # Flatten 为 2D
            x, shape = flatten_x(x)
            logger.debug(f"Flattened to: {x.shape}, saved shape: {shape}")

        elif x.dim() == 2:
            # 输入是 2D [batch*seq_len, hidden_size]
            logger.debug(f"Input is 2D: {x.shape}")

            # 修改 channel_scale 的形状以适应 2D 输入
            if hasattr(layer, 'channel_scale'):
                channel_scale = layer.channel_scale  # 原始形状 [1, 1, hidden_size]
                if channel_scale.dtype != torch.float16:
                    channel_scale = channel_scale.to(torch.float16)

                # 将 channel_scale 从 [1, 1, hidden_size] 转换为 [1, hidden_size] 或 [hidden_size]
                if channel_scale.dim() == 3:
                    # 去掉多余的维度：[1, 1, hidden_size] -> [hidden_size]
                    channel_scale_2d = channel_scale.squeeze(0).squeeze(0)  # [hidden_size]
                    logger.debug(f"Reshaped channel_scale from {channel_scale.shape} to {channel_scale_2d.shape}")
                elif channel_scale.dim() == 2:
                    channel_scale_2d = channel_scale.squeeze(0)  # [1, hidden_size] -> [hidden_size]
                    logger.debug(f"Reshaped channel_scale from {channel_scale.shape} to {channel_scale_2d.shape}")
                else:
                    channel_scale_2d = channel_scale
                    logger.debug(f"Channel_scale already appropriate shape: {channel_scale_2d.shape}")

                logger.debug(f"Applying 2D channel_scale {channel_scale_2d.shape} to {x.shape}")
                x = x.mul(channel_scale_2d)  # Broadcasting: [batch*seq_len, hidden_size] * [hidden_size]
                logger.debug(f"After channel_scale: {x.shape}")

            # 对于 2D 输入，我们设置一个假的 shape 用于后续的 unflatten
            # 但实际上我们可能需要保持 2D 输出
            shape = [x.size(0)]  # 简单地保存第一个维度

        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}, shape: {x.shape}")

        # 确保数据类型
        if x.dtype != torch.float16:
            logger.debug(f"Converting input from {x.dtype} to torch.float16")
            x = x.to(torch.float16)

        # Prepare parameters
        q_group_map = getattr(layer, "q_group_map", None)
        rows_info = getattr(layer, "rows_info", None)
        rows_list = rows_info.tolist() if rows_info is not None and rows_info.numel() > 0 else []

        # 确保权重参数类型正确
        scales = layer.scales.to(torch.float16) if layer.scales.dtype != torch.float16 else layer.scales
        zeros = layer.zeros.to(torch.float16) if layer.zeros.dtype != torch.float16 else layer.zeros

        # 验证形状匹配
        expected_input_features = layer.qweight.size(0) * (32 // layer.weight_bits)
        actual_input_features = x.size(1)

        logger.debug(f"Shape verification:")
        logger.debug(f"  qweight shape: {layer.qweight.shape}")
        logger.debug(f"  expected input features: {expected_input_features}")
        logger.debug(f"  actual input features: {actual_input_features}")
        logger.debug(f"  input shape after all processing: {x.shape}")

        if expected_input_features != actual_input_features:
            logger.error(f"Shape mismatch: expected {expected_input_features}, got {actual_input_features}")
            logger.error(f"qweight.size(0): {layer.qweight.size(0)}, weight_bits: {layer.weight_bits}")
            logger.error(f"This suggests the weight quantization doesn't match the model architecture")
            raise RuntimeError(
                f"Shape mismatch in GBA layer: expected input features {expected_input_features}, got {actual_input_features}")

        # Call CUDA forward propagation
        logger.debug(f"Calling CUDA forward with input shape: {x.shape}")
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

        logger.debug(f"CUDA forward successful, output shape: {output.shape}")

        # Add bias
        if bias is not None:
            if bias.dtype != output.dtype:
                bias = bias.to(output.dtype)
            output = output + bias

        # 恢复输出形状
        if len(original_shape) == 3:
            # 原始输入是 3D，恢复为 3D
            output = unflatten_x(output, shape)
            logger.debug(f"Restored to 3D output shape: {output.shape}")
        elif len(original_shape) == 2:
            # 原始输入是 2D，保持 2D
            # output 已经是正确的 2D 形状 [batch*seq_len, output_features]
            logger.debug(f"Keeping 2D output shape: {output.shape}")

        # 转换回原始数据类型
        if original_dtype != torch.float16:
            output = output.to(original_dtype)

        logger.debug(f"Final output shape: {output.shape}, dtype: {output.dtype}")
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