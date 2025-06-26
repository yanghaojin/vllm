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
        """Create GBA quantized weight parameters - 简化版本，支持分离层"""

        output_size_per_partition = sum(output_partition_sizes)

        # 获取 GBA 权重加载器
        gba_weight_loader = self._get_gba_weight_loader()

        # layer specific config
        layer_prefix = extra_weight_attrs.get("prefix", "")

        logger.info(f"Creating GBA weights for layer: {layer_prefix}")
        logger.info(f"  input_size_per_partition: {input_size_per_partition}")
        logger.info(f"  output_size_per_partition: {output_size_per_partition}")

        layer_config = self._get_layer_config(layer_prefix)

        weight_bits = layer_config.get("weight_bits", self.quant_config.weight_bits)
        group_size = layer_config.get("group_size", self.quant_config.group_size)

        if group_size == -1:
            group_size = input_size_per_partition

        num_groups = input_size_per_partition // group_size

        # 简化的权重形状计算 - 不再处理融合层的复杂情况
        if not self.quant_config.use_mbw:
            # Standard quantization mode
            packed_rows = input_size_per_partition * weight_bits // 32
            qweight_shape = (packed_rows, output_size_per_partition)
        else:
            # Mixed bit-width mode
            qweight_shape = (input_size_per_partition // 32, output_size_per_partition)

        # Create quantization scales and zero points
        scale_zero_shape = (num_groups, output_size_per_partition)

        logger.info(f"Creating parameter shapes:")
        logger.info(f"  qweight: {qweight_shape}")
        logger.info(f"  scales/zeros: {scale_zero_shape}")

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

        layer._gba_weights_initialized = True
        logger.info(f"Successfully created GBA weights for {layer_prefix}")

    def _get_gba_weight_loader(self):
        """Get GBA-specific weight loader function - 简化版本"""

        def gba_weight_loader_wrapper(param: torch.nn.Parameter, loaded_weight: torch.Tensor, *args, **kwargs):
            """GBA权重加载器包装函数 - 简化版本"""

            # 推断参数名
            param_name = self._infer_param_name(param, args[0] if args else None)

            logger.debug(f"GBA weight loader: {param_name} - param: {param.shape}, loaded: {loaded_weight.shape}")

            return self._gba_weight_loader(param, loaded_weight, param_name)

        return gba_weight_loader_wrapper

    def _infer_param_name(self, param: torch.nn.Parameter, shard_id=None) -> str:
        """参数名推断逻辑 - 简化版本"""

        # 方法1：直接从参数属性获取
        if hasattr(param, '_param_name'):
            return param._param_name

        # 方法2：根据参数特征推断
        param_shape = param.shape
        param_dtype = param.dtype
        param_dim = param.dim()

        # 量化权重 (int32, 2D)
        if param_dtype == torch.int32 and param_dim == 2:
            return "qweight"

        # 排列索引 (int16, 1D)
        elif param_dtype == torch.int16:
            if param_dim == 1:
                return "q_perm"
            else:
                return "q_groups"

        # 通道缩放因子 (float, 3D, 第一两维为1)
        elif param_dim == 3 and param_shape[0] == 1 and param_shape[1] == 1:
            return "channel_scale"

        # scales和zeros (float, 2D) - 使用简单的顺序判断
        elif param_dtype in [torch.float16, torch.bfloat16, torch.float32] and param_dim == 2:
            if not hasattr(self, '_scale_zero_counter'):
                self._scale_zero_counter = 0

            # 简单的交替逻辑：第一个是scales，第二个是zeros
            if self._scale_zero_counter % 2 == 0:
                self._scale_zero_counter += 1
                return "scales"
            else:
                self._scale_zero_counter += 1
                return "zeros"

        # 兜底策略
        return f"unknown_{param_dim}d_{param_dtype}"

    def _gba_weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor, param_name: str) -> None:
        """GBA quantized weight loader implementation - 简化版本"""

        logger.debug(f"Loading GBA weight: {param_name}")
        logger.debug(f"  Param shape: {param.shape}, Loaded shape: {loaded_weight.shape}")

        def ensure_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            if tensor.dtype != dtype:
                return tensor.to(dtype)
            return tensor

        def ensure_shape(tensor: torch.Tensor, target_shape: torch.Size, param_name: str) -> torch.Tensor:
            """确保张量具有正确的形状 - 简化版本"""
            if tensor.shape == target_shape:
                return tensor

            # 如果元素数量相同，直接reshape
            if tensor.numel() == target_shape.numel():
                logger.debug(f"Reshaping {param_name}: {tensor.shape} -> {target_shape}")
                return tensor.view(target_shape)

            # 形状不匹配且元素数量不同 - 报错
            raise ValueError(
                f"Shape mismatch for {param_name}:\n"
                f"  Expected: {target_shape} ({target_shape.numel()} elements)\n"
                f"  Got: {tensor.shape} ({tensor.numel()} elements)\n"
                f"  This indicates a mismatch between model architecture and weight file."
            )

        # 应用修复
        loaded_weight = ensure_dtype(loaded_weight, param.dtype)
        shaped_weight = ensure_shape(loaded_weight, param.shape, param_name)

        # 数值安全检查
        if 'scale' in param_name.lower():
            if torch.all(shaped_weight == 0):
                logger.warning(f"Warning: {param_name} is all zeros! Fixing...")
                shaped_weight = torch.where(shaped_weight == 0,
                                            torch.ones_like(shaped_weight) * 1e-6,
                                            shaped_weight)
            elif torch.any(shaped_weight <= 0):
                logger.warning(f"Warning: {param_name} has non-positive values! Fixing...")
                shaped_weight = torch.where(shaped_weight <= 0,
                                            torch.ones_like(shaped_weight) * 1e-6,
                                            shaped_weight)

        # 复制数据
        param.data.copy_(shaped_weight)

        logger.debug(f"Successfully loaded {param_name}")

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
        required_weights = ["qweight", "scales", "zeros", "q_perm", "channel_scale"]
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
        """获取层特定的量化配置 - 简化版本"""

        if not hasattr(self.quant_config, 'strategy') or not self.quant_config.strategy:
            return {}

        strategy = self.quant_config.strategy

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
            if part in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
                proj_type = part
                break

        if layer_num is not None and proj_type is not None:
            layer_key = f"model.layers.{layer_num}"

            if layer_key in strategy:
                layer_config = strategy[layer_key]

                if proj_type in layer_config:
                    proj_config = layer_config[proj_type]

                    # 从策略配置中提取参数
                    config = {}

                    # 提取weight_bits
                    if 'bits' in proj_config and proj_config['bits']:
                        if isinstance(proj_config['bits'], list):
                            config['weight_bits'] = proj_config['bits'][0]
                        else:
                            config['weight_bits'] = proj_config['bits']

                    # 提取group_size
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

        # 记录原始信息
        original_dtype = x.dtype
        original_shape = x.shape

        # 处理输入形状和 channel_scale 应用
        if x.dim() == 3:
            # 输入是 3D [batch, seq_len, hidden_size]
            # 应用 channel_scale（原始方式）
            if hasattr(layer, 'channel_scale'):
                channel_scale = layer.channel_scale
                if channel_scale.dtype != torch.float16:
                    channel_scale = channel_scale.to(torch.float16)
                x = x.mul(channel_scale)

            # Flatten 为 2D
            x, shape = flatten_x(x)

        elif x.dim() == 2:
            # 输入是 2D [batch*seq_len, hidden_size]
            # 修改 channel_scale 的形状以适应 2D 输入
            if hasattr(layer, 'channel_scale'):
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

            # 对于 2D 输入，设置假的 shape 用于后续处理
            shape = [x.size(0)]

        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}, shape: {x.shape}")

        # 确保数据类型
        if x.dtype != torch.float16:
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

        # 恢复输出形状
        if len(original_shape) == 3:
            # 原始输入是 3D，恢复为 3D
            output = unflatten_x(output, shape)
        elif len(original_shape) == 2:
            # 原始输入是 2D，保持 2D
            pass

        # 转换回原始数据类型
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