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

        # layer specific config
        layer_prefix = extra_weight_attrs.get("prefix", "")

        if not layer_prefix:
            # 尝试获取层的完整名称
            if hasattr(layer, '_get_name'):
                layer_prefix = layer._get_name()
            elif hasattr(layer, '__class__'):
                # 最后的备选方案：使用类名，但这不是最佳选择
                layer_prefix = layer.__class__.__name__
                logger.warning(f"Using class name as prefix: {layer_prefix}. "
                               "Consider providing full layer path for better strategy matching.")

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
        set_weight_attrs(qweight, {
            "input_dim": 0,
            "output_dim": 1,
            "weight_loader": self._get_gba_weight_loader()
        })

        # Create quantization scales and zero points
        scale_zero_shape = (num_groups, output_size_per_partition)

        scales = torch.nn.Parameter(
            torch.empty(scale_zero_shape, dtype=params_dtype, device="cuda"),
            requires_grad=False,
        )
        set_weight_attrs(scales, {"input_dim": 0, "output_dim": 1})

        zeros = torch.nn.Parameter(
            torch.empty(scale_zero_shape, dtype=params_dtype, device="cuda"),
            requires_grad=False,
        )
        set_weight_attrs(zeros, {
            "input_dim": 0,
            "output_dim": 1,
            "weight_loader": self._get_gba_weight_loader()
        })

        # Create permutation indices
        q_perm = torch.nn.Parameter(
            torch.empty(input_size_per_partition, dtype=torch.int16, device="cuda"),
            requires_grad=False,
        )
        set_weight_attrs(q_perm, {
            "input_dim": 0,
            "output_dim": -1,
            "weight_loader": self._get_gba_weight_loader()
        })

        channel_scale = torch.nn.Parameter(
            torch.ones((1, 1, input_size_per_partition), dtype=params_dtype, device="cuda"),
            requires_grad=False,
        )
        set_weight_attrs(channel_scale, {
            "input_dim": 2,
            "output_dim": -1,
            "weight_loader": self._get_gba_weight_loader()
        })
        layer.register_parameter("channel_scale", channel_scale)

        # Mixed bit-width mode requires additional parameters
        if self.quant_config.use_mbw:
            # Group information
            q_groups = torch.nn.Parameter(
                torch.empty(num_groups * 2, dtype=torch.int16, device="cuda"),
                requires_grad=False,
            )
            set_weight_attrs(q_groups, {
                "input_dim": -1,
                "output_dim": -1,
                "weight_loader": self._get_gba_weight_loader()
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
        """Get GBA-specific weight loader function"""

        def gba_weight_loader_wrapper(param: torch.nn.Parameter, loaded_weight: torch.Tensor):
            param_name = None
            # Try to get parameter name from the parameter itself or its context
            for name, p in param.named_parameters() if hasattr(param, 'named_parameters') else []:
                if p is param:
                    param_name = name
                    break

            if param_name is None:
                # Fallback: infer from parameter attributes or tensor properties
                if hasattr(param, '_param_name'):
                    param_name = param._param_name
                else:
                    # Guess based on tensor properties
                    if param.dtype == torch.int32:
                        param_name = "qweight"
                    elif param.dtype == torch.int16:
                        param_name = "q_perm" if param.dim() == 1 else "q_groups"
                    elif "scale" in str(param.shape):
                        param_name = "scales"
                    else:
                        param_name = "unknown"

            logger.info(f"GBA weight loader called for parameter: {param_name}, "
                        f"param shape: {param.shape}, loaded shape: {loaded_weight.shape}")

            return self._gba_weight_loader(param, loaded_weight, param_name)

        return gba_weight_loader_wrapper

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
            """Ensure tensor has the correct shape, with fallback strategies"""
            if tensor.shape == target_shape:
                return tensor

            # Try to reshape if same number of elements
            if tensor.numel() == target_shape.numel():
                logger.info(f"Reshaping {param_name} from {tensor.shape} to {target_shape}")
                return tensor.view(target_shape)

            # For quantized weights, handle potential packing differences
            if "qweight" in param_name:
                # Handle different bit-width packing
                if tensor.numel() * 2 == target_shape.numel():
                    logger.info(f"Handling bit-width packing difference for {param_name}")
                    return tensor.repeat_interleave(2, dim=0)[:target_shape[0]].view(target_shape)
                elif tensor.numel() == target_shape.numel() * 2:
                    logger.info(f"Handling bit-width packing difference for {param_name}")
                    return tensor[::2].view(target_shape)

            logger.warning(
                f"Shape mismatch for {param_name}: param {target_shape} vs loaded {tensor.shape}, "
                f"attempting default loading"
            )
            return tensor

        # Handle different parameter types
        if "qweight" in param_name:
            loaded_weight = ensure_dtype(loaded_weight, param.dtype)
            loaded_weight = ensure_shape(loaded_weight, param.shape, param_name)

        elif "scales" in param_name or "qscales" in param_name:
            loaded_weight = ensure_dtype(loaded_weight, param.dtype)
            loaded_weight = ensure_shape(loaded_weight, param.shape, param_name)

        elif "zeros" in param_name or "qzeros" in param_name:
            loaded_weight = ensure_dtype(loaded_weight, param.dtype)
            loaded_weight = ensure_shape(loaded_weight, param.shape, param_name)

        elif "q_perm" in param_name:
            loaded_weight = ensure_dtype(loaded_weight, param.dtype)
            if loaded_weight.dim() != 1:
                loaded_weight = loaded_weight.flatten()
            if loaded_weight.shape != param.shape:
                logger.warning(f"q_perm shape mismatch: param {param.shape} vs loaded {loaded_weight.shape}")
                if loaded_weight.numel() > param.numel():
                    loaded_weight = loaded_weight[:param.numel()].view(param.shape)
                else:
                    padded = torch.zeros(param.shape, dtype=loaded_weight.dtype, device=loaded_weight.device)
                    padded[:loaded_weight.numel()] = loaded_weight.flatten()
                    loaded_weight = padded

        elif "q_groups" in param_name:
            loaded_weight = ensure_dtype(loaded_weight, param.dtype)
            if loaded_weight.dim() != 1:
                loaded_weight = loaded_weight.flatten()
            if loaded_weight.shape != param.shape:
                logger.warning(f"q_groups shape mismatch: param {param.shape} vs loaded {loaded_weight.shape}")
                if loaded_weight.numel() > param.numel():
                    loaded_weight = loaded_weight[:param.numel()].view(param.shape)
                else:
                    padded = torch.zeros(param.shape, dtype=loaded_weight.dtype, device=loaded_weight.device)
                    padded[:loaded_weight.numel()] = loaded_weight.flatten()
                    loaded_weight = padded

        elif "channel_scale" in param_name:
            loaded_weight = ensure_dtype(loaded_weight, param.dtype)
            if loaded_weight.dim() == 1:
                loaded_weight = loaded_weight.reshape(1, 1, -1)
            elif loaded_weight.dim() == 2:
                if loaded_weight.shape[0] == 1:
                    loaded_weight = loaded_weight.reshape(1, 1, -1)
            if loaded_weight.shape != param.shape:
                logger.warning(f"channel_scale shape mismatch: param {param.shape} vs loaded {loaded_weight.shape}")
                if loaded_weight.numel() == param.numel():
                    loaded_weight = loaded_weight.reshape(param.shape)
                else:
                    logger.warning(f"Cannot reshape channel_scale, using ones")
                    loaded_weight = torch.ones_like(param)

        # Final loading
        try:
            if param.numel() == 1 and loaded_weight.numel() == 1:
                param.data.fill_(loaded_weight.item())
            else:
                if param.size() != loaded_weight.size():
                    logger.error(
                        f"Final size mismatch for {param_name}: param {param.size()} vs loaded {loaded_weight.size()}")
                    raise AssertionError(
                        f"Attempted to load weight ({loaded_weight.size()}) into parameter ({param.size()})")
                param.data.copy_(loaded_weight)
            logger.info(f"Successfully loaded GBA weight: {param_name}")
        except Exception as e:
            logger.error(f"Failed to load GBA weight {param_name}: {e}")
            raise

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
                layer.qscales.size(0),
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
                layer.qscales.size(0),
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
        """获取层特定的量化配置 - 修复版本"""
        logger.info(f"Getting layer config for prefix: {layer_prefix}")

        if not hasattr(self.quant_config, 'strategy') or not self.quant_config.strategy:
            logger.debug("No strategy config available")
            return {}

        strategy = self.quant_config.strategy
        logger.debug(f"Available strategy keys: {list(strategy.keys())}")

        # 如果layer_prefix是类名（如"RowParallelLinear"），无法直接匹配策略
        # 这种情况下使用默认配置
        if layer_prefix in ["RowParallelLinear", "ColumnParallelLinear", "QKVParallelLinear",
                            "MergedColumnParallelLinear", "ReplicatedLinear"]:
            logger.info(f"Layer prefix is class name: {layer_prefix}, using default config")
            return {}

        # 解析完整的层路径，如 "model.layers.0.self_attn.q_proj"
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

        # 尝试层特定配置（JSON格式：model.layers.0.q_proj）
        if layer_num is not None and proj_type is not None:
            layer_key = f"model.layers.{layer_num}"
            logger.info(f"Looking for layer key: {layer_key}, proj type: {proj_type}")

            if layer_key in strategy:
                layer_config = strategy[layer_key]
                if proj_type in layer_config:
                    proj_config = layer_config[proj_type]
                    logger.info(f"Found layer-specific proj config: {proj_config}")

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
                            # 获取第一个可用的group_size值
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
                    logger.debug(f"Projection type {proj_type} not found in layer {layer_key}")
            else:
                logger.debug(f"Layer key {layer_key} not found in strategy")

        # 使用通用策略匹配（apply_quant_strategy函数）
        logger.debug("No layer-specific config found, trying generic strategy matching")

        generic_strategy = self.apply_quant_strategy(layer_prefix, strategy)
        if generic_strategy:
            logger.info(f"Found generic strategy: {generic_strategy}")

            # 转换策略格式到配置格式
            config = {}

            # 提取weight_bits
            if 'bits' in generic_strategy and generic_strategy['bits']:
                if isinstance(generic_strategy['bits'], list):
                    config['weight_bits'] = generic_strategy['bits'][0]
                else:
                    config['weight_bits'] = generic_strategy['bits']

            # 提取group_size - 修复访问逻辑
            if 'group_size' in generic_strategy:
                group_size_config = generic_strategy['group_size']
                if isinstance(group_size_config, dict):
                    weight_bits_str = str(config.get('weight_bits', 4))
                    if weight_bits_str in group_size_config:
                        config['group_size'] = group_size_config[weight_bits_str]
                    else:
                        # 如果没有对应的bits，取第一个值
                        config['group_size'] = next(iter(group_size_config.values()))
                else:
                    config['group_size'] = group_size_config

            logger.info(f"Extracted generic config: {config}")
            return config

        logger.info("No configuration found, using defaults")
        return {}

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Execute forward propagation"""

        # Ensure weights are processed
        if not hasattr(layer, '_gba_weights_processed'):
            self.process_weights_after_loading(layer)

        # perform channel_scale
        if hasattr(layer, 'channel_scale'):
            x = x.mul(layer.channel_scale)

        # Prepare parameters
        q_group_map = getattr(layer, "q_group_map", None)
        rows_info = getattr(layer, "rows_info", None)

        if rows_info is not None and rows_info.numel() > 0:
            rows_list = rows_info.tolist()
        else:
            rows_list = []  # Empty list instead of None

        # Call CUDA forward propagation
        output = ops.gba_linear_forward(
            x,
            layer.qweight,
            layer.scales,
            layer.zeros,
            layer.q_perm,
            layer.group_size,
            layer.weight_bits,
            self.quant_config.use_mbw,
            q_group_map,
            rows_list,
        )

        # Add bias
        if bias is not None:
            output = output + bias

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