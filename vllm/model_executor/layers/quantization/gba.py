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
        return ["quantize_config.json", "quant_strategy.json"]

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
            if "layer-mix" in model_name or "channel-mix" in model_name:
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
        """Create GBA quantized weight parameters"""

        output_size_per_partition = sum(output_partition_sizes)

        # Calculate quantization parameter dimensions
        group_size = self.quant_config.group_size
        if group_size == -1:
            group_size = input_size_per_partition

        num_groups = (input_size_per_partition + group_size - 1) // group_size

        # Determine weight shape based on whether mixed bit-width is used
        if not self.quant_config.use_mbw:
            # Standard quantization mode
            packed_rows = input_size_per_partition * self.quant_config.weight_bits // 32
            qweight_shape = (packed_rows, output_size_per_partition)
        else:
            # Mixed bit-width mode - use dynamic packing size
            qweight_shape = (input_size_per_partition // 32, output_size_per_partition)

        # Create quantized weights
        qweight = torch.nn.Parameter(
            torch.empty(qweight_shape, dtype=torch.int32, device="cuda"),
            requires_grad=False,
        )
        set_weight_attrs(qweight, {"input_dim": 0, "output_dim": 1})

        # Create quantization scales and zero points
        scale_zero_shape = (num_groups, output_size_per_partition)

        qscales = torch.nn.Parameter(
            torch.empty(scale_zero_shape, dtype=params_dtype, device="cuda"),
            requires_grad=False,
        )
        set_weight_attrs(qscales, {"input_dim": 0, "output_dim": 1})

        qzeros = torch.nn.Parameter(
            torch.empty(scale_zero_shape, dtype=params_dtype, device="cuda"),
            requires_grad=False,
        )
        set_weight_attrs(qzeros, {"input_dim": 0, "output_dim": 1})

        # Create permutation indices
        q_perm = torch.nn.Parameter(
            torch.empty(input_size_per_partition, dtype=torch.int16, device="cuda"),
            requires_grad=False,
        )
        set_weight_attrs(q_perm, {"input_dim": 0, "output_dim": -1})

        # Mixed bit-width mode requires additional parameters
        if self.quant_config.use_mbw:
            # Group information
            q_groups = torch.nn.Parameter(
                torch.empty(num_groups * 2, dtype=torch.int16, device="cuda"),
                requires_grad=False,
            )
            set_weight_attrs(q_groups, {"input_dim": -1, "output_dim": -1})
            layer.register_parameter("q_groups", q_groups)

            # Group mapping and row information (created in prepare_weights)
            layer.register_buffer("q_group_map", None)
            layer.register_buffer("rows_info", None)

        # Register all parameters
        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qscales", qscales)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("q_perm", q_perm)

        # Store configuration
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.group_size = group_size

        layer._gba_weights_initialized = False

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
        required_weights = ["qweight", "qscales", "qzeros", "q_perm"]
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
            layer.rows_info = rows

            # Create group mapping
            layer.q_group_map = ops.make_group_map(layer.q_groups, layer.qweight.size(0))

            logger.info(f"Applied mixed bitwidth quantization: {len(rows)} row groups")

        else:
            # Standard quantization mode
            qweight, _ = ops.gba_trans_qweight(
                layer.qweight,
                torch.empty(1, dtype=torch.int16, device=layer.qweight.device),
                False, # use_mbw=False
                layer.input_size_per_partition,
                layer.qscales.size(0),
                self.quant_config.weight_bits,
            )
            layer.qweight.data = qweight

            logger.info("Applied standard quantization")

        # Mark as processed
        layer._gba_weights_processed = True

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

        # Prepare parameters
        q_group_map = getattr(layer, "q_group_map", None)
        rows_info = getattr(layer, "rows_info", None)

        # Call CUDA forward propagation
        output = ops.gba_linear_forward(
            x,
            layer.qweight,
            layer.qscales,
            layer.qzeros,
            layer.q_perm,
            layer.group_size,
            self.quant_config.weight_bits,
            self.quant_config.use_mbw,
            q_group_map,
            rows_info,
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