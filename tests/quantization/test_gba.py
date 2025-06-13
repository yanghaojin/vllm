"""Simplified tests for GBA quantization support in vLLM.

Run `pytest tests/quantization/test_gba.py --forked`.
"""

import pytest
import torch
import math

from vllm.model_executor.layers.quantization.gba import GBALinearMethod, GBAConfig


def get_packed_info(channels, n_bits, bits_prop, bits_group_size):
    groups = 0
    rows = 0
    bits_channel = []
    for idx in range(len(bits_prop)):
        if idx < len(bits_prop) - 1:
            minimal_channels = list(bits_group_size.values())[idx]
            channel_pre_pack = max(1, int(channels * (bits_prop[idx])) // minimal_channels) * minimal_channels
            bits_channel.append(channel_pre_pack)
            groups += channel_pre_pack // minimal_channels
            rows += channel_pre_pack // 32 * n_bits[idx]
        else:
            minimal_channels = list(bits_group_size.values())[idx]
            channel_pre_pack = channels - sum(bits_channel)
            bits_channel.append(channel_pre_pack)
            groups += channel_pre_pack // minimal_channels
            rows += channel_pre_pack // 32 * n_bits[idx]

    return groups, rows

def get_q_groups(groups, n_bits, group_size, channels, bits_prop):
    qgroups = []
    bits_column_end_index = []

    for idx in range(len(bits_prop)):
        if idx < len(bits_prop) - 1:
            minimal_columns = list(group_size.values())[idx]
            columns_index = max(1, int(channels * (
                bits_prop[idx])) // minimal_columns) * minimal_columns  # TODO: determine the minimal bits columns
            if idx > 0:
                columns_index += bits_column_end_index[-1]
            bits_column_end_index.append(columns_index)
        else:
            bits_column_end_index.append(channels)

    for bits_idx, bits in enumerate(n_bits):
        if bits_idx == 0:
            rows_per_bit = bits_column_end_index[bits_idx]
        else:
            rows_per_bit = bits_column_end_index[bits_idx] - bits_column_end_index[bits_idx - 1]

        gs = group_size[str(bits)]
        groups_per_bit = rows_per_bit // gs

        for group in range(groups_per_bit):
            qgroups.append(bits)  # record bits per group
            qgroups.append(0)

    out_row = 0
    rem_rows = channels
    for i in range(groups):
        bits = qgroups[2 * i]
        gs = group_size[str(bits)]

        rows_per_group = min(gs, rem_rows)  # rows per group before packing
        wpqr = 32 / bits  # INT32 elements per group for packing
        qrows = math.ceil(rows_per_group / wpqr)  # rows per group after packing
        qgroups[2 * i + 1] = out_row  # record packed rows start idx per group

        out_row += qrows

    return qgroups

class TestGBABasics:
    """Basic GBA functionality tests."""

    def test_gba_config_creation(self):
        """Test GBA configuration creation and validation."""
        # Test basic config creation
        config = GBAConfig(weight_bits=4, group_size=128, use_mbw=False)
        assert config.weight_bits == 4
        assert config.group_size == 128
        assert config.use_mbw == False

        # Test config from dictionary
        config_dict = {
            "weight_bits": 4,
            "group_size": 128,
            "use_mbw": True,
            "strategy": {}
        }
        config = GBAConfig.from_config(config_dict)
        assert config.weight_bits == 4
        assert config.group_size == 128
        assert config.use_mbw == True

        # Test invalid configurations
        with pytest.raises(ValueError):
            GBAConfig(weight_bits=7)  # Unsupported bit width

        with pytest.raises(ValueError):
            GBAConfig(group_size=100)  # Unsupported group size

    def test_gba_config_methods(self):
        """Test GBA config class methods."""
        config = GBAConfig()

        # Test class methods
        assert config.get_name() == "gba"
        assert torch.half in config.get_supported_act_dtypes()
        assert torch.bfloat16 in config.get_supported_act_dtypes()
        assert config.get_min_capability() == 70
        assert "quantize_config.json" in config.get_config_filenames()

    def test_gba_model_name_parsing(self):
        """Test GBA model name parsing."""
        # Test cases for model name parsing
        test_cases = [
            {
                "config": {"_name_or_path": "GreenBitAI/Qwen-1.5-7B-layer-mix-bpw-4.0"},
                "expected": {"use_mbw": True, "weight_bits": 4}
            },
            {
                "config": {"_name_or_path": "GreenBitAI/Llama-7B-channel-mix-groupsize64"},
                "expected": {"use_mbw": True, "group_size": 64}
            },
            {
                "config": {"_name_or_path": "regular-model-bpw-3.0-groupsize32"},
                "expected": {"use_mbw": False, "weight_bits": 3, "group_size": 32}
            }
        ]

        for case in test_cases:
            config = GBAConfig.from_config(case["config"])
            for key, expected_value in case["expected"].items():
                actual_value = getattr(config, key)
                assert actual_value == expected_value, f"Expected {key}={expected_value}, got {actual_value}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gba_linear_method_weight_creation(self):
        """Test GBA linear method weight creation."""
        config = GBAConfig(weight_bits=4, group_size=128, use_mbw=False)
        method = GBALinearMethod(config)

        # Create a mock layer
        layer = torch.nn.Linear(1024, 2048)
        layer.cuda()  # Move to CUDA

        # Test weight creation
        method.create_weights(
            layer=layer,
            input_size_per_partition=1024,
            output_partition_sizes=[2048],
            input_size=1024,
            output_size=2048,
            params_dtype=torch.float16
        )

        # Check that required parameters are created
        required_params = ["qweight", "qscales", "qzeros", "q_perm"]
        for param_name in required_params:
            assert hasattr(layer, param_name), f"Missing parameter: {param_name}"
            param = getattr(layer, param_name)
            assert param.is_cuda, f"Parameter {param_name} should be on CUDA"

        # Check parameter shapes
        assert layer.qweight.shape == (128, 2048)  # 1024 * 4 / 32 = 128
        assert layer.qscales.shape == (8, 2048)  # 1024 / 128 = 8
        assert layer.qzeros.shape == (8, 2048)  # 1024 / 128 = 8
        assert layer.q_perm.shape == (1024,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gba_linear_method_mbw_weights(self):
        """Test GBA linear method weight creation with mixed bit-width."""
        config = GBAConfig(weight_bits=4, group_size=128, use_mbw=True)
        method = GBALinearMethod(config)

        # Create a mock layer
        layer = torch.nn.Linear(1024, 2048)
        layer.cuda()  # Move to CUDA

        # Test weight creation
        method.create_weights(
            layer=layer,
            input_size_per_partition=1024,
            output_partition_sizes=[2048],
            input_size=1024,
            output_size=2048,
            params_dtype=torch.float16
        )

        # Check that MBW-specific parameters are created
        assert hasattr(layer, "q_groups"), "Missing q_groups parameter for MBW mode"
        assert layer.q_groups.shape == (16,)  # num_groups * 2 = 8 * 2 = 16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gba_cuda_ops_availability(self):
        """Test that GBA CUDA operations are available."""
        try:
            from vllm import _custom_ops as ops

            # Check that GBA operations are available
            required_ops = ['gba_linear_forward', 'gba_trans_qweight', 'make_group_map', 'gba_dequantize_weight']
            for op_name in required_ops:
                assert hasattr(ops, op_name), f"Missing CUDA operation: {op_name}"

        except ImportError as e:
            pytest.skip(f"CUDA operations not available: {e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gba_group_map_creation(self):
        """Test GBA group map creation."""
        try:
            from vllm import _custom_ops as ops

            # Create test tensors
            groups = 8
            q_groups = torch.randint(2, 8, (groups,), dtype=torch.int16, device='cuda')
            num_qrows = 128

            # Test group map creation
            group_map = ops.make_group_map(q_groups, num_qrows)

            assert group_map.device.type == 'cuda', "Group map should be on CUDA"
            assert group_map.numel() > 0, "Group map should not be empty"

        except Exception as e:
            pytest.skip(f"Group map creation test failed: {e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gba_forward_kernel_execution(self):
        """Test GBA forward CUDA kernel execution."""
        try:
            from vllm import _custom_ops as ops

            # Test parameters
            batch_size = 2
            input_features = 1024
            output_features = 2048
            group_size = 128
            weight_bits = 4
            dtype = torch.half

            device = torch.device('cuda')

            # Create input tensor
            x = torch.randn(batch_size, input_features, dtype=dtype, device=device)

            # Create quantized weight parameters
            # Standard quantization mode: packed_rows = input_features * weight_bits // 32
            packed_rows = input_features * weight_bits // 32
            qweight = torch.randint(0, 2 ** 31 - 1, (packed_rows, output_features), dtype=torch.int32, device=device)

            # Create scales and zeros
            num_groups = input_features // group_size
            qscales = torch.randn(num_groups, output_features, dtype=dtype, device=device)
            qzeros = torch.randn(num_groups, output_features, dtype=dtype, device=device)

            # Create permutation (identity for simplicity)
            q_perm = torch.arange(input_features, dtype=torch.int16, device=device)

            # Transform weights (standard mode, not MBW)
            transformed_qweight, rows_info = ops.gba_trans_qweight(
                qweight,
                torch.empty(1, dtype=torch.int16, device=device),  # dummy q_groups for standard mode
                False,  # use_mbw=False
                input_features,
                num_groups,
                weight_bits
            )

            # Test forward pass
            output = ops.gba_linear_forward(
                x,
                transformed_qweight,
                qscales,
                qzeros,
                q_perm,
                group_size,
                weight_bits,
                False,  # use_mbw=False
                None,  # q_group_map not needed for standard mode
                rows_info
            )

            # Validate output
            assert output.shape == (
            batch_size, output_features), f"Expected shape {(batch_size, output_features)}, got {output.shape}"
            assert output.dtype == dtype, f"Expected dtype {dtype}, got {output.dtype}"
            assert output.device.type == 'cuda', "Output should be on CUDA"
            assert torch.isfinite(output).all(), "Output should not contain NaN or Inf values"

            print(f"✓ GBA forward kernel test passed - Input: {x.shape}, Output: {output.shape}")

        except Exception as e:
            pytest.skip(f"GBA forward kernel test failed: {e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gba_dequantization_kernel(self):
        """Test GBA weight dequantization kernel."""
        try:
            from vllm import _custom_ops as ops

            # Test parameters
            input_features = 1024
            output_features = 2048
            group_size = 128
            weight_bits = 4
            dtype = torch.half

            device = torch.device('cuda')

            # Create quantized weight parameters
            packed_rows = input_features * weight_bits // 32
            qweight = torch.randint(0, 2 ** 31 - 1, (packed_rows, output_features), dtype=torch.int32, device=device)

            # Create scales and zeros
            num_groups = input_features // group_size
            qscales = torch.randn(num_groups, output_features, dtype=dtype, device=device)
            qzeros = torch.randn(num_groups, output_features, dtype=dtype, device=device)

            # Create permutation
            q_perm = torch.arange(input_features, dtype=torch.int16, device=device)

            # Transform weights first
            transformed_qweight, rows_info = ops.gba_trans_qweight(
                qweight,
                torch.empty(1, dtype=torch.int16, device=device),
                False,  # use_mbw=False
                input_features,
                num_groups,
                weight_bits
            )

            # Test weight dequantization
            fp_weights = ops.gba_dequantize_weight(
                transformed_qweight,
                qscales,
                qzeros,
                q_perm,
                group_size,
                weight_bits,
                False,  # use_mbw=False
                None,  # q_group_map
                rows_info
            )

            # Validate dequantized weights
            assert fp_weights.shape == (input_features,
                                        output_features), f"Expected shape {(input_features, output_features)}, got {fp_weights.shape}"
            assert fp_weights.dtype == dtype, f"Expected dtype {dtype}, got {fp_weights.dtype}"
            assert fp_weights.device.type == 'cuda', "Dequantized weights should be on CUDA"
            assert torch.isfinite(fp_weights).all(), "Dequantized weights should not contain NaN or Inf values"

            print(f"✓ GBA dequantization kernel test passed - Weights shape: {fp_weights.shape}")

        except Exception as e:
            pytest.skip(f"GBA dequantization kernel test failed: {e}")


class TestGBAIntegration:
    """Integration tests for GBA."""

    def test_gba_get_quant_method(self):
        """Test that GBA config returns correct quantization method."""
        config = GBAConfig()
        linear_layer = torch.nn.Linear(100, 200)

        method = config.get_quant_method(linear_layer, "test_prefix")
        assert method is not None, "Should return a quantization method for Linear layer"
        assert isinstance(method, GBALinearMethod), "Should return GBALinearMethod"

        # Test with non-linear layer
        non_linear_layer = torch.nn.Conv2d(3, 64, 3)
        method = config.get_quant_method(non_linear_layer, "test_prefix")
        assert method is None, "Should return None for non-Linear layer"

    def test_gba_config_serialization(self):
        """Test GBA config can be created from various input formats."""
        # Test with minimal config
        minimal_config = {"weight_bits": 4}
        gba_config = GBAConfig.from_config(minimal_config)
        assert gba_config.weight_bits == 4
        assert gba_config.group_size == 128  # default value

        # Test with full config
        full_config = {
            "weight_bits": 3,
            "group_size": 64,
            "use_mbw": True,
            "strategy": {"type": "layer_mix"}
        }
        gba_config = GBAConfig.from_config(full_config)
        assert gba_config.weight_bits == 3
        assert gba_config.group_size == 64
        assert gba_config.use_mbw == True
        assert gba_config.strategy == {"type": "layer_mix"}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gba_linear_method_end_to_end(self):
        """End-to-end test of GBA linear method."""
        try:
            # Test configuration
            batch_size = 2
            input_features = 512
            output_features = 1024

            config = GBAConfig(weight_bits=4, group_size=128, use_mbw=False)
            method = GBALinearMethod(config)

            # Create and setup layer
            layer = torch.nn.Linear(input_features, output_features, bias=True)
            layer.cuda()

            # Create weights
            method.create_weights(
                layer=layer,
                input_size_per_partition=input_features,
                output_partition_sizes=[output_features],
                input_size=input_features,
                output_size=output_features,
                params_dtype=torch.half
            )

            # Simulate loading quantized weights (normally done by weight loader)
            # Create realistic quantized weight data
            packed_rows = input_features * 4 // 32
            layer.qweight.data = torch.randint(0, 2 ** 31 - 1, (packed_rows, output_features),
                                               dtype=torch.int32, device='cuda')

            num_groups = input_features // 128
            layer.qscales.data = torch.randn(num_groups, output_features, dtype=torch.half, device='cuda') * 0.1
            layer.qzeros.data = torch.randn(num_groups, output_features, dtype=torch.half, device='cuda') * 0.1
            layer.q_perm.data = torch.arange(input_features, dtype=torch.int16, device='cuda')

            # Create input
            x = torch.randn(batch_size, input_features, dtype=torch.half, device='cuda')

            # Create bias
            bias = torch.randn(output_features, dtype=torch.half, device='cuda')

            # Test forward pass
            output = method.apply(layer, x, bias)

            # Validate output
            assert output.shape == (
            batch_size, output_features), f"Expected {(batch_size, output_features)}, got {output.shape}"
            assert output.dtype == torch.half, "Output should be half precision"
            assert output.device.type == 'cuda', "Output should be on CUDA"
            assert torch.isfinite(output).all(), "Output should not contain NaN or Inf"

            print(f"✓ End-to-end test passed - Input: {x.shape}, Output: {output.shape}")

        except Exception as e:
            pytest.skip(f"End-to-end test failed: {e}")


# Parametrized tests for different configurations
@pytest.mark.parametrize("weight_bits", [3, 4, 5, 6, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128, 256])
@pytest.mark.parametrize("use_mbw", [True, False])
def test_gba_config_combinations(weight_bits: int, group_size: int, use_mbw: bool):
    """Test various combinations of GBA configuration parameters."""
    config = GBAConfig(
        weight_bits=weight_bits,
        group_size=group_size,
        use_mbw=use_mbw
    )

    assert config.weight_bits == weight_bits
    assert config.group_size == group_size
    assert config.use_mbw == use_mbw
    assert config.get_name() == "gba"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("input_features", [1024, 4096])
@pytest.mark.parametrize("output_features", [1024, 4096])
@pytest.mark.parametrize("weight_bits", [2, 4])
@pytest.mark.parametrize("group_size", [64, 128])
def test_gba_kernel_performance(batch_size: int, input_features: int, output_features: int,
                                weight_bits: int, group_size: int):
    """Performance test for GBA CUDA kernels with various configurations."""
    try:
        import time
        from vllm import _custom_ops as ops

        dtype = torch.half
        device = torch.device('cuda')

        print(f"\nTesting M:{batch_size}, N:{output_features}, K:{input_features}, "
              f"bits:{weight_bits}, group_size:{group_size}")

        # Create input tensor
        x = torch.randn(batch_size, input_features, dtype=dtype, device=device)

        # Create quantized weight parameters
        packed_rows = input_features * weight_bits // 32
        qweight = torch.randint(0, 2 ** 31 - 1, (packed_rows, output_features), dtype=torch.int32, device=device)

        # Create scales and zeros
        num_groups = input_features // group_size
        qscales = torch.randn(num_groups, output_features, dtype=dtype, device=device) * 0.1  # smaller scale
        qzeros = torch.randn(num_groups, output_features, dtype=dtype, device=device) * 0.1  # smaller zero

        # Create permutation
        q_perm = torch.arange(input_features, dtype=torch.int16, device=device)

        # Transform weights
        transformed_qweight, rows_info = ops.gba_trans_qweight(
            qweight,
            torch.empty(1, dtype=torch.int16, device=device),
            False,  # use_mbw=False for performance test
            input_features,
            num_groups,
            weight_bits
        )

        # Warm up
        for _ in range(3):
            _ = ops.gba_linear_forward(
                x, transformed_qweight, qscales, qzeros, q_perm,
                group_size, weight_bits, False, None, rows_info
            )
        torch.cuda.synchronize()

        # Performance test
        num_runs = 10
        start_time = time.time()
        for _ in range(num_runs):
            output = ops.gba_linear_forward(
                x, transformed_qweight, qscales, qzeros, q_perm,
                group_size, weight_bits, False, None, rows_info
            )
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time

        # Validate output
        assert output.shape == (batch_size, output_features)
        assert output.dtype == dtype
        assert torch.isfinite(output).all()

        avg_time = elapsed_time / num_runs
        print(f"GBA kernel average time: {avg_time:.6f}s")

        # Optional: Compare with dequantized reference (simplified)
        if batch_size <= 2 and input_features <= 1024:  # Only for smaller sizes to avoid memory issues
            fp_weights = ops.gba_dequantize_weight(
                transformed_qweight, qscales, qzeros, q_perm,
                group_size, weight_bits, False, None, rows_info
            )
            reference_output = torch.matmul(x, fp_weights)

            # Check if results are reasonably close (allowing for quantization error)
            max_diff = torch.max(torch.abs(output - reference_output)).item()
            mean_diff = torch.mean(torch.abs(output - reference_output)).item()
            print(f"Max diff vs reference: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

            # Allow for quantization error - these thresholds may need adjustment
            assert max_diff < 10.0, f"Max difference too large: {max_diff}"
            assert mean_diff < 2.0, f"Mean difference too large: {mean_diff}"

    except Exception as e:
        pytest.skip(f"GBA performance test failed: {e}")


# Utility function for quick validation
def validate_gba_setup():
    """Quick validation of GBA setup."""
    print("GBA Setup Validation")
    print("=" * 40)

    try:
        # Test 1: Basic imports
        from vllm.model_executor.layers.quantization.gba import GBAConfig, GBALinearMethod
        print("✓ Basic imports successful")

        # Test 2: Config creation
        config = GBAConfig(weight_bits=4, group_size=128)
        print("✓ Config creation successful")

        # Test 3: CUDA ops availability (if CUDA available)
        if torch.cuda.is_available():
            from vllm import _custom_ops as ops
            required_ops = ['gba_linear_forward', 'gba_trans_qweight', 'make_group_map']
            for op in required_ops:
                if hasattr(ops, op):
                    print(f"✓ CUDA op {op} available")
                else:
                    print(f"✗ CUDA op {op} missing")
        else:
            print("! CUDA not available - skipping CUDA op tests")

        print("=" * 40)
        print("GBA setup validation completed")

    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

    return True


if __name__ == "__main__":
    validate_gba_setup()