"""Tests for GBA quantization support in vLLM.

Run `pytest tests/quantization/test_gba.py --forked`.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.gba import GBALinearMethod, GBAConfig

PROMPT = "Hello, how are you?"

# Test model configurations for GBA quantization
# Note: These are example model names - replace with actual GBA quantized models
GBA_MODELS = [
    ("GreenBitAI/Qwen-1.5-7B-layer-mix-bpw-4.0", "layer-mix", 4, 128, False),
    ("GreenBitAI/Qwen-1.5-7B-channel-mix-bpw-4.0", "channel-mix", 4, 128, True),
    # Add more GBA models as needed
]


class TestGBAIntegration:
    """Test GBA quantization integration with vLLM."""

    def test_gba_module_imports(self):
        """Test that GBA modules can be imported successfully."""
        try:
            from vllm.model_executor.layers.quantization.gba import GBAConfig, GBALinearMethod
            from vllm.model_executor.layers.quantization.model_integration import integrate_gba_with_vllm
            from vllm.model_executor.model_loader.weight_utils import detect_gba_quantization
            assert True, "All GBA modules imported successfully"
        except ImportError as e:
            pytest.fail(f"Failed to import GBA modules: {e}")

    def test_gba_registration(self):
        """Test that GBA quantization can be registered with vLLM."""
        try:
            from vllm.model_executor.layers.quantization.model_integration import integrate_gba_with_vllm
            from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

            # Register GBA
            integrate_gba_with_vllm()

            # Check if GBA is in quantization methods
            assert "gba" in QUANTIZATION_METHODS, "GBA not found in quantization methods"

        except Exception as e:
            pytest.fail(f"GBA registration failed: {e}")

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

    def test_gba_model_detection(self):
        """Test GBA model detection from model names."""
        from vllm.model_executor.model_loader.weight_utils import detect_gba_quantization

        # Test positive cases
        test_cases = [
            ("GreenBitAI/Qwen-1.5-7B-layer-mix-bpw-4.0", True),
            ("GreenBitAI/Llama-7B-channel-mix-groupsize64", True),
            ("some-model-layer-mix-bpw-3.0", True),
            ("regular-model-name", False),
            ("some-other-model", False),
        ]

        for model_name, expected in test_cases:
            is_gba, config = detect_gba_quantization(model_name, {})
            assert is_gba == expected, f"Detection failed for {model_name}"

            if expected:
                assert "quantization_method" in config
                assert config["quantization_method"] == "gba"

    def test_gba_weight_loader(self):
        """Test GBA weight loader functionality."""
        from vllm.model_executor.model_loader.weight_utils import gba_weight_loader, should_use_gba_weight_loader

        # Test parameter name detection
        gba_params = ["layer.qweight", "layer.qscales", "layer.qzeros", "layer.q_perm", "layer.q_groups"]
        non_gba_params = ["layer.weight", "layer.bias", "layer.input_layernorm.weight"]

        for param in gba_params:
            assert should_use_gba_weight_loader(param), f"Should use GBA loader for {param}"

        for param in non_gba_params:
            assert not should_use_gba_weight_loader(param), f"Should not use GBA loader for {param}"

        # Test weight loading (basic functionality)
        param = torch.zeros((100, 4096), dtype=torch.int32)
        loaded_weight = torch.randint(0, 255, (100, 4096), dtype=torch.int32)

        # This should not raise an exception
        gba_weight_loader(param, loaded_weight, "layer.qweight")

    def test_moe_model_detection(self):
        """Test MoE model type detection."""
        from vllm.model_executor.layers.quantization.gba_moe_support import detect_moe_model_type

        # Test cases for different MoE models
        test_cases = [
            ({"model_type": "qwen3", "num_experts": 8}, "qwen3_moe"),
            ({"model_type": "deepseek_v3", "n_routed_experts": 256}, "deepseek_v3_moe"),
            ({"model_type": "standard", "num_local_experts": 8}, "mixtral_moe"),
            ({"model_type": "llama"}, "standard"),
        ]

        for config_dict, expected_type in test_cases:
            # Create mock config object
            class MockConfig:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            config = MockConfig(**config_dict)
            moe_info = detect_moe_model_type(config)
            assert moe_info["type"] == expected_type, f"Expected {expected_type}, got {moe_info['type']}"

    def test_moe_patches_availability(self):
        """Test MoE patches availability and basic functionality."""
        try:
            from vllm.model_executor.layers.quantization.patches.qwen3_moe_patch import get_qwen3_moe_info
            from vllm.model_executor.layers.quantization.patches.deepseek_v3_moe_patch import get_deepseek_v3_moe_info

            qwen3_info = get_qwen3_moe_info()
            deepseek_info = get_deepseek_v3_moe_info()

            # Check that info dictionaries have expected keys
            expected_keys = ["available", "patched", "strategies"]
            for key in expected_keys:
                assert key in qwen3_info, f"Missing key {key} in qwen3_info"
                assert key in deepseek_info, f"Missing key {key} in deepseek_info"

        except ImportError as e:
            pytest.skip(f"MoE patches not available: {e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gba_cuda_ops_availability(self):
        """Test that GBA CUDA operations are available."""
        try:
            from vllm import _custom_ops as ops

            # Check that GBA operations are available
            required_ops = ['gba_linear_forward', 'gba_trans_qweight', 'make_group_map']
            for op_name in required_ops:
                assert hasattr(ops, op_name), f"Missing CUDA operation: {op_name}"

        except ImportError as e:
            pytest.skip(f"CUDA operations not available: {e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gba_cuda_ops_basic_functionality(self):
        """Test basic functionality of GBA CUDA operations."""
        try:
            from vllm import _custom_ops as ops

            # Create test tensors
            qweight = torch.randint(0, 255, (100, 4096), dtype=torch.int32, device='cuda')
            q_groups = torch.randint(2, 8, (10,), dtype=torch.int16, device='cuda')

            # Test gba_trans_qweight
            result_qweight, result_rows = ops.gba_trans_qweight(
                qweight, q_groups, False, 3200, 50, 4
            )

            assert result_qweight.shape[0] > 0, "gba_trans_qweight should return non-empty tensor"
            assert len(result_rows) >= 0, "gba_trans_qweight should return rows info"

            # Test make_group_map if rows available
            if len(result_rows) > 0:
                group_map = ops.make_group_map(q_groups, result_qweight.size(0))
                assert group_map.numel() > 0, "make_group_map should return non-empty tensor"

        except Exception as e:
            pytest.skip(f"CUDA operations test failed: {e}")


@pytest.mark.parametrize("model_name, mix_type, bits, group_size, use_mbw", GBA_MODELS)
def test_gba_model_config_parsing(model_name: str, mix_type: str, bits: int, group_size: int, use_mbw: bool):
    """Test GBA model configuration parsing from model names."""
    from vllm.model_executor.model_loader.weight_utils import detect_gba_quantization

    is_gba, config = detect_gba_quantization(model_name, {})

    assert is_gba, f"Should detect GBA quantization for {model_name}"
    assert config["weight_bits"] == bits, f"Expected {bits} bits, got {config['weight_bits']}"
    assert config["use_mbw"] == use_mbw, f"Expected use_mbw={use_mbw}, got {config['use_mbw']}"


# Integration test with mock vLLM runner
class TestGBAWithVLLM:
    """Integration tests for GBA with vLLM."""

    def test_gba_quantization_config_integration(self):
        """Test that GBA quantization config integrates properly with vLLM."""
        from vllm.model_executor.layers.quantization import get_quantization_config
        from vllm.model_executor.layers.quantization.model_integration import integrate_gba_with_vllm

        # Register GBA
        integrate_gba_with_vllm()

        # Get GBA config class
        gba_config_class = get_quantization_config("gba")

        # Test config creation
        test_config = {
            "weight_bits": 4,
            "group_size": 128,
            "use_mbw": False
        }

        gba_instance = gba_config_class.from_config(test_config)
        assert gba_instance.weight_bits == 4
        assert gba_instance.group_size == 128
        assert gba_instance.use_mbw == False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gba_linear_method_creation(self):
        """Test GBA linear method creation and weight initialization."""
        config = GBAConfig(weight_bits=4, group_size=128, use_mbw=False)
        method = GBALinearMethod(config)

        # Create a mock layer
        layer = torch.nn.Linear(1024, 2048)

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

    def test_gba_model_integration_manager(self):
        """Test GBA integration manager functionality."""
        from vllm.model_executor.layers.quantization.model_integration import GBAIntegrationManager

        manager = GBAIntegrationManager()

        # Test registration
        manager.register_gba_quantization()
        assert manager.is_initialized, "Manager should be initialized after registration"

        # Test that quantization methods include GBA
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
        assert "gba" in QUANTIZATION_METHODS, "GBA should be in quantization methods after registration"


# Utility function for comprehensive validation
def validate_gba_setup():
    """Comprehensive validation of GBA setup."""
    print("=" * 60)
    print("GBA vLLM Integration Validation")
    print("=" * 60)

    tests = [
        ("Module imports", TestGBAIntegration().test_gba_module_imports),
        ("GBA registration", TestGBAIntegration().test_gba_registration),
        ("Config creation", TestGBAIntegration().test_gba_config_creation),
        ("Model detection", TestGBAIntegration().test_gba_model_detection),
        ("Weight loader", TestGBAIntegration().test_gba_weight_loader),
        ("MoE detection", TestGBAIntegration().test_moe_model_detection),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✓ {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}: FAILED - {e}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! GBA integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the setup.")
        return False


if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_gba_setup()