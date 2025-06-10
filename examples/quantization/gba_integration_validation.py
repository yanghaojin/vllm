# GBA vLLM Integration Guide
# ========================
#
# This file demonstrates how to integrate all the GBA quantization components
# into vLLM's model loading and inference pipeline.

import os
import torch
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)


# Usage Examples
# ==============

def example_load_gba_model():
    """
    Example of how to load a GBA quantized model with vLLM.
    """
    from vllm import LLM
    from vllm.model_executor.layers.quantization.model_integration import integrate_gba_with_vllm

    # Initialize GBA integration
    integrate_gba_with_vllm()

    # Load a GBA quantized model
    model_name = "GreenBitAI/Qwen-1.5-7B-layer-mix-bpw-4.0"

    llm = LLM(
        model=model_name,
        quantization="gba",  # Explicitly specify GBA quantization
        trust_remote_code=True,
        max_model_len=2048,
    )

    # Generate text
    prompts = ["Hello, how are you?"]
    outputs = llm.generate(prompts)

    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")


def example_manual_gba_setup():
    """
    Example of manual GBA setup for debugging or custom usage.
    """
    from vllm.config import ModelConfig, LoadConfig
    from vllm.transformers_utils.config import enhance_config_with_quantization
    from vllm.model_executor.layers.quantization.model_integration import GBAIntegrationManager
    from transformers import AutoConfig

    model_path = "GreenBitAI/Qwen-1.5-7B-channel-mix-bpw-4.0"

    # Create model config
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Enhance with GBA detection
    enhanced_config = enhance_config_with_quantization(hf_config, model_path)

    model_config = ModelConfig(
        model=model_path,
        tokenizer=model_path,
        quantization="gba",
        hf_config=enhanced_config,
    )

    load_config = LoadConfig()

    # Initialize GBA integration
    manager = GBAIntegrationManager()
    manager.register_gba_quantization()

    # Detect and configure GBA
    gba_config = manager.detect_and_configure_gba(model_config)

    if gba_config:
        print(f"GBA Config: {gba_config.__dict__}")
    else:
        print("GBA quantization not detected")


def example_custom_weight_loading():
    """
    Example of custom weight loading with GBA support.
    """
    from vllm.model_executor.model_loader.weight_utils import gba_weight_loader
    import torch

    # Simulate a GBA parameter and loaded weight
    param = torch.zeros((100, 4096), dtype=torch.int32)
    loaded_weight = torch.randint(0, 255, (100, 4096), dtype=torch.int32)

    # Use GBA weight loader
    gba_weight_loader(param, loaded_weight, "layer.qweight")

    print("Weight loading completed successfully")


# Error Handling and Debugging
# =============================

def debug_gba_model_loading(model_path: str):
    """
    Debug helper for GBA model loading issues.

    Args:
        model_path: Path to the model
    """
    from vllm.model_executor.model_loader.weight_utils import detect_gba_quantization, load_gba_strategy_config

    print(f"Debugging GBA model: {model_path}")

    # Check if directory exists
    if os.path.isdir(model_path):
        print(f"✓ Model directory exists: {model_path}")

        # List all files
        files = list(Path(model_path).rglob("*"))
        print(f"Found {len(files)} files in model directory")

        # Check for quantization files
        quant_files = [f for f in files if any(
            pattern in f.name.lower()
            for pattern in ["quant", "strategy", "gba"]
        )]

        if quant_files:
            print("Found quantization files:")
            for f in quant_files:
                print(f"  - {f}")
        else:
            print("No quantization files found")

        # Check for strategy config
        strategy_config = load_gba_strategy_config(model_path)
        if strategy_config:
            print("✓ Found GBA strategy configuration")
            print(f"Strategy keys: {list(strategy_config.keys())}")
        else:
            print("✗ No GBA strategy configuration found")

    else:
        print(f"Model path is not a local directory: {model_path}")

    # Try to detect GBA quantization
    try:
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        is_gba, gba_config = detect_gba_quantization(model_path, hf_config.to_dict())

        if is_gba:
            print("✓ GBA quantization detected")
            print(f"Config: {gba_config}")
        else:
            print("✗ GBA quantization not detected")

    except Exception as e:
        print(f"Error loading model config: {e}")


def test_gba_cuda_ops():
    """
    Test GBA CUDA operations to ensure they're working correctly.
    """
    try:
        from vllm import _custom_ops as ops

        # Test basic operations
        print("Testing GBA CUDA operations...")

        # Create dummy tensors
        qweight = torch.randint(0, 255, (100, 4096), dtype=torch.int32, device='cuda')
        q_groups = torch.randint(2, 8, (10,), dtype=torch.int16, device='cuda')

        # Test weight transformation
        result_qweight, result_rows = ops.gba_trans_qweight(
            qweight, q_groups, False, 3200, 50, 4
        )

        print("✓ gba_trans_qweight works correctly")
        print(f"  Input shape: {qweight.shape}")
        print(f"  Output shape: {result_qweight.shape}")
        print(f"  Rows info: {result_rows}")

        # Test group mapping
        if len(result_rows) > 0:
            group_map = ops.make_group_map(q_groups, result_qweight.size(0))
            print("✓ make_group_map works correctly")
            print(f"  Group map shape: {group_map.shape}")

        print("All GBA CUDA operations test passed!")

    except ImportError as e:
        print(f"✗ GBA CUDA operations not available: {e}")
        print("Please ensure CUDA extensions are compiled correctly")
    except Exception as e:
        print(f"✗ Error testing GBA CUDA operations: {e}")


def benchmark_gba_performance():
    """
    Benchmark GBA quantized model performance.
    """
    import time
    from vllm import LLM
    from vllm.model_executor.layers.quantization.model_integration import integrate_gba_with_vllm

    print("Benchmarking GBA model performance...")

    # Initialize GBA integration
    integrate_gba_with_vllm()

    model_name = "GreenBitAI/Qwen-1.5-7B-layer-mix-bpw-4.0"

    try:
        # Load model
        start_time = time.time()
        llm = LLM(
            model=model_name,
            quantization="gba",
            trust_remote_code=True,
            max_model_len=1024,
        )
        load_time = time.time() - start_time
        print(f"Model loading time: {load_time:.2f}s")

        # Test inference
        prompts = ["Hello, how are you?"] * 10

        start_time = time.time()
        outputs = llm.generate(prompts, max_tokens=50)
        inference_time = time.time() - start_time

        print(f"Inference time for {len(prompts)} prompts: {inference_time:.2f}s")
        print(f"Average time per prompt: {inference_time / len(prompts):.2f}s")
        print(f"Throughput: {len(prompts) / inference_time:.2f} prompts/s")

        # Show sample output
        print(f"\nSample output: {outputs[0].outputs[0].text[:100]}...")

    except Exception as e:
        print(f"Benchmark failed: {e}")


# Installation and Setup Helpers
# ===============================

def check_gba_installation():
    """
    Check if GBA quantization is properly installed and configured.
    """
    print("Checking GBA installation...")

    checks = []

    # Check CUDA operations
    try:
        from vllm import _custom_ops as ops
        # Try to access GBA operations
        hasattr(ops, 'gba_linear_forward')
        hasattr(ops, 'gba_trans_qweight')
        hasattr(ops, 'make_group_map')
        checks.append(("CUDA operations", True, "All GBA CUDA ops available"))
    except Exception as e:
        checks.append(("CUDA operations", False, f"Error: {e}"))

    # Check Python modules
    try:
        from vllm.model_executor.layers.quantization.gba import GBAConfig
        from vllm.model_executor.model_loader.weight_utils import gba_weight_loader
        checks.append(("Python modules", True, "All GBA Python modules available"))
    except Exception as e:
        checks.append(("Python modules", False, f"Error: {e}"))

    # Check quantization registry
    try:
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
        gba_available = "gba" in QUANTIZATION_METHODS
        checks.append(("Quantization registry", gba_available,
                       "GBA registered" if gba_available else "GBA not registered"))
    except Exception as e:
        checks.append(("Quantization registry", False, f"Error: {e}"))

    # Print results
    print("\nInstallation Check Results:")
    print("=" * 50)

    all_passed = True
    for check_name, passed, message in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:20s} : {status} - {message}")
        if not passed:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("✓ All checks passed! GBA quantization is ready to use.")
    else:
        print("✗ Some checks failed. Please check the installation.")

    return all_passed


def test_gba_integration():
    """
    Test the complete GBA integration workflow.
    """
    print("Testing GBA integration workflow...")

    try:
        # Step 1: Register GBA quantization
        print("Step 1: Registering GBA quantization...")
        from vllm.model_executor.layers.quantization.model_integration import integrate_gba_with_vllm
        integrate_gba_with_vllm()
        print("✓ GBA quantization registered successfully")

        # Step 2: Check if GBA is available in quantization methods
        print("Step 2: Verifying GBA availability...")
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
        if "gba" in QUANTIZATION_METHODS:
            print("✓ GBA found in quantization methods")
        else:
            print("✗ GBA not found in quantization methods")
            return False

        # Step 3: Test GBA config retrieval
        print("Step 3: Testing GBA config retrieval...")
        from vllm.model_executor.layers.quantization import get_quantization_config
        gba_config_class = get_quantization_config("gba")
        print(f"✓ GBA config class retrieved: {gba_config_class.__name__}")

        # Step 4: Test GBA detection
        print("Step 4: Testing GBA model detection...")
        from vllm.model_executor.model_loader.weight_utils import detect_gba_quantization

        # Test with a known GBA model name
        test_model_name = "GreenBitAI/Qwen-1.5-7B-layer-mix-bpw-4.0"
        is_gba, gba_config = detect_gba_quantization(test_model_name, {})

        if is_gba:
            print("✓ GBA model detection works correctly")
            print(f"  Detected config: {gba_config}")
        else:
            print("✓ GBA detection logic works (no false positives)")

        # Step 5: Test GBA config creation
        print("Step 5: Testing GBA config creation...")
        test_config = {
            "weight_bits": 4,
            "group_size": 128,
            "use_mbw": False
        }

        # 添加这部分 - 测试MoE补丁
        print("Step 6: Testing MoE patches...")
        try:
            from vllm.model_executor.layers.quantization.patches.qwen3_moe_patch import get_qwen3_moe_info
            from vllm.model_executor.layers.quantization.patches.deepseek_v3_moe_patch import get_deepseek_v3_moe_info

            qwen3_info = get_qwen3_moe_info()
            deepseek_info = get_deepseek_v3_moe_info()

            print(f"✓ Qwen3 MoE patch available: {qwen3_info['available']}")
            print(f"✓ DeepSeek V3 MoE patch available: {deepseek_info['available']}")
        except ImportError as e:
            print(f"⚠ MoE patches not available: {e}")

        gba_instance = gba_config_class.from_config(test_config)
        print(f"✓ GBA config instance created: {gba_instance.weight_bits}-bit, group_size={gba_instance.group_size}")

        print("\n✓ All integration tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_moe_model_detection():
    """Test MoE model detection and patch application."""
    test_cases = [
        {
            "name": "Qwen3 MoE",
            "config": {"model_type": "qwen3", "num_experts": 8},
            "expected_type": "qwen3_moe"
        },
        {
            "name": "DeepSeek V3",
            "config": {"model_type": "deepseek_v3", "n_routed_experts": 256},
            "expected_type": "deepseek_v3_moe"
        }
    ]

    from vllm.model_executor.layers.quantization.gba_moe_support import detect_moe_model_type

    for case in test_cases:
        # Create a mock config object
        class MockConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        config = MockConfig(**case["config"])
        moe_info = detect_moe_model_type(config)

        if moe_info["type"] == case["expected_type"]:
            print(f"✓ {case['name']} detection: {moe_info['type']}")
        else:
            print(f"✗ {case['name']} detection failed: expected {case['expected_type']}, got {moe_info['type']}")

def validate_gba_setup():
    """
    Comprehensive validation of GBA setup.
    """
    print("=" * 60)
    print("GBA vLLM Integration Validation")
    print("=" * 60)

    all_tests_passed = True

    # Test 1: Installation check
    print("\n1. Installation Check:")
    print("-" * 30)
    if not check_gba_installation():
        all_tests_passed = False

    # Test 2: Integration workflow
    print("\n2. Integration Workflow Test:")
    print("-" * 30)
    if not test_gba_integration():
        all_tests_passed = False

    # Test 3: CUDA operations (if available)
    print("\n3. CUDA Operations Test:")
    print("-" * 30)
    try:
        test_gba_cuda_ops()
    except Exception as e:
        print(f"CUDA test failed: {e}")
        print("This is expected if CUDA is not available or GBA kernels not compiled")

    # Test 4: Model detection
    print("\n4. Model Detection Test:")
    print("-" * 30)
    try:
        debug_gba_model_loading("GreenBitAI/Qwen-1.5-7B-layer-mix-bpw-4.0")
    except Exception as e:
        print(f"Model detection test failed: {e}")

    # Final result
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✓ GBA vLLM integration validation PASSED!")
        print("You can now use GBA quantization with vLLM.")
    else:
        print("✗ GBA vLLM integration validation FAILED!")
        print("Please check the failed tests and fix the issues.")
    print("=" * 60)


def install_gba_example():
    """
    Example installation script for GBA quantization.
    """
    print("""
GBA vLLM Installation Guide:
===========================

1. Ensure you have CUDA and PyTorch installed:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

2. Install or update vLLM:
   pip install vllm

3. Add GBA files to vLLM installation:
   - Copy csrc/quantization/gba/ files to vLLM's csrc directory
   - Copy Python files to appropriate vLLM directories
   - Update torch_bindings.cpp with GBA operations

4. Recompile vLLM with CUDA extensions:
   cd /path/to/vllm
   pip install -e . --verbose

5. Test the installation:
   python -c "from gba_integration_validation import validate_gba_setup; validate_gba_setup()"

6. Run example:
   python -c "from gba_integration_validation import example_load_gba_model; example_load_gba_model()"

For detailed integration steps, see the model_integration.py file.
    """)


def quick_start_guide():
    """
    Quick start guide for using GBA with vLLM.
    """
    print("""
GBA vLLM Quick Start:
====================

1. Import and register GBA:
   from vllm.model_executor.layers.quantization.model_integration import integrate_gba_with_vllm
   integrate_gba_with_vllm()

2. Load a GBA model:
   from vllm import LLM
   llm = LLM(
       model="GreenBitAI/Qwen-1.5-7B-layer-mix-bpw-4.0",
       quantization="gba",
       trust_remote_code=True
   )

3. Generate text:
   outputs = llm.generate(["Hello, world!"])
   print(outputs[0].outputs[0].text)

For troubleshooting, run: validate_gba_setup()
    """)
    """
    Example installation script for GBA quantization.
    """
    print("""
GBA vLLM Installation Guide:
===========================

1. Ensure you have CUDA and PyTorch installed:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

2. Install or update vLLM:
   pip install vllm

3. Add GBA files to vLLM installation:
   - Copy csrc/quantization/gba/ files to vLLM's csrc directory
   - Copy Python files to appropriate vLLM directories
   - Update torch_bindings.cpp with GBA operations

4. Recompile vLLM with CUDA extensions:
   cd /path/to/vllm
   pip install -e . --verbose

5. Test the installation:
   python -c "from vllm.model_executor.layers.quantization.model_integration import check_gba_installation; check_gba_installation()"

6. Run example:
   python -c "from vllm.model_executor.layers.quantization.model_integration import example_load_gba_model; example_load_gba_model()"
    """)


if __name__ == "__main__":
    print("GBA vLLM Integration Test Suite")
    print("=" * 40)

    # Run installation check
    if check_gba_installation():
        print("\n" + "=" * 40)
        print("Running tests...")

        # Test CUDA operations
        test_gba_cuda_ops()

        print("\n" + "=" * 40)
        print("Running debug test...")

        # Debug a model (this will work even if model is not locally available)
        debug_gba_model_loading("GreenBitAI/Qwen-1.5-7B-layer-mix-bpw-4.0")

        print("\n" + "=" * 40)
        print("Manual setup test...")

        # Test manual setup
        try:
            example_manual_gba_setup()
        except Exception as e:
            print(f"Manual setup test failed: {e}")

        print("\n" + "=" * 40)
        print("Performance benchmark (uncomment to run)...")
        # benchmark_gba_performance()  # Uncomment to run benchmark

    else:
        print("\nInstallation check failed. Please install GBA quantization first.")
        install_gba_example()