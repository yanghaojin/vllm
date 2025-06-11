# GBA Quantization for vLLM

This directory contains the implementation of GreenBitAI's quantization support for vLLM, enabling efficient inference of mixed bit-width quantized models.

## Overview

GBA quantization supports:
- **Ultra low bit quantization** (2-4 bits)
- **Layer-wise and channel-wise quantization strategies**
- **Optimized CUDA kernels** for high-performance inference
- **Automatic model detection** from model names and configurations

## File Structure

```
vllm/
├── model_executor/
│   ├── layers/
│   │   └── quantization/
│   │       ├── gba_moe_support.py        # GBA moe related support methods
│   │       ├── gba.py                    # Core GBA quantization implementation
│   │       ├── model_integration.py     # Model integration and layer replacement
│   │       └── patches/                  # MoE model patches
│   │           ├── __init__.py
│   │           ├── qwen3_moe_patch.py
│   │           └── deepseek_v3_moe_patch.py
│   └── model_loader/
│       └──weight_utils.py                   # Enhanced weight loading (modified)
├── transformers_utils/
│   └── config.py                         # Configuration detection (modified)
├── csrc/
│   ├── quantization/
│   │   └── gba/
│   │       ├── gba_ops.h                # CUDA operations header
│   │       ├── gba_ops.cu               # CUDA kernel implementations
│   │       └── exl2/...                 # Exl2 cuda kernel implementations
│   └── torch_bindings.cpp               # PyTorch bindings (modified)
├── examples/
│    └── quantization/
│        ├── gba_integration_validation.py # Integration tests and examples
│        └── README.md                     # This file
└── tests/
    └── quantization/
        └── test_gba.py                  # unit tests
```

## Quick Start

### 1. Installation

Ensure GBA CUDA extensions are compiled:
```bash
pip install -e . --verbose
```

### 2. Basic Usage

```python
from vllm import LLM
from vllm.model_executor.layers.quantization.model_integration import integrate_gba_with_vllm

# Initialize GBA support
integrate_gba_with_vllm()

# Load GBA quantized model
llm = LLM(
    model="GreenBitAI/Qwen-1.5-7B-layer-mix-bpw-4.0",
    quantization="gba",
    trust_remote_code=True
)

# Generate text
outputs = llm.generate(["Hello, how are you?"])
print(outputs[0].outputs[0].text)
```

### 3. Supported Models

GBA quantization automatically detects models with these naming patterns:
- `GreenBitAI/*` - Official GreenBitAI models
- `*-layer-mix-*` - Layer-wise mixed quantization
- `*-channel-mix-*` - Channel-wise mixed quantization  
- `*-bpw-X.X` - Specific bit-width (e.g., bpw-4.0)
- `*-groupsize128` - Group size specifications

## Configuration

### Automatic Detection
```python
# Model name parsing examples:
"GreenBitAI/Qwen-1.5-7B-layer-mix-bpw-4.0"     # 4-bit layer-wise
"GreenBitAI/Llama-7B-channel-mix-groupsize64"   # Channel-wise, group size 64
```

### Manual Configuration
```python
from vllm.model_executor.layers.quantization.gba import GBAConfig

config = GBAConfig(
    weight_bits=4,
    group_size=128,
    use_mbw=True,  # Mixed bit-width
    strategy={}    # Optional strategy dict
)
```

## Testing

Run the validation script to test your installation:
```python
python examples/quantization/gba_integration_validation.py
```

This will check:
- CUDA operations availability
- Python module imports
- Quantization registration
- Model detection functionality

## Key Features

### Mixed Bit-Width Support
- Supports 2.2, 2.5, 3, 4-bit quantization
- Automatic strategy loading from `quant_strategy.json`
- Layer-wise and channel-wise quantization modes

### Optimized Performance
- Custom CUDA kernels for quantized operations
- Efficient weight layout transformations
- Memory-optimized inference

### Easy Integration
- Automatic model detection
- Compatible with existing vLLM workflows
- Minimal code changes required

### MoE Model Support
- **Qwen3 MoE**: Adaptive processing strategies for up to 64 experts
- **DeepSeek V3**: Optimized for large-scale MoE (256+ experts) 
- **Automatic patch application**: Smart strategy selection based on model type
- **Memory optimization**: Efficient processing for large expert counts

## Troubleshooting

### Common Issues

1. **CUDA compilation errors**:
   ```bash
   pip cache purge
   pip install -e . --force-reinstall --no-cache-dir
   ```

2. **Module not found errors**:
   - Ensure all GBA files are in correct locations
   - Check if CUDA extensions compiled successfully

3. **Model not detected**:
   - Verify model name contains GBA indicators
   - Check for strategy files in model directory

### MoE Model Issues

1. **Patch application failed**:
   - Check transformers version compatibility
   - Verify model type detection
   - Review MoE patch logs

2. **Large expert count models (DeepSeek V3)**:
   - Use tensor parallelism: `tensor_parallel_size=2`
   - Monitor GPU memory usage
   - Consider conservative processing strategy

### Debug Tools

Use the validation script for debugging:
```python
from examples.quantization.gba_integration_validation import debug_gba_model_loading
debug_gba_model_loading("path/to/model")
```

## Examples

See `gba_integration_validation.py` for complete examples including:
- Basic model loading
- Manual configuration
- Performance benchmarking
- Error handling

## Performance Notes

- GBA quantization typically provides 2-4x memory reduction
- Inference speed depends on model size and quantization strategy
- Mixed bit-width models may have varying performance characteristics

## Support

For issues specific to GBA quantization integration:
1. Check the validation script output
2. Verify CUDA compilation completed successfully
3. Ensure model files contain proper quantization metadata

For general vLLM issues, refer to the main vLLM documentation.