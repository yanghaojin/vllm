# GBA Quantization for vLLM

This directory contains the implementation of GreenBitAI's quantization support for vLLM, enabling efficient inference of mixed bit-width quantized models.

### Supported and planned [GBA's Low-bit LLMs](https://huggingface.co/GreenBitAI):
- Qwen-3 DENSE series: ✅ 
- Qwen-3 MoE: 🔧
- Deepseek MoE: 🔧

## Installation

### 1. Build From Source on Linux (with CUDA)

To use these instructions, you need to have [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) and a suitable C++ compiler installed.
1. Create Environment for Python 3.11 and activate it:
```bash
conda create -y --name gba-vllm python=3.11
conda activate gba-vllm
```

2. Get gba-vllm source code and ensure GBA CUDA extensions are compiled:
```bash
git clone --branch gba_quantization --single-branch https://github.com/GreenBitAI/vllm.git
cd vllm
pip install -e . --verbose
```

3. Install FlashInfer
```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
python -m pip install -v .
```

### 2. Binary Release

**A first experimental binary release for Linux with CUDA 12.? is ready.**
It only supports GPUs with CUDA compute capability with 8.6 or higher ([check here](https://developer.nvidia.com/cuda-gpus)).
We recommend to create a conda environment to manage the installed CUDA version and other packages:

1. Create Environment for Python 3.11 and activate it:
```bash
conda create -y --name gba-vllm python=3.11
conda activate gba-vllm
```
2. Install CUDA (if it is not installed already on the system):
```bash
conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit
```
3. Install our pre-compiled whl with pip (this URL is for CUDA 12.1  and Python 3.11 - 
you can find other versions [here](https://packages.greenbit.ai/whl/)):
```bash
pip install \
  "https://packages.greenbit.ai/whl/cu121/vllm/vllm-??????-linux_x86_64.whl"
```

## Examples

### 1. Basic Usage

It will download and execute the GBA's low-bit LLM from their Hugging Face repo:
```python
from vllm import LLM
from vllm.sampling_params import SamplingParams

# Load GBA quantized model
llm = LLM(
    model="GreenBitAI/Qwen-3-0.6B-layer-mix-bpw-4.0",
    quantization="gba",
    trust_remote_code=True,
    enforce_eager=True,
    dtype="float16",
    max_model_len=100, 
    gpu_memory_utilization=0.8,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

# Generate text
outputs = llm.generate(["Python is a programming language that"], sampling_params)
print(outputs[0].outputs[0].text)
```

### 2. Use vLLM's built-in OpenAI API server

You can start the API server by using this script:
```python
# Grants execution permission
chmod +x examples/gba/start_vllm_server.sh

# Starts the API server
examples/gba/./start_vllm_server.sh
```
or you can also simply start the API server by running this COMMAND:
```shell
vllm serve GreenBitAI/Qwen-3-0.6B-layer-mix-bpw-4.0 \
  --quantization gba \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --host 0.0.0.0 \
  --port 8000  \
  --trust-remote-code  \
  --enforce-eager
```
Then, you can run the following test script.
```shell
python examples/gba/test_vllm_api.py \ 
  --url http://localhost:8000/v1 \
  --model GreenBitAI/Qwen-3-0.6B-layer-mix-bpw-4.0 \ 
  --max-tokens 512
```
You can also use the ***--enable-thinking*** option to enable the thinking mode of Qwen-3 LLMs.

### 3. Single- and multi-GPU consistency test

Run the following script to perform single- and multi-GPU result consistency test.
```shell
python examples/gba/single_vs_multi_test.py 
```

## File Structure

```
vllm/
├── README.md                                                                          (modified)
│
├── model_executor/
│   ├── layers/
│   │   └── quantization/
│   │       ├── gba_moe_support.py        # GBA moe related support methods            (added)
│   │       ├── gba.py                    # Core GBA quantization implementation       (added)
│   │       └── model_integration.py      # Model integration and layer replacement    (added)
│   │
│   └── model_loader/
│       └──weight_utils.py                # Enhanced weight loading                    (modified)
│
├── transformers_utils/
│   └── config.py                         # Configuration detection                    (modified)
│
├── csrc/
│   ├── quantization/
│   │   └── gba/                                                                       (added)
│   │       ├── gba_ops.cu               # CUDA kernel implementations                 
│   │       └── exl2/...                 # Exl2 cuda kernel implementations            
│   ├── ops.h                            # GBA operators                               (modified)
│   └── torch_bindings.cpp               # PyTorch bindings                            (modified)
│
└── examples/
    └── gba/                                                                           (added)
        ├── single_vs_multi_test.py       # single and multi GPU execution tests
        ├── start_vllm_server.sh          # Start a testing server using gba models
        └── test_vllm_api.py              # Testing client in conjunction with start_vllm_server.sh
```