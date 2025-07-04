#!/bin/bash

# vLLM GBA enhanced startup script
# Use vLLM's built-in OpenAI API server with configurable parameters

# You can also start serving as follows:
#vllm serve GreenBitAI/Qwen-3-0.6B-layer-mix-bpw-4.0 \
#  --quantization gba \
#  --tensor-parallel-size 4 \
#  --dtype float16 \
#  --max-model-len 4096 \
#  --gpu-memory-utilization 0.8 \
#  --host 0.0.0.0 \
#  --port 8000  \
#  --trust-remote-code  \
#  --enforce-eager


# Default values
DEFAULT_MODEL="GreenBitAI/Qwen-3-0.6B-layer-mix-bpw-4.0"
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"
DEFAULT_TP_SIZE=""

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model MODEL         Model to use (default: $DEFAULT_MODEL)"
    echo "  --tp-size SIZE        Tensor parallel size (default: auto-detect based on GPU count)"
    echo "  --host HOST           Host address (default: $DEFAULT_HOST)"
    echo "  --port PORT           Port number (default: $DEFAULT_PORT)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use all defaults"
    echo "  $0 --model my-model --port 8001      # Custom model and port"
    echo "  $0 --tp-size 2 --host 127.0.0.1     # Custom TP size and host"
    echo ""
}

# Initialize variables with defaults
MODEL="$DEFAULT_MODEL"
HOST="$DEFAULT_HOST"
PORT="$DEFAULT_PORT"
TP_SIZE="$DEFAULT_TP_SIZE"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --tp-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validation
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
    echo "❌ Error: Port must be a number between 1 and 65535"
    exit 1
fi

if [[ -n "$TP_SIZE" ]] && (! [[ "$TP_SIZE" =~ ^[0-9]+$ ]] || [ "$TP_SIZE" -lt 1 ]); then
    echo "❌ Error: TP_SIZE must be a positive integer"
    exit 1
fi

echo "🚀 Start vLLM GBA API server"
echo "Model: $MODEL"
echo "Host: $HOST"
echo "Port: $PORT"
echo "================================"

# Check the number of GPUs
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "$GPU_COUNT GPUs detected"

# Set TP_SIZE if not provided
if [[ -z "$TP_SIZE" ]]; then
    if [ $GPU_COUNT -lt 4 ]; then
        echo "⚠️ It is recommended to use 4 GPUs, there are currently only $GPU_COUNT, which will be adjusted automatically"
        TP_SIZE=$GPU_COUNT
    else
        TP_SIZE=4
    fi
    echo "Auto-detected TP_SIZE: $TP_SIZE"
else
    echo "Using specified TP_SIZE: $TP_SIZE"
    if [ $TP_SIZE -gt $GPU_COUNT ]; then
        echo "⚠️ Warning: TP_SIZE ($TP_SIZE) is greater than available GPUs ($GPU_COUNT)"
    fi
fi

echo "Uses $TP_SIZE GPU in parallel."
echo ""

# Check if model contains the expected pattern for GBA quantization
if [[ "$MODEL" == *"GreenBitAI"* ]] || [[ "$MODEL" == *"gba"* ]] || [[ "$MODEL" == *"GBA"* ]]; then
    QUANTIZATION="gba"
    echo "Using GBA quantization for model: $MODEL"
else
    echo "⚠️ Warning: Model doesn't appear to be GBA quantized. Using GBA quantization anyway."
    QUANTIZATION="gba"
fi

# Start service
echo "🔥 Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --quantization "$QUANTIZATION" \
    --tensor-parallel-size $TP_SIZE \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8 \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --enforce-eager