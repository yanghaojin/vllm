#!/usr/bin/env python3
"""
Single GPU vs Multi-GPU Comparison Test (Chat Template Version)
Optimized for Qwen3 Chat models
"""

import torch
import time
from vllm import LLM
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer

MODEL_NAME = "GreenBitAI/Qwen-3-0.6B-layer-mix-bpw-4.0"

def format_chat_prompt(prompt, tokenizer):
    """Format prompt using chat template"""
    try:
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False
            )
            return formatted_prompt
        else:
            # If no chat template, fall back to Qwen3 default format
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    except Exception as e:
        print(f"Chat template formatting failed: {e}")
        # Fall back to Qwen3 default format
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def compare_single_vs_multi():
    """Compare response quality between single GPU and multi-GPU (Chat version)"""
    print("🔍 Single GPU vs Multi-GPU Comparison Test (Chat Template Version)")
    print("=" * 60)

    gpu_count = torch.cuda.device_count()
    print(f"Available GPU count: {gpu_count}")

    # More suitable test cases for chat models
    test_cases = [
        "Hi, what is your name?",
        "What is 2+2?",
        "What is the capital of France?",
        "Tell me about Python programming language.",
        "How are you today?",
    ]

    # Sampling parameters optimized for Chat models
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=50,
        stop=["<|im_end|>", "<|endoftext|>"]  # Stop tokens for Qwen3
    )

    results = {}

    # Test configurations
    configs = [
        {"name": "Single GPU", "tp_size": 1},
        {"name": "Multiple GPUs", "tp_size": min(4, gpu_count)} if gpu_count >= 2 else None
    ]

    # Remove None values
    configs = [c for c in configs if c is not None]

    # Initialize tokenizer (for chat template)
    print("📋 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        print(f"✅ Tokenizer loaded successfully")
        print(f"   Chat template available: {tokenizer.chat_template is not None}")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        tokenizer = None

    for config in configs:
        print(f"\n{'=' * 20} {config['name']} Test {'=' * 20}")

        try:
            print(f"📋 Initializing {config['name']} model (TP={config['tp_size']})...")
            start_time = time.time()

            llm = LLM(
                model=MODEL_NAME,
                quantization="gba",
                trust_remote_code=True,
                tensor_parallel_size=config['tp_size'],
                dtype="float16",
                max_model_len=2048,  # Increased length for chat template
                gpu_memory_utilization=0.8,
                enforce_eager=True,
            )

            init_time = time.time() - start_time
            print(f"Initialization time: {init_time:.2f} seconds")

            # Warm-up (using chat format)
            print("🔥 Warming up...")
            if tokenizer:
                warmup_prompt = format_chat_prompt("Hello", tokenizer)
            else:
                warmup_prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

            _ = llm.generate([warmup_prompt], sampling_params)

            # Test each case
            config_results = []

            for i, original_prompt in enumerate(test_cases):
                print(f"\n--- Test {i + 1}: '{original_prompt}' ---")

                # Format as chat format
                if tokenizer:
                    formatted_prompt = format_chat_prompt(original_prompt, tokenizer)
                else:
                    formatted_prompt = f"<|im_start|>user\n{original_prompt}<|im_end|>\n<|im_start|>assistant\n"

                print(f"   Formatted: '{formatted_prompt[:80]}...'")

                start_time = time.time()
                outputs = llm.generate([formatted_prompt], sampling_params)
                gen_time = time.time() - start_time

                response = outputs[0].outputs[0].text.strip()
                tokens = len(outputs[0].outputs[0].token_ids)

                print(f"   Output: '{response}'")
                print(f"   Time: {gen_time:.3f} seconds")
                print(f"   Token count: {tokens}")

                config_results.append({
                    'prompt': original_prompt,
                    'formatted_prompt': formatted_prompt,
                    'response': response,
                    'time': gen_time,
                    'tokens': tokens
                })

            # Batch testing
            print(f"\n🔄 Batch testing...")

            # Prepare batch formatted prompts
            if tokenizer:
                batch_formatted_prompts = [format_chat_prompt(prompt, tokenizer) for prompt in test_cases]
            else:
                batch_formatted_prompts = [f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                                           for prompt in test_cases]

            batch_start = time.time()
            batch_outputs = llm.generate(batch_formatted_prompts, sampling_params)
            batch_time = time.time() - batch_start

            print(f"Batch results:")
            for i, (original_prompt, output) in enumerate(zip(test_cases, batch_outputs)):
                response = output.outputs[0].text.strip()
                print(f"   {i + 1}. '{original_prompt}' -> '{response}'")

            results[config['name']] = {
                'individual': config_results,
                'batch_time': batch_time,
                'init_time': init_time
            }

            # Clean up
            del llm
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ {config['name']} test failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compare results
    print(f"\n{'=' * 60}")
    print("📊 Detailed Comparison Results")
    print(f"{'=' * 60}")

    if len(results) >= 2:
        compare_results_detail(results)
    else:
        print("❌ Cannot compare, at least two configurations are required")


def compare_results_detail(results):
    """Detailed comparison of results"""
    configs = list(results.keys())

    if len(configs) < 2:
        return

    single_results = results.get("Single GPU", {})
    multi_results = results.get("Multiple GPUs", {})

    if not single_results or not multi_results:
        print("❌ Missing single or multi GPU results")
        return

    print(f"\n⏱️  Performance Comparison:")
    print(
        f"   Initialization Time: Single GPU {single_results['init_time']:.2f}s vs Multi-GPU {multi_results['init_time']:.2f}s")
    print(
        f"   Batch Processing Time: Single GPU {single_results['batch_time']:.3f}s vs Multi-GPU {multi_results['batch_time']:.3f}s")

    print(f"\n📝 Response Quality Comparison:")

    single_individual = single_results.get('individual', [])
    multi_individual = multi_results.get('individual', [])

    for i, (single, multi) in enumerate(zip(single_individual, multi_individual)):
        prompt = single['prompt']
        single_resp = single['response']
        multi_resp = multi['response']

        print(f"\n   Question {i + 1}: '{prompt}'")
        print(f"     Single GPU: '{single_resp}'")
        print(f"     Multi-GPU: '{multi_resp}'")

        if single_resp == multi_resp:
            print(f"     ✅ Results match")
        else:
            print(f"     ❌ Results differ")

            # Quality evaluation for chat models
            single_quality = evaluate_chat_quality(prompt, single_resp)
            multi_quality = evaluate_chat_quality(prompt, multi_resp)

            print(f"     Quality Score: Single {single_quality}/10, Multi {multi_quality}/10")

            if single_quality > multi_quality:
                print(f"     🏆 Single GPU performs better")
            elif multi_quality > single_quality:
                print(f"     🏆 Multi-GPU performs better")
            else:
                print(f"     ⚖️  Quality is comparable")


def evaluate_chat_quality(prompt, response):
    """Evaluate quality for chat model responses"""
    if not response or len(response.strip()) == 0:
        return 0

    score = 5  # Base score

    # Reasonable length (Chat models usually respond longer)
    if 10 <= len(response) <= 100:
        score += 2
    elif len(response) < 5:
        score -= 2

    # Check for valid conversational reply
    if any(word in response.lower() for word in ["hello", "hi", "i am", "i'm", "my name", "sure", "yes", "no"]):
        score += 1

    # Specific question evaluation
    if "name" in prompt.lower():
        if any(word in response.lower() for word in ["i am", "i'm", "my name", "qwen", "assistant"]):
            score += 2
        elif "sorry" in response.lower():
            score -= 1

    elif "2+2" in prompt or "what is 2+2" in prompt.lower():
        if "4" in response or "four" in response:
            score += 3
        else:
            score -= 1

    elif "france" in prompt.lower():
        if "paris" in response.lower():
            score += 3

    elif "python" in prompt.lower():
        if any(word in response.lower() for word in ["programming", "language", "code", "software"]):
            score += 2

    elif "how are you" in prompt.lower():
        if any(word in response.lower() for word in ["fine", "good", "well", "great", "thank"]):
            score += 2
        elif "sorry" in response.lower():
            score -= 1

    # Penalize obviously wrong answers
    if any(bad in response.lower() for bad in ["i cannot", "i can't", "i don't know", "sorry, i"]):
        score -= 1

    return max(0, min(10, score))


def main():
    """Main function"""
    print("🎯 Single GPU vs Multi-GPU Quality Comparison Test (Chat Template Version)")
    print("Goal: Test Qwen3 model using proper chat formatting")
    print()

    compare_single_vs_multi()


if __name__ == "__main__":
    main()
