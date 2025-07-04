#!/usr/bin/env python3
"""
Simplified vLLM GBA Test Script
Test vLLM built-in OpenAI API server with proper enable_thinking support
"""

import argparse
import time
from openai import OpenAI


def evaluate_quality(prompt: str, response: str) -> int:
    """Simple quality evaluation"""
    if not response or len(response.strip()) == 0:
        return 0

    score = 5  # Base score

    # Reasonable length
    if 10 <= len(response) <= 300:
        score += 2
    elif len(response) < 5:
        score -= 2

    # Specific question evaluation
    if "name" in prompt.lower():
        if any(word in response.lower() for word in ["i am", "i'm", "my name", "qwen", "assistant"]):
            score += 2
    elif "2+2" in prompt:
        if "4" in response or "four" in response:
            score += 3
    elif "france" in prompt.lower():
        if "paris" in response.lower():
            score += 3
    elif "python" in prompt.lower():
        if any(word in response.lower() for word in ["programming", "language", "code"]):
            score += 2

    return max(0, min(10, score))


def main():
    parser = argparse.ArgumentParser(description="Simplified vLLM GBA API testing script")
    parser.add_argument("--url", default="http://localhost:8000/v1", help="API server URL")
    parser.add_argument("--model", default="GreenBitAI/Qwen-3-0.6B-layer-mix-bpw-4.0", help="Model name")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum token count")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode")

    args = parser.parse_args()


    thinking_enabled = False
    if args.enable_thinking:
        thinking_enabled = True

    print("🎯 Simplified vLLM GBA API Test")
    print("=" * 50)
    print(f"API URL: {args.url}")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")

    print(f"Thinking mode: {'Enabled' if thinking_enabled else 'Disabled'}")
    print()

    # Initialize OpenAI client
    client = OpenAI(
        api_key="EMPTY",
        base_url=args.url,
    )

    # Test cases
    test_cases = [
        "Hi, what is your name?",
        "What is 2+2?",
        "What is the capital of France?",
        "Tell me about Python programming language.",
        "How are you today?"
    ]

    print("🔍 Running tests...")
    results = []

    for i, prompt in enumerate(test_cases):
        print(f"\n--- Test {i + 1}/{len(test_cases)} ---")
        print(f"Prompt: '{prompt}'")

        try:
            # Prepare request parameters
            request_params = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
            }

            # Add thinking control if specified
            request_params["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": thinking_enabled}
            }

            start_time = time.time()
            response = client.chat.completions.create(**request_params)
            response_time = time.time() - start_time

            content = response.choices[0].message.content
            tokens = response.usage.total_tokens

            print(f"Response: '{content}'")
            print(f"Duration: {response_time:.3f}s")
            print(f"Tokens: {tokens}")

            # Quality evaluation
            quality = evaluate_quality(prompt, content)
            print(f"Quality: {quality}/10")

            if quality >= 7:
                print("✅ Good quality")
            elif quality >= 4:
                print("⚠️ Average quality")
            else:
                print("❌ Poor quality")

            results.append({
                "success": True,
                "prompt": prompt,
                "response": content,
                "response_time": response_time,
                "tokens": tokens,
                "quality": quality
            })

        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                "success": False,
                "prompt": prompt,
                "error": str(e)
            })

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)

    successful_tests = [r for r in results if r["success"]]
    failed_tests = [r for r in results if not r["success"]]

    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")

    if successful_tests:
        avg_time = sum(r["response_time"] for r in successful_tests) / len(successful_tests)
        avg_tokens = sum(r["tokens"] for r in successful_tests) / len(successful_tests)
        avg_quality = sum(r["quality"] for r in successful_tests) / len(successful_tests)

        print(f"\nAverage response time: {avg_time:.3f}s")
        print(f"Average tokens: {avg_tokens:.1f}")
        print(f"Average quality: {avg_quality:.1f}/10")

        if avg_quality >= 7:
            print("\n🎉 Overall excellent quality!")
        elif avg_quality >= 4:
            print("\n👍 Overall good quality")
        else:
            print("\n⚠️ Quality needs improvement")

    if failed_tests:
        print(f"\n❌ Failed tests:")
        for test in failed_tests:
            print(f"  - '{test['prompt']}': {test['error']}")

    print("\n🎯 Testing complete!")


if __name__ == "__main__":
    main()