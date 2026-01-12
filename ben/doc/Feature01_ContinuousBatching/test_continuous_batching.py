# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Phase 1: Continuous Batching Debug Test

This script tests vLLM's continuous batching behavior with requests of different lengths.

Key Observations:
1. How batch size changes dynamically
2. Sequence lifecycle (WAITING → RUNNING → FINISHED)
3. Prefill vs Decode interleaving
4. Scheduler decision points

Debug Strategy:
- Set breakpoints at marked locations (🎯)
- Step through the execution
- Observe batch composition changes
"""

import time

from vllm import LLM, SamplingParams


def test_continuous_batching():
    """
    Core test: 3 requests with different lengths
    - Short request: completes quickly
    - Long request: takes more time
    - Medium request: in between

    Observation points:
    1. How they're batched together
    2. How batch shrinks when short request finishes
    3. How batch expands when new requests arrive
    """

    print("=" * 80)
    print("Initializing vLLM...")
    print("=" * 80)

    # Initialize LLM with small model for quick testing
    llm = LLM(
        model="facebook/opt-125m",  # Small model, fast
        max_model_len=512,
        # Key parameters for observing batching behavior
        max_num_batched_tokens=2048,  # Max tokens per step
        max_num_seqs=256,  # Max sequences per step
    )

    # Prepare prompts of different lengths
    prompts = [
        # Request 1: Short prompt, expected to finish quickly
        "Short prompt",
        # Request 2: Long prompt, will take more time
        "This is a much longer prompt that will take more time to process. " * 10,
        # Request 3: Medium length
        "This is a medium length prompt for testing purposes.",
    ]

    # Sampling parameters: different output lengths
    sampling_params_list = [
        SamplingParams(temperature=0.0, max_tokens=10),  # Short output
        SamplingParams(temperature=0.0, max_tokens=100),  # Long output
        SamplingParams(temperature=0.0, max_tokens=50),  # Medium output
    ]

    print("\n" + "=" * 80)
    print("Test Scenario Setup:")
    print("=" * 80)
    for i, (prompt, params) in enumerate(zip(prompts, sampling_params_list)):
        print(f"Request {i + 1}:")
        print(f"  Prompt length: {len(prompt)} chars")
        print(f"  Max tokens: {params.max_tokens}")
        print()

    # Start inference
    print("=" * 80)
    print("Starting inference - observing Continuous Batching behavior...")
    print("=" * 80)
    print("\n🔍 Set breakpoint here to start debugging!\n")

    start_time = time.time()

    # 🎯 KEY: This is the entry point, start debugging from here
    # Suggested breakpoints:
    # 1. vllm/entrypoints/llm.py - LLM.generate()
    # 2. vllm/engine/llm_engine.py - LLMEngine.step()
    # 3. vllm/core/scheduler.py - Scheduler.schedule()
    outputs = llm.generate(prompts, sampling_params_list)

    end_time = time.time()

    # Print results
    print("\n" + "=" * 80)
    print("Inference Complete! Results:")
    print("=" * 80)
    for i, output in enumerate(outputs):
        print(f"\nRequest {i + 1}:")
        print(f"  Generated tokens: {len(output.outputs[0].token_ids)}")
        print(f"  Generated text: {output.outputs[0].text[:100]}...")

    print(f"\nTotal time: {end_time - start_time:.2f}s")


def test_batch_arrival():
    """
    Test 2: Requests arriving in batches
    Simulates real serving scenario: requests don't all arrive at once
    """
    print("\n" + "=" * 80)
    print("Test Scenario 2: Requests Arriving in Batches")
    print("=" * 80)

    llm = LLM(
        model="facebook/opt-125m",
        max_model_len=256,
    )

    # First batch of requests
    batch1 = ["First batch request 1", "First batch request 2"]
    params1 = SamplingParams(temperature=0.0, max_tokens=20)

    print("\nSending first batch...")
    outputs1 = llm.generate(batch1, params1)

    # Second batch of requests (arrives during first batch processing)
    batch2 = ["Second batch request 1"]
    params2 = SamplingParams(temperature=0.0, max_tokens=30)

    print("\nSending second batch...")
    outputs2 = llm.generate(batch2, params2)

    print("\nAll requests completed!")


def test_with_logging():
    """
    Test 3: With detailed logging enabled

    To see scheduler decisions, set environment variable:
    export VLLM_LOGGING_LEVEL=DEBUG
    """
    import os

    print("\n" + "=" * 80)
    print("Test Scenario 3: With Debug Logging")
    print("=" * 80)

    # Enable debug logging
    os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

    llm = LLM(
        model="facebook/opt-125m",
        max_model_len=256,
    )

    prompts = [
        "Request 1: short",
        "Request 2: medium length prompt here",
        "Request 3: this is a much longer prompt to observe different behavior",
    ]

    params = SamplingParams(temperature=0.0, max_tokens=30)

    print("\nStarting inference with debug logging...")
    outputs = llm.generate(prompts, params)

    print("\nCompleted!")


def test_varying_arrival_times():
    """
    Test 4: Simulate varying request arrival times

    This would require async API, but we can simulate by running
    generate() calls with different prompts sequentially
    """
    print("\n" + "=" * 80)
    print("Test Scenario 4: Sequential Requests")
    print("=" * 80)

    llm = LLM(
        model="facebook/opt-125m",
        max_model_len=256,
    )

    # Request 1: arrives first
    print("\n[T=0s] Request 1 arrives")
    output1 = llm.generate(
        ["First request to arrive"], SamplingParams(temperature=0.0, max_tokens=20)
    )
    print(f"Request 1 completed: {output1[0].outputs[0].text[:50]}...")

    # Request 2: arrives after request 1 started
    print("\n[T=1s] Request 2 arrives")
    output2 = llm.generate(
        ["Second request, arriving later"],
        SamplingParams(temperature=0.0, max_tokens=30),
    )
    print(f"Request 2 completed: {output2[0].outputs[0].text[:50]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test vLLM Continuous Batching")
    parser.add_argument(
        "--test",
        type=str,
        default="basic",
        choices=["basic", "batch", "logging", "sequential", "all"],
        help="Which test to run",
    )

    args = parser.parse_args()

    if args.test == "basic" or args.test == "all":
        test_continuous_batching()

    if args.test == "batch" or args.test == "all":
        test_batch_arrival()

    if args.test == "logging" or args.test == "all":
        test_with_logging()

    if args.test == "sequential" or args.test == "all":
        test_varying_arrival_times()
