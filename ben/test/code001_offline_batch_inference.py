# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Step 1. import offline inference class
from vllm import LLM, SamplingParams

# Step 2. Create a (offline) input batch
prompts = [
    "Hello, my name is",
    "The president of the Unite States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Step 3. Create a offline inference model instance
llm = LLM(model="facebook/opt-125m")

# Step 4. Generate the output batch
# The return value is a list of RequestOutput objects
# that contain the input prompt, generated output text and other info
outputs = llm.generate(prompts, sampling_params)

print("\nGenerated Outputs:\n" + "-" * 60)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt:     {prompt!r}")
    print(f"Output:     {generated_text!r}")
    print("-" * 60)
