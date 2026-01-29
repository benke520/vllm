# Speculative Decoding: The "Bonus Token" Mechanism

## Overview

The key insight behind speculative decoding's efficiency is that **even when a draft token is rejected, we still get a verified token** from that position because the target model has already computed logits for it.

## How It Works

```
Draft tokens:     [d0] [d1] [d2] [d3] [d4]
                   │    │    │    │    │
Target forward:   ─┴────┴────┴────┴────┴─→  (parallel forward pass on all 5)
                   │    │    │    │    │
Target logits:    L0   L1   L2   L3   L4   (logits computed for each position)
                   │    │    │    │    │
Verification:     ✓    ✓    ✓    ✗    ─    (compare against target's distribution)
                   │    │    │    │
                   │    │    │    └── d3 rejected, but L3 already computed!
                   │    │    │              → Sample from L3 → get "correct" token
                   │    │    │
Result:          [d0] [d1] [d2] [T3]       = 4 verified tokens
                  └─accepted──┘  └─bonus─┘
                  
                  [d4] discarded (it was conditioned on wrong d3)
```

## Step-by-Step Process

1. **Draft Generation**: Draft model cheaply generates k=5 candidate tokens
2. **Parallel Verification**: Target model runs **one** forward pass on all 5 positions in parallel
3. **Logits Computation**: At each position i, target computes P_target(token | context)
4. **Comparison**: Check if d_i matches what target would have sampled
5. **Rejection Detected**: First rejection at i=3 → d3 ≠ target's choice
6. **Bonus Token**: L3 (logits at position 3) already exists → sample the correct token from it
7. **Cleanup**: Tokens after rejection (d4) are invalid and discarded

## The Efficiency Guarantee

You always get **at least 1 token** per target model forward pass:

| Scenario          | Draft Accuracy       | Tokens Gained   | Formula                  |
|-------------------|----------------------|-----------------|--------------------------|
| **Best case**     | 100% (all accepted)  | k+1 tokens      | 5 accepted + 1 bonus = 6 |
| **Typical case**  | Partial acceptance   | 1 to k+1 tokens | accepted + 1 bonus       |
| **Worst case**    | 0% (first rejected)  | 1 token         | 0 accepted + 1 bonus = 1 |

## Why This Matters

This "bonus token" mechanism is why speculative decoding achieves speedups even with imperfect draft models:

1. **No wasted computation**: The target model's forward pass always produces usable output
2. **Amortized cost**: One expensive target forward pass can yield multiple tokens
3. **Graceful degradation**: Even 0% draft accuracy still produces 1 token per step (same as standard decoding)

## Example Calculation

Given:
- Draft model proposes 5 tokens: `[d0, d1, d2, d3, d4]`
- Target model verifies in parallel
- Rejection at position 3

Result:
- 3 accepted tokens: `d0, d1, d2`
- 1 bonus token: sampled from `L3` (the logits at the rejection point)
- **Total: 4 verified tokens** from a single target model forward pass

Without speculative decoding, this would have required 4 separate forward passes.
