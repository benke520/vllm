# LLM Parallelism Strategies: DP, TP, and PP

This document explains the three main parallelism strategies used in vLLM for distributed inference across multiple GPUs.

## Table of Contents
- [Overview](#overview)
- [Single GPU Baseline](#single-gpu-baseline)
- [Tensor Parallelism (TP)](#tensor-parallelism-tp)
- [Pipeline Parallelism (PP)](#pipeline-parallelism-pp)
- [Data Parallelism (DP)](#data-parallelism-dp)
- [3D Parallelism (Combined)](#3d-parallelism-combined)
- [Summary](#summary)
- [Usage in vLLM](#usage-in-vllm)

---

## Overview

| Strategy | What it Splits | Communication | Best For |
|----------|----------------|---------------|----------|
| **TP** (Tensor) | Weights within each layer (horizontal) | All-Reduce every layer (frequent) | Large layers, fast NVLink within single node |
| **PP** (Pipeline) | Layers across stages (vertical) | Point-to-point between stages (less frequent) | Very deep models, slower interconnect across nodes |
| **DP** (Data) | Requests across model replicas (replication) | None during inference! | High throughput, model already fits in memory |

**Key Formulas:**
- Memory per GPU = Model Size / (TP × PP)
- Total GPUs = TP × PP × DP
- Throughput ≈ Single_GPU × DP (ideally)

---

## Single GPU Baseline

No parallelism - entire model on one GPU.

```
┌─────────────────────────────────────────────────────────────────┐
│                      SINGLE GPU VIEW                            │
│                  (No Parallelism: Baseline)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input Tokens                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    Embedding Layer                      │   │
│   └─────────────────────────────────────────────────────────┘   │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Transformer Block 0                                    │   │
│   │  ┌───────────────────────┐  ┌─────────────────────────┐ │   │
│   │  │  Multi-Head Attention │  │  Feed-Forward Network   │ │   │ ## FFN is a logic path: linear (up projection 4096->14336) -> activation -> linear (down projection 14336 -> 4096)
│   │  │  ┌───┬───┬───┬───┐    │  │  ┌─────────────────┐    │ │   │ ## up projection means dimension expanding and down projection means dimension compressing
│   │  │  │ H │ H │ H │ H │    │→ │  │  Up Projection  │    │ │   │
│   │  │  │ 0 │ 1 │ 2 │ 3 │    │  │  ├─────────────────┤    │ │   │
│   │  │  └───┴───┴───┴───┘    │  │  │ Down Projection │    │ │   │
│   │  └───────────────────────┘  └─────────────────────────┘ │   │
│   └─────────────────────────────────────────────────────────┘   │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Transformer Block 1  (same structure)                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│        │                                                        │
│       ...                                                       │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Transformer Block N-1                                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    LM Head (Output)                     │   │
│   └─────────────────────────────────────────────────────────┘   │
│        │                                                        │
│        ▼                                                        │
│   Output Logits                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tensor Parallelism (TP)

**Splits WITHIN each layer horizontally, i.e., weights splitting**

- Each GPU has ALL layers, but only PART of each layer's `weights`
- Communication: All-Reduce after operation layer, i.e. after embedding and after each attention/FFN sublayer
    - Each layer needs a complete input data but splits its weights to calcalate independently
- Requires fast interconnect like NVLink

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                          TENSOR PARALLELISM (TP=2)                            │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   Input Tokens (broadcast to all GPUs)                                        │
│        │                                                                      │
│        ├──────────────────────────────────┬──────────────────────────────────┐│
│        ▼                                  ▼                                  ││
│   ┌─────────────────────────────┐   ┌─────────────────────────────┐          ││
│   │          GPU 0              │   │          GPU 1              │          ││# Embedding linear layer weights splitting
│   ├─────────────────────────────┤   ├─────────────────────────────┤          ││
│   │ Embedding (vocab 0:16K)     │   │ Embedding (vocab 16K:32K)   │          ││
│   └──────────────┬──────────────┘   └──────────────┬──────────────┘          ││
│                  │                                 │                         ││
│                  └────────────┬────────────────────┘                         ││
│                        ╔══════▼══════╗                                       ││
│                        ║ ALL-REDUCE  ║                                       ││
│                        ╚══════╬══════╝                                       ││
│                  ┌────────────┴────────────┐                                 ││
│                  ▼                         ▼                                 ││
│   ┌─────────────────────────────┐   ┌─────────────────────────────┐          ││
│   │ X [full 4096] (same!)       │   │ X [full 4096] (same!)       │          ││
│   ├─────────────────────────────┤   ├─────────────────────────────┤          ││
│   │ Transformer Block 0         │   │ Transformer Block 0         │          ││
│   │ ┌─────────────────────────┐ │   │ ┌─────────────────────────┐ │          ││# Attention layer head-wise weights splitting
│   │ │ Attention (heads 0-15)  │ │   │ │ Attention (heads 16-31) │ │          ││
│   │ │ ┌───┬───┬───┬───┐       │ │   │ │ ┌───┬───┬───┬───┐       │ │          ││
│   │ │ │H0 │...│H14│H15│       │ │   │ │ │H16│...│H30│H31│       │ │          ││
│   │ │ └───┴───┴───┴───┘       │ │   │ │ └───┴───┴───┴───┘       │ │          ││
│   │ │ partial_out @ W_O_top   │ │   │ │ partial_out @ W_O_bot   │ │          ││# Attention linear layer weights splitting
│   │ └────────────┬────────────┘ │   │ └────────────┬────────────┘ │          ││
│   │              │ ALL-REDUCE ══╪═══╪══════════════╡              │          ││
│   │ ┌────────────▼────────────┐ │   │ ┌────────────▼────────────┐ │          ││
│   │ │ FFN (half intermediate) │ │   │ │ FFN (half intermediate) │ │          ││# FFN has three operations: up(dim expansion) -> activation -> down(dim compression)
│   │ │ W_up[:,:half]           │ │   │ │ W_up[:,half:]           │ │          ││# Two GPUs split W_up by column into two independent halves
│   │ │ W_down[:half,:]         │ │   │ │ W_down[half:,:]         │ │          ││# Each half further goes through the down (dim compression) operation all projecting to the full result 4096 dimension
│   │ └────────────┬────────────┘ │   │ └────────────┬────────────┘ │          ││
│   │              │ ALL-REDUCE ══╪═══╪══════════════╡              │          ││# All-REDUCE then do an element-wise addition for the 4096 dimensions to get the final complete n x q x 4096 result
│   └──────────────┼──────────────┘   └──────────────┼──────────────┘          ││
│                  ▼                                 ▼                         ││
│             Block 1...N-1 (same pattern)                                     ││
│                  │                                 │                         ││
│   ┌──────────────▼──────────────┐   ┌──────────────▼──────────────┐          ││# LM Head calculates the logits of each vocabulary token using the full hidden dimensions
│   │ LM Head (vocab 0:16K)       │   │ LM Head (vocab 16K:32K)     │          ││# i.e., a linear translation from n_hidden_dim to vocab_size
│   └──────────────┬──────────────┘   └──────────────┬──────────────┘          ││# This is the final FC in the decoder part of the transformer architecture
│                  │                                 │                         ││# Check out D2L book page 447, and the code P452 last line
│                  └────────────┬────────────────────┘                         ││
│                        ╔══════▼══════╗                                       ││# All-gather means collecting all the results of all GPUs
│                        ║ ALL-GATHER  ║                                       ││# AllGather(GPU0:[1, 2, 3], GPU1:[4, 5, 6]) --> GPU0: [1, 2, 3, 4, 5, 6] and GPU1: [1, 2, 3, 4, 5, 6]
│                        ╚══════╬══════╝                                       ││# AllReduce(GPU0:[1, 2, 3], GPU1:[4, 5, 6]) --> GPU0: [5, 7, 9] and GPU1: [5, 7, 9]
│                               │                                              ││
│                               ▼                                              ││
│                     Output Logits [32K vocab]                                ││
│                                                                               │
├───────────────────────────────────────────────────────────────────────────────┤
│ KEY: • Weights split (embedding, Q/K/V/O, FFN, LM head)                       │
│      • Activations stay FULL on both GPUs (same X, same block outputs)        │
│      • All-Reduce: 1 after embedding + 2 per block (attention, FFN)           │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Pipeline Parallelism (PP)

**Splits model VERTICALLY by layers.**

- Each GPU has SOME layers, but COMPLETE layers
- Communication: Point-to-point activations between stages

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE PARALLELISM (PP=2)                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Input Tokens                                                                  │
│        │                                                                        │
│        ▼                                                                        │
│   ┌─────────────────────────────────────────────────────┐                       │
│   │                   GPU 0 (Stage 0)                   │                       │
│   │              First half of layers                   │                       │
│   ├─────────────────────────────────────────────────────┤                       │
│   │                                                     │                       │
│   │   ┌───────────────────────────────────────────┐     │                       │
│   │   │            Embedding Layer                │     │                       │
│   │   └───────────────────────────────────────────┘     │                       │
│   │                       │                             │                       │
│   │   ┌───────────────────▼───────────────────────┐     │                       │
│   │   │         Transformer Block 0               │     │                       │
│   │   └───────────────────────────────────────────┘     │                       │
│   │                       │                             │                       │
│   │   ┌───────────────────▼───────────────────────┐     │                       │
│   │   │         Transformer Block 1               │     │                       │
│   │   └───────────────────────────────────────────┘     │                       │
│   │                       │                             │                       │
│   │                      ...                            │                       │
│   │                       │                             │                       │
│   │   ┌───────────────────▼───────────────────────┐     │                       │
│   │   │       Transformer Block (N/2 - 1)         │     │                       │
│   │   └───────────────────────────────────────────┘     │                       │
│   │                       │                             │                       │
│   └───────────────────────┼─────────────────────────────┘                       │
│                           │                                                     │
│                           │  Send activations (hidden states)                   │
│                           │  ════════════════════════════════                   │
│                           ▼                                                     │
│   ┌─────────────────────────────────────────────────────┐                       │
│   │                   GPU 1 (Stage 1)                   │                       │
│   │              Second half of layers                  │                       │
│   ├─────────────────────────────────────────────────────┤                       │
│   │                       │                             │                       │
│   │   ┌───────────────────▼───────────────────────┐     │                       │
│   │   │         Transformer Block N/2             │     │                       │
│   │   └───────────────────────────────────────────┘     │                       │
│   │                       │                             │                       │
│   │   ┌───────────────────▼───────────────────────┐     │                       │
│   │   │       Transformer Block (N/2 + 1)         │     │                       │
│   │   └───────────────────────────────────────────┘     │                       │
│   │                       │                             │                       │
│   │                      ...                            │                       │
│   │                       │                             │                       │
│   │   ┌───────────────────▼───────────────────────┐     │                       │
│   │   │       Transformer Block (N - 1)           │     │                       │
│   │   └───────────────────────────────────────────┘     │                       │
│   │                       │                             │                       │
│   │   ┌───────────────────▼───────────────────────┐     │                       │
│   │   │              LM Head                      │     │                       │
│   │   └───────────────────────────────────────────┘     │                       │
│   │                       │                             │                       │
│   └───────────────────────┼─────────────────────────────┘                       │
│                           │                                                     │
│                           ▼                                                     │
│                    Output Logits                                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Scheduling (Eliminating Bubbles)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE SCHEDULING STRATEGY                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   NAIVE APPROACH (step() without batch_queue, and pipeline bubble happens):     │
│   ════════════════════════════════════════════                                  │
│                                                                                 │
│   def step():                                                                   │
│       output = scheduler.schedule()      # Schedule batch 1                     │
│       result = executor.execute(output)  # BLOCKING - wait for full completion  │# Wait for completion of both GPU0 and GPU1 on batch 1 before scheduling batch 2
│       scheduler.update(result)           # Process result                       │
│       return result                                                             │
│                                                                                 │
│   Problem: execute() BLOCKS until Batch 1 finishes through ALL stages           │
│   GPU 0 can't start Batch 2 until Batch 1 fully completes!                      │
│                                                                                 │
│   Time ──────────────────────────────────────────────────────►                  │
│                                                                                 │
│   GPU 0: │▓▓▓Batch1▓▓▓│   wait    │▓▓▓Batch2▓▓▓│   wait    │                    │# A complete cycle needs a relaying style cooperation between GPU0 and GPU1
│   GPU 1: │   wait     │▓▓▓Batch1▓▓▓│   wait    │▓▓▓Batch2▓▓▓│                   │# When GPU0 is calculating the first half of layers of the module, GPU1 has to be idle to wait wait the result
│                       ↑                                                         │# In this case, there is simply no need to have two GPUs. They are splitting a single thread of work across
│              Scheduler blocked here waiting for result                          │
│                                                                                 │
│                                                                                 │
│   SMART APPROACH (step_with_batch_queue):                                       │
│   ═══════════════════════════════════════                                       │
│                                                                                 │
│   def step_with_batch_queue():                                                  │
│       # Schedule NEW batch immediately (non-blocking)                           │
│       output = scheduler.schedule()                                             │
│       future = executor.execute(output, non_block=True)  # NON-BLOCKING!        │
│       batch_queue.append(future)                                                │
│                                                                                 │
│       # Only wait if queue is full                                              │
│       if queue_full:                                                            │
│           result = batch_queue.pop().result()  # Wait for oldest                │
│           scheduler.update(result)                                              │
│                                                                                 │
│   GPU 0 immediately schedules Batch 2 while Batch 1 is in GPU 1!                │
│                                                                                 │
│   Time ──────────────────────────────────────────────────────────────────────►  │
│                                                                                 │
│   GPU 0: │▓▓▓Batch1▓▓▓│▓▓▓Batch2▓▓▓│▓▓▓Batch3▓▓▓│▓▓▓Batch4▓▓▓│                  │#There's still synchronzation between GPUs for the same batch
│   GPU 1:              │▓▓▓Batch1▓▓▓│▓▓▓Batch2▓▓▓│▓▓▓Batch3▓▓▓│▓▓▓Batch4▓▓▓│     │#But across batches, the processes are streamlined
│                       ↑                                                         │
│          GPU1 starts Batch1 AFTER GPU0 finishes Batch1                          │
│          But GPU0 starts Batch2 immediately (overlap across batches!)           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Parallelism (DP)

**Replicates ENTIRE model on multiple GPU sets.**

- Complete model copy on each replica
- Processes DIFFERENT requests on each replica
- No communication needed between replicas during inference!

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DATA PARALLELISM (DP=2)                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                    Incoming Requests                                            │
│                            │                                                    │
│                            │  Load Balancer / Router                            │
│                            │                                                    │
│            ┌───────────────┴──────────────┐                                     │
│            │                              │                                     │
│            ▼                              ▼                                     │
│   ┌─────────────────────┐    ┌─────────────────────┐                            │
│   │   DP Replica 0      │    │   DP Replica 1      │                            │
│   │      (GPU 0)        │    │      (GPU 1)        │                            │
│   ├─────────────────────┤    ├─────────────────────┤                            │
│   │                     │    │                     │                            │
│   │  ┌───────────────┐  │    │  ┌───────────────┐  │                            │
│   │  │   Embedding   │  │    │  │   Embedding   │  │                            │
│   │  └───────────────┘  │    │  └───────────────┘  │                            │
│   │         │           │    │         │           │                            │
│   │  ┌──────▼────────┐  │    │  ┌──────▼────────┐  │                            │
│   │  │   Block 0     │  │    │  │   Block 0     │  │                            │
│   │  └───────────────┘  │    │  └───────────────┘  │                            │
│   │         │           │    │         │           │                            │
│   │  ┌──────▼────────┐  │    │  ┌──────▼────────┐  │                            │
│   │  │   Block 1     │  │    │  │   Block 1     │  │                            │
│   │  └───────────────┘  │    │  └───────────────┘  │                            │
│   │         :           │    │         :           │                            │
│   │  ┌──────▼────────┐  │    │  ┌──────▼────────┐  │                            │
│   │  │  Block N-1    │  │    │  │  Block N-1    │  │                            │
│   │  └───────────────┘  │    │  └───────────────┘  │                            │
│   │         │           │    │         │           │                            │
│   │  ┌──────▼────────┐  │    │  ┌──────▼────────┐  │                            │
│   │  │   LM Head     │  │    │  │   LM Head     │  │                            │
│   │  └───────────────┘  │    │  └───────────────┘  │                            │
│   │         │           │    │         │           │                            │
│   └─────────┼───────────┘    └─────────┼───────────┘                            │
│             │                          │                                        │
│             ▼                          ▼                                        │
│        Requests A,C,E             Requests B,D,F                                │
│                                                                                 │
│   Key: Complete model copy on each replica, processes DIFFERENT requests        │
│        No communication needed between replicas during inference                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3D Parallelism (Combined)

**Combining TP=2 × PP=2 × DP=2 = 8 GPUs**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    3D PARALLELISM: TP=2 × PP=2 × DP=2 = 8 GPUs                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                           Load Balancer                                         │# Data Parallelism across requests
│                                │                                                │
│              ┌─────────────────┴─────────────────┐                              │
│              ▼                                   ▼                              │
│   ┌─────────────────────────────┐   ┌─────────────────────────────┐             │
│   │      DP REPLICA 0           │   │      DP REPLICA 1           │             │
│   │      (Requests A)           │   │      (Requests B)           │             │
│   ├─────────────────────────────┤   ├─────────────────────────────┤             │
│   │                             │   │                             │             │
│   │  PP Stage 0 (Layers 0-15)   │   │  PP Stage 0 (Layers 0-15)   │             │# Pipeline Parallelism across layers when processing a batch of requests
│   │  ┌───────────┬───────────┐  │   │  ┌───────────┬───────────┐  │             │
│   │  │   GPU 0   │   GPU 1   │  │   │  │   GPU 4   │   GPU 5   │  │             │# Tensor Parallelism within a layer when processing a batch of requests
│   │  │   (TP=0)  │   (TP=1)  │  │   │  │   (TP=0)  │   (TP=1)  │  │             │
│   │  │           │           │  │   │  │           │           │  │             │
│   │  │  Layers   │  Layers   │  │   │  │  Layers   │  Layers   │  │             │
│   │  │  0-15     │  0-15     │  │   │  │  0-15     │  0-15     │  │             │
│   │  │  (left    │  (right   │  │   │  │  (left    │  (right   │  │             │
│   │  │   half)   │   half)   │  │   │  │   half)   │   half)   │  │             │
│   │  └─────┬─────┴─────┬─────┘  │   │  └─────┬─────┴─────┬─────┘  │             │
│   │        │           │        │   │        │           │        │             │
│   │        │ AllReduce │        │   │        │ AllReduce │        │             │
│   │        │◄─────────►│        │   │        │◄─────────►│        │             │
│   │        │           │        │   │        │           │        │             │
│   │        └─────┬─────┘        │   │        └─────┬─────┘        │             │
│   │              │ activations  │   │              │ activations  │             │
│   │              ▼              │   │              ▼              │             │
│   │  PP Stage 1 (Layers 16-31)  │   │  PP Stage 1 (Layers 16-31)  │             │
│   │  ┌───────────┬───────────┐  │   │  ┌───────────┬───────────┐  │             │
│   │  │   GPU 2   │   GPU 3   │  │   │  │   GPU 6   │   GPU 7   │  │             │
│   │  │   (TP=0)  │   (TP=1)  │  │   │  │   (TP=0)  │   (TP=1)  │  │             │
│   │  │           │           │  │   │  │           │           │  │             │
│   │  │  Layers   │  Layers   │  │   │  │  Layers   │  Layers   │  │             │
│   │  │  16-31    │  16-31    │  │   │  │  16-31    │  16-31    │  │             │
│   │  │  (left    │  (right   │  │   │  │  (left    │  (right   │  │             │
│   │  │   half)   │   half)   │  │   │  │   half)   │   half)   │  │             │
│   │  └─────┬─────┴─────┬─────┘  │   │  └─────┬─────┴─────┬─────┘  │             │
│   │        │           │        │   │        │           │        │             │
│   │        │ AllReduce │        │   │        │ AllReduce │        │             │
│   │        │◄─────────►│        │   │        │◄─────────►│        │             │
│   │        │           │        │   │        │           │        │             │
│   │        └─────┬─────┘        │   │        └─────┬─────┘        │             │
│   │              ▼              │   │              ▼              │             │
│   │         Output A            │   │         Output B            │             │
│   └─────────────────────────────┘   └─────────────────────────────┘             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### GPU Assignment in 3D Parallelism

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    GPU ASSIGNMENT: TP=2 × PP=2 × DP=2                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   GPU ID    DP Rank    PP Stage    TP Rank    Layers          Weights           │
│   ═══════   ═══════    ════════    ═══════    ══════          ═══════           │
│   GPU 0       0           0           0       0-15            Left half         │
│   GPU 1       0           0           1       0-15            Right half        │
│   GPU 2       0           1           0       16-31           Left half         │
│   GPU 3       0           1           1       16-31           Right half        │
│   GPU 4       1           0           0       0-15            Left half         │
│   GPU 5       1           0           1       0-15            Right half        │
│   GPU 6       1           1           0       16-31           Left half         │
│   GPU 7       1           1           1       16-31           Right half        │
│                                                                                 │
│   Communication Groups:                                                         │
│   ═════════════════════                                                         │
│   TP groups (AllReduce within):  {0,1}, {2,3}, {4,5}, {6,7}                     │
│   PP groups (Point-to-point):    {0,1}→{2,3}, {4,5}→{6,7}                       │
│   DP groups (Independent):       {0,1,2,3} vs {4,5,6,7}                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
┌──────────────┬────────────────────┬─────────────────────┬──────────────────────┐
│   Strategy   │    What it Splits  │   Communication     │     Best For         │
├──────────────┼────────────────────┼─────────────────────┼──────────────────────┤
│              │                    │                     │                      │
│      TP      │  Weights within    │  All-Reduce every   │  Large layers,       │
│   (Tensor)   │  each layer        │  layer (frequent)   │  fast NVLink         │
│              │  (horizontal)      │                     │  within single node  │
│              │                    │                     │                      │
├──────────────┼────────────────────┼─────────────────────┼──────────────────────┤
│              │                    │                     │                      │
│      PP      │  Layers across     │  Point-to-point     │  Very deep models,   │
│  (Pipeline)  │  stages            │  between stages     │  slower interconnect │
│              │  (vertical)        │  (less frequent)    │  across nodes        │
│              │                    │                     │                      │
├──────────────┼────────────────────┼─────────────────────┼──────────────────────┤
│              │                    │                     │                      │
│      DP      │  Requests across   │  None during        │  High throughput,    │
│    (Data)    │  model replicas    │  inference!         │  model already fits  │
│              │  (replication)     │                     │  in memory           │
│              │                    │                     │                      │
└──────────────┴────────────────────┴─────────────────────┴──────────────────────┘
```

### Practical Guidelines

| GPUs | Common Config | Reason |
|------|---------------|--------|
| 1 | TP=1, PP=1, DP=1 | Single GPU |
| 2 | TP=2 | Usually prefer TP with NVLink |
| 4 | TP=4 or TP=2×PP=2 | TP=4 if good interconnect |
| 8 | TP=8 or TP=4×PP=2 or TP=2×DP=4 | Depends on model & hardware |

**Rule of thumb:**
1. **TP first** (within a node with NVLink)
2. **PP second** (across nodes or if TP alone isn't enough)
3. **DP last** (for throughput scaling when model already fits)

---

## Usage in vLLM

```bash
# Tensor Parallelism only (TP=4)
vllm serve model --tensor-parallel-size 4

# Pipeline Parallelism only (PP=2)
vllm serve model --pipeline-parallel-size 2

# Data Parallelism only (DP=2)
vllm serve model --data-parallel-size 2

# Combined: TP=2, PP=2 on 4 GPUs
vllm serve model --tensor-parallel-size 2 --pipeline-parallel-size 2

# 3D: TP=2, PP=2, DP=2 on 8 GPUs
vllm serve model --tensor-parallel-size 2 --pipeline-parallel-size 2 --data-parallel-size 2
```

### Related vLLM Code

- **Scheduler**: `vllm/v1/core/sched/scheduler.py`
- **Engine Core**: `vllm/v1/engine/core.py`
  - `step()` - used for single batch (no PP or PP=1)
  - `step_with_batch_queue()` - used for PP > 1 to overlap batches
- **Parallel Config**: `vllm/config.py` - `ParallelConfig` class
