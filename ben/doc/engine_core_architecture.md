# vLLM v1 Engine Core Architecture

This document explains the communication architecture between `EngineCoreClient`, `EngineCore`, and `Scheduler` in vLLM v1.

## Overview

The vLLM v1 engine has two operating modes based on the `multiprocess_mode` setting:

1. **Single-process mode** (`multiprocess_mode=False`) - All components in one process
2. **Multi-process mode** (`multiprocess_mode=True`) - Client and EngineCore in separate processes

## Component Relationships

```
EngineCoreClient  ←──→  EngineCore
                            │
                            ├── Scheduler
                            ├── ModelExecutor
                            └── StructuredOutputManager
```

**Key insight:** The `Scheduler` is always **inside** `EngineCore`. However, there are **two layers of IPC** in the system:

1. **EngineCoreClient ↔ EngineCore** (ZMQ sockets) - when `multiprocess_mode=True`
2. **Executor ↔ Worker Processes** (Message Queues / Shared Memory) - when using multi-GPU with `MultiprocExecutor`

---

## Single-Process Mode (`multiprocess_mode=False`)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Single Process                         │
│                                                             │
│  ┌─────────────────┐         direct        ┌─────────────┐  │
│  │ EngineCoreClient│ ─────────────────────►│ EngineCore  │  │
│  │  (InprocClient) │      method calls     │             │  │
│  └─────────────────┘                       │  Scheduler  │  │
│                                            │  Executor   │  │
│                                            └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### How It Works

The `InprocClient` instantiates `EngineCore` directly within the same process:

```python
class InprocClient(EngineCoreClient):
    def __init__(self, *args, **kwargs):
        self.engine_core = EngineCore(*args, **kwargs)  # Direct instantiation

    def get_output(self) -> EngineCoreOutputs:
        outputs, model_executed = self.engine_core.step_fn()  # Direct call
        self.engine_core.post_step(model_executed=model_executed)
        return outputs

    def add_request(self, request: EngineCoreRequest) -> None:
        req, request_wave = self.engine_core.preprocess_add_request(request)
        self.engine_core.add_request(req, request_wave)  # Direct call
```

### Characteristics

| Aspect        | Description                                      |
|---------------|--------------------------------------------------|
| Communication | Direct Python method calls                       |
| Latency       | Minimal (no serialization overhead)              |
| Use Case      | V0-style `LLMEngine`, offline batch processing   |
| Complexity    | Simple                                           |

---

## Multi-Process Mode (`multiprocess_mode=True`)

### Architecture

```
┌───────────────────────────┐              ┌───────────────────────────┐
│       Process 1           │     ZMQ      │        Process 2          │
│      (Frontend)           │   Sockets    │       (Backend)           │
│                           │              │                           │
│  ┌─────────────────────┐  │              │  ┌─────────────────────┐  │
│  │  EngineCoreClient   │  │              │  │     EngineCore      │  │
│  │    (MPClient)       │  │              │  │                     │  │
│  │                     │  │   request    │  │  ┌───────────────┐  │  │
│  │   input_socket ─────┼──┼─────────────►├──┼─►│   Scheduler   │  │  │
│  │                     │  │              │  │  └───────────────┘  │  │
│  │                     │  │   output     │  │                     │  │
│  │   output_socket ◄───┼──┼──────────────┼──┼─ ModelExecutor      │  │
│  │                     │  │              │  │                     │  │
│  └─────────────────────┘  │              │  └─────────────────────┘  │
└───────────────────────────┘              └───────────────────────────┘
```

### How It Works

1. **Client Side (Process 1):** `MPClient` sends `EngineCoreRequest` objects via ZMQ sockets
2. **Server Side (Process 2):** `EngineCore` receives requests, processes them with `Scheduler`, and sends back `EngineCoreOutputs`

```python
class MPClient(EngineCoreClient):
    """
    MPClient: base client for multi-proc EngineCore.
        EngineCore runs in a background process busy loop, getting
        new EngineCoreRequests and returning EngineCoreOutputs

        * pushes EngineCoreRequests via input_socket
        * pulls EngineCoreOutputs via output_socket
    """
```

### Client Variants

| Client          | Mode         | Description                                              |
|-----------------|--------------|----------------------------------------------------------|
| `SyncMPClient`  | Synchronous  | ZMQ + background proc EngineCore (for `LLM`)             |
| `AsyncMPClient` | Asynchronous | ZMQ + background proc EngineCore w/ asyncio (for `AsyncLLM`) |

### Characteristics

| Aspect        | Description                                      |
|---------------|--------------------------------------------------|
| Communication | ZMQ sockets (IPC)                                |
| Serialization | Msgpack encoding/decoding                        |
| Latency       | Higher (serialization + IPC overhead)            |
| Use Case      | Production serving (`vllm serve`), async workloads |
| Benefits      | Process isolation, better resource management    |

---

## Client Selection Logic

The client type is determined by `EngineCoreClient.make_client()`:

```python
@staticmethod
def make_client(
    multiprocess_mode: bool,
    asyncio_mode: bool,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
) -> "EngineCoreClient":
    
    if multiprocess_mode and asyncio_mode:
        return AsyncMPClient(...)      # IPC via ZMQ
    
    if multiprocess_mode and not asyncio_mode:
        return SyncMPClient(...)       # IPC via ZMQ
    
    return InprocClient(...)           # No IPC, same process
```

### Decision Matrix

| `multiprocess_mode` | `asyncio_mode` | Client Type     | IPC?           |
|---------------------|----------------|-----------------|----------------|
| `False`             | `False`        | `InprocClient`  | ❌ No          |
| `True`              | `False`        | `SyncMPClient`  | ✅ Yes (ZMQ)   |
| `True`              | `True`         | `AsyncMPClient` | ✅ Yes (ZMQ)   |
| `False`             | `True`         | Not supported   | -              |

---

## Executor ↔ Worker IPC (Second Layer)

When using multiple GPUs with `MultiprocExecutor`, each GPU worker runs in a **separate process**. This introduces a second layer of IPC:

### Architecture

```
┌─────────────────────┐       ZMQ        ┌─────────────────────────────────────────┐
│   Client Process    │ ◄──────────────► │           EngineCore Process            │
│  (EngineCoreClient) │       IPC        │  ┌───────────┐  ┌────────────────────┐  │
└─────────────────────┘                  │  │ Scheduler │  │  MultiprocExecutor │  │
                                         │  └───────────┘  └─────────┬──────────┘  │
                                         └───────────────────────────┼─────────────┘
                                                                     │ MQ/SHM (IPC)
                                         ┌───────────────────────────┼─────────────┐
                                         │            Worker Processes             │
                                         │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
                                         │  │Worker 0 │  │Worker 1 │  │Worker N │  │
                                         │  │ (GPU 0) │  │ (GPU 1) │  │ (GPU N) │  │
                                         │  └─────────┘  └─────────┘  └─────────┘  │
                                         └─────────────────────────────────────────┘
```

### Communication Mechanisms

| Layer                      | Mechanism                    | When Used                          |
|----------------------------|------------------------------|------------------------------------|
| Client ↔ EngineCore        | ZMQ sockets                  | `multiprocess_mode=True`           |
| Executor ↔ Workers         | Message Queues / Shared Mem  | `MultiprocExecutor` (multi-GPU)    |
| Executor ↔ Ray Workers     | Ray RPC                      | Distributed inference with Ray     |

### Executor Types

| Executor             | Worker Location      | IPC Mechanism              |
|----------------------|----------------------|----------------------------|
| `UniprocExecutor`    | Same process         | Direct method calls        |
| `MultiprocExecutor`  | Separate processes   | Message Queues / Shared Memory |
| `RayExecutor`        | Ray actors           | Ray RPC                    |

---

## EngineCore Step Methods

Within `EngineCore`, there are two step methods depending on whether pipeline parallelism is used:

### `step()` - Synchronous Execution

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    """Schedule, execute, and make output."""
    if not self.scheduler.has_requests():
        return {}, False
    
    scheduler_output = self.scheduler.schedule()
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    model_output = future.result()  # Blocking wait
    
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )
    return engine_core_outputs, True
```

**Flow:** Schedule → Execute → **Wait** → Process → Repeat

### `step_with_batch_queue()` - Pipeline Parallel Execution

Used when pipeline parallelism is enabled to eliminate GPU bubbles:

```python
def step_with_batch_queue(self) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
    """Schedule and execute batches with the batch queue."""
    # Try to schedule new batch if queue not full
    # Don't block unless necessary
    # Return early to keep pipeline filled
```

**Flow:** 
- Schedule new batches into queue (non-blocking)
- Only block when queue is full or no more requests
- Enables GPU0 and GPU1 to process different batches concurrently

```
Without batch queue:          With batch queue:
┌─────┐ ┌─────┐               ┌─────┐ ┌─────┐
│GPU0 │ │GPU1 │               │GPU0 │ │GPU1 │
├─────┤ ├─────┤               ├─────┤ ├─────┤
│ B1  │ │     │               │ B1  │ │     │
├─────┤ ├─────┤               ├─────┤ ├─────┤
│     │ │ B1  │               │ B2  │ │ B1  │  ← Overlap!
├─────┤ ├─────┤               ├─────┤ ├─────┤
│ B2  │ │     │               │ B3  │ │ B2  │
├─────┤ ├─────┤               ├─────┤ ├─────┤
│     │ │ B2  │               │ B4  │ │ B3  │
└─────┘ └─────┘               └─────┘ └─────┘
  Bubbles!                      No bubbles!
```

---

## Summary

1. **Scheduler is always inside EngineCore** - they communicate via direct method calls
2. **Two layers of IPC exist:**
   - **Layer 1:** EngineCoreClient ↔ EngineCore (ZMQ sockets, when `multiprocess_mode=True`)
   - **Layer 2:** Executor ↔ Workers (Message Queues/Shared Memory, when using `MultiprocExecutor`)
3. **Single-process mode**: `InprocClient` makes direct calls to `EngineCore`
4. **Multi-process mode**: `MPClient` uses ZMQ sockets for IPC with `EngineCore` in a background process
5. **Multi-GPU mode**: `MultiprocExecutor` spawns worker processes, communicating via message queues and shared memory
6. **Production serving** typically uses multi-process mode for better isolation and async handling
