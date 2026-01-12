* `torch.distributed`
    * It's a general distributed computing framework for both CPUs and (nvidia, amd, etc.) GPUs, designed to abstract different hardware. 
    * It provides the abstraction layer that allows the same code to work on NVIDIA (NCCL), AMD (RCCL), CPU (Gloo), and other hardware. vLLM then adds platform-specific optimizations on top (Custom All-Reduce for NVIDIA, QuickAllReduce for AMD).
    * structure
    ```
    torch.distributed (Hardware-Agnostic Abstraction)
    ├── Backend: "gloo" → CPU/Ethernet (any hardware)
    ├── Backend: "nccl" → NVIDIA GPUs (uses NCCL library)
    ├── Backend: "nccl" → AMD GPUs (uses RCCL - ROCm version)
    ├── Backend: "mpi" → Generic MPI-based clusters
    └── Custom backends → Your own implementation
    ```
    * Example user code
    ```
        torch.distributed.all_reduce(tensor, group=group)
                    ↓
        torch.distributed runtime detects backend
                    ↓
        ┌───────────┴────────────┐
        ▼                        ▼
        NVIDIA GPU              AMD GPU
        ↓                        ↓
        NCCL library            RCCL library
        ↓                        ↓
        NVLink/InfiniBand       Infinity Fabric/XGMI
    ```
    * torch.distributed provides a unified abstraction for both intra-node and inter-node distributed computing
    ```
        torch.distributed.ProcessGroup API
                │
                ├─ Same interface regardless of topology
                │
                ▼
        ┌───────────────────────────────────────────────────────────────┐
        │  Single Node (Intra-node)         Multi-Node (Inter-node)     │
        ├───────────────────────────────────┬───────────────────────────┤
        │  Node 0                           │  Node 0        Node 1     │
        │  ├─ GPU 0 ─┐                      │  ├─ GPU 0 ─┐  ├─ GPU 2 ─┐ │
        │  ├─ GPU 1 ─┤ NVLink/PCIe          │  ├─ GPU 1 ─┤  ├─ GPU 3 ─┤ │
        │  └─ CPU ───┘                      │  └─ CPU ───┼──┼─ CPU ───┘ │
        │                                   │            │  │           │
        │  NCCL: GPU-GPU (local)            │  InfiniBand/RDMA/TCP      │
        │  Gloo: CPU coordination           │  (inter-node network)     │
        └───────────────────────────────────┴───────────────────────────┘
    ```
    * How torch.distributed abstracts topology
        ```python
            # Same code works for both single-node and multi-node!
            import torch.distributed as dist

            # Initialize - automatically detects topology
            dist.init_process_group(backend="nccl")

            # Rank 0 might be on Node 0, Rank 1 on Node 1
            # Or both on same node - your code doesn't change!
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            # All-reduce works the same way
            tensor = torch.randn(1000, device='cuda')
            dist.all_reduce(tensor)  # ← Handles local OR remote automatically
        ```
    * How it routes communication:
        ```
            All-reduce call from your code
                    ↓
            torch.distributed determines:
                    ↓
            ┌───────┴────────────────────────┐
            ▼                                ▼
            Intra-node (same machine)     Inter-node (different machines)
            ↓                                ↓
            NCCL uses:                     NCCL uses:
            • NVLink (GPU-GPU)             • InfiniBand/RDMA (GPU-GPU across nodes)
            • PCIe (if no NVLink)          • Or RoCE over Ethernet
            • Shared memory (CPU)          • Or TCP/IP (slower fallback)
            ↓                                ↓
            Gloo uses:                     Gloo uses:
            • Shared memory (fast)         • TCP/IP sockets (network)
        ```
    * vLLM builds on torch.distributed, adds hardware-specific optimizations:
        * For NVIDIA:
            - torch.distributed (NCCL backend) → base layer
            - PyNccl → direct NCCL wrapper for more control
            - Custom All-Reduce → CUDA kernels + NVLink
        * For AMD:
            - torch.distributed (RCCL backend) → base layer
            - PyNccl wrapper works with RCCL
            - QuickAllReduce → AMD-specific optimization (MI300 series)
        * Both share:
            - torch.distributed (Gloo backend) → CPU coordination
            - Shared memory → CPU IPC (hardware-agnostic)
* So the overall communication architecture in vLLM:
    ```
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                        Application Layer (vLLM Workers)                    │
    │                        GroupCoordinator / Communication Ops                │
    └───────────────────────────────┬────────────────────────────────────────────┘
                                    ↓
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                   torch.distributed (Foundation Layer)                     │
    │                                                                            │
    │  • ProcessGroup management (create groups, rank assignment)                │
    │  • Backend abstraction (gloo/nccl/rccl/custom)                             │
    │  • Distributed primitives (init, barrier, get_rank, get_world_size)        │
    │                                                                            │
    │  ┌─────────────────────────────┬────────────────────────────────────────┐  │
    │  ▼ CPU Group (Gloo backend)    ▼ Device Group (NCCL/RCCL/Custom)        │  │
    └──┼─────────────────────────────┼────────────────────────────────────────┼──┘
       │                             │                                        │
       ▼                             ▼                                        ▼
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                      CPU Level (Process Coordination)                      │
    │                                                                            │
    │  ┌─────────────────┐                              ┌─────────────────┐      │
    │  │ Worker Process 0│ ← Shared Memory (OS IPC) →   │ Worker Process 1│      │
    │  │   (Python/CPU)  │   • IPC handles              │   (Python/CPU)  │      │
    │  │                 │   • Metadata                 │                 │      │
    │  │                 │ ↕ torch.distributed (Gloo)   │                 │      │
    │  │                 │   • broadcast_object()       │                 │      │
    │  │                 │   • send() / recv()          │                 │      │
    │  │                 │   • barrier()                │                 │      │
    │  └────────┬────────┘                              └────────┬────────┘      │
    │           │                                                │               │
    │           │ Control: Launch kernels, coordinate operations │               │
    └───────────┼────────────────────────────────────────────────┼───────────────┘
                │                                                │
                ▼                                                ▼
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                       GPU Level (Data Transfer)                            │
    │                    via torch.distributed Device Group                      │
    │                                                                            │
    │      ┌────────┐                                          ┌────────┐        │
    │      │ GPU 0  │                                          │ GPU 1  │        │
    │      │Memory  │                                          │Memory  │        │
    │      └───┬────┘                                          └───┬────┘        │
    │          │                                                   │             │
    │          │ ┌───────────────────────────────────────────────┐ │             │
    │          │ │  GPU Communication (via torch.distributed):   │ │             │
    │          │ │                                               │ │             │
    │          ├─┤  1. Custom All-Reduce (vLLM optimization)     │─┤             │
    │          │ │     • Registered as custom torch backend      │ │             │
    │          │ │     • IPC + NVLink direct access              │ │             │
    │          │ │     • Small tensors (<8MB), single node       │ │             │
    │          │ │     • Custom CUDA kernels                     │ │             │
    │          │ │                                               │ │             │
    │          ├─┤  2. PyNccl (Direct NCCL/RCCL wrapper)         │─┤             │
    │          │ │     • Bypasses torch.distributed for GPU ops  │ │             │
    │          │ │     • Large tensors, multi-node support       │ │             │
    │          │ │     • More control for graph mode             │ │             │
    │          │ │     • NCCL (NVIDIA) / RCCL (AMD)              │ │             │
    │          │ │                                               │ │             │
    │          ├─┤  3. torch.distributed.all_reduce() (Fallback) │─┤             │
    │          │ │     • Standard PyTorch NCCL backend           │ │             │
    │          │ │     • Used for broadcast, gather, etc.        │ │             │
    │          │ │     • Hardware-agnostic abstraction           │ │             │
    │          │ └───────────────────────────────────────────────┘ │             │
    │          │                                                   │             │
    │          └───────────── NVLink / Infinity Fabric ────────────┘             │
    │                       (Physical interconnect)                              │
    └────────────────────────────────────────────────────────────────────────────┘

    Hardware Support (via torch.distributed backends):
    • NVIDIA GPUs: NCCL backend → uses NCCL library → NVLink/InfiniBand
    • AMD GPUs:    RCCL backend → uses RCCL library → Infinity Fabric/XGMI  
    • CPUs:        Gloo backend → uses TCP/IP sockets
    • Custom:      User-defined backend → custom implementation

    Legend:
    ━━━ torch.distributed provides infrastructure at all levels
    ─── vLLM adds optimizations on top (Custom All-Reduce, PyNccl wrapper)
    ```
* Communication Backend Libraries:
    ```
        NCCL (NVIDIA Collective Communications Library)
        ├─ Purpose: GPU-to-GPU communication
        ├─ Optimized for: NVIDIA CUDA GPUs
        └─ Transport: NVLink, InfiniBand, PCIe

        RCCL (ROCm Communication Collectives Library)  
        ├─ Purpose: GPU-to-GPU communication
        ├─ Optimized for: AMD ROCm GPUs
        └─ Transport: Infinity Fabric/XGMI, InfiniBand

        Gloo (Facebook/Meta)
        ├─ Purpose: CPU-to-CPU communication
        ├─ Optimized for: Multi-core CPUs
        └─ Transport: Shared memory (intra-node), TCP/IP sockets (inter-node)
    ```
* In torch.distributed
    ```python
        import torch.distributed as dist

        # CPU group - uses Gloo
        cpu_group = dist.new_group(ranks, backend="gloo")
        dist.broadcast_object_list(objects, group=cpu_group)  # CPU memory

        # GPU group - uses NCCL (or RCCL on AMD)
        gpu_group = dist.new_group(ranks, backend="nccl")
        dist.all_reduce(gpu_tensor, group=gpu_group)  # GPU memory
    ```
* vLLM uses both
    ```python
        # In GroupCoordinator.
        # GPU communication - fast tensor transfers
        device_group = torch.distributed.new_group(ranks, backend="nccl")
        # CPU communication - metadata, Python objects, coordination
        cpu_group = torch.distributed.new_group(ranks, backend="gloo")
    ```