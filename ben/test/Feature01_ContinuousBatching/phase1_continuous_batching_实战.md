# Phase 1: Continuous Batching 实战Debug指南

## 🎯 学习目标
通过debug实际代码，深入理解vLLM的Continuous Batching机制：
- Batch如何动态增长和收缩
- Sequence的完整生命周期
- Prefill和Decode的交错执行
- Scheduler的决策时机和逻辑

---

## 📝 准备测试代码

### Step 1: 创建测试脚本

创建文件 `vllm/ben/test/test_continuous_batching.py`:

```python
"""
测试Continuous Batching的核心行为
通过不同长度的prompts观察batch的动态变化
"""
import sys
import time
from vllm import LLM, SamplingParams

def test_continuous_batching():
    """
    核心测试：3个不同长度的请求
    - 短请求：快速完成
    - 长请求：占用时间长
    - 中等请求：介于两者之间
    
    观察重点：
    1. 它们如何被分配到同一个batch
    2. 短请求先完成后，batch如何收缩
    3. 新请求到达时，batch如何扩张
    """
    
    # 初始化LLM（使用小模型快速测试）
    print("=" * 80)
    print("初始化vLLM...")
    print("=" * 80)
    
    llm = LLM(
        model="facebook/opt-125m",  # 小模型，快速
        max_model_len=512,
        # 关键参数：观察batching行为
        max_num_batched_tokens=2048,  # 单step最大token数
        max_num_seqs=256,  # 单step最大sequence数
    )
    
    # 准备不同长度的prompts
    prompts = [
        # 请求1：短prompt，期望快速完成
        "Short prompt",
        
        # 请求2：长prompt，会占用更多时间
        "This is a much longer prompt that will take more time to process. " * 10,
        
        # 请求3：中等长度
        "This is a medium length prompt for testing purposes.",
    ]
    
    # Sampling参数：让不同请求生成不同长度的输出
    sampling_params_list = [
        SamplingParams(temperature=0.0, max_tokens=10),   # 短输出
        SamplingParams(temperature=0.0, max_tokens=100),  # 长输出
        SamplingParams(temperature=0.0, max_tokens=50),   # 中等输出
    ]
    
    print("\n" + "=" * 80)
    print("测试场景设置：")
    print("=" * 80)
    for i, (prompt, params) in enumerate(zip(prompts, sampling_params_list)):
        print(f"请求 {i+1}:")
        print(f"  Prompt长度: {len(prompt)} chars")
        print(f"  Max tokens: {params.max_tokens}")
        print()
    
    # 开始推理
    print("=" * 80)
    print("开始推理 - 观察Continuous Batching行为...")
    print("=" * 80)
    print("\n🔍 在这里设置断点，开始单步调试！\n")
    
    start_time = time.time()
    
    # 🎯 关键：这里是入口点，从这里开始debug
    outputs = llm.generate(prompts, sampling_params_list)
    
    end_time = time.time()
    
    # 输出结果
    print("\n" + "=" * 80)
    print("推理完成！结果：")
    print("=" * 80)
    for i, output in enumerate(outputs):
        print(f"\n请求 {i+1}:")
        print(f"  生成的token数: {len(output.outputs[0].token_ids)}")
        print(f"  生成的文本: {output.outputs[0].text[:100]}...")
    
    print(f"\n总耗时: {end_time - start_time:.2f}秒")


def test_batch_arrival():
    """
    测试2：请求分批到达的场景
    模拟真实serving场景：请求不是一次性全部到达
    """
    print("\n" + "=" * 80)
    print("测试场景2：分批到达的请求")
    print("=" * 80)
    
    llm = LLM(
        model="facebook/opt-125m",
        max_model_len=256,
    )
    
    # 第一批请求
    batch1 = ["First batch request 1", "First batch request 2"]
    params1 = SamplingParams(temperature=0.0, max_tokens=20)
    
    print("\n发送第一批请求...")
    outputs1 = llm.generate(batch1, params1)
    
    # 第二批请求（在第一批处理过程中到达）
    batch2 = ["Second batch request 1"]
    params2 = SamplingParams(temperature=0.0, max_tokens=30)
    
    print("\n发送第二批请求...")
    outputs2 = llm.generate(batch2, params2)
    
    print("\n所有请求完成！")


if __name__ == "__main__":
    # 运行主测试
    test_continuous_batching()
    
    # 可选：运行第二个测试
    # test_batch_arrival()
```

---

## 🔍 Debug路径：从入口到核心

### 第一步：从`llm.generate()`开始

**文件**: `vllm/entrypoints/llm.py`

```python
class LLM:
    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None,
        ...
    ) -> List[RequestOutput]:
        """
        🎯 断点1：在这里设置第一个断点
        
        观察：
        - prompts如何被处理
        - sampling_params如何对应
        """
        
        # 这里会调用engine的generate
        # 继续跟进 self.llm_engine.generate()
```

**观察点**：
- [ ] `prompts`的数量和内容
- [ ] 每个prompt对应的`sampling_params`
- [ ] Request ID如何生成

---

### 第二步：进入LLMEngine

**文件**: `vllm/engine/llm_engine.py`

```python
class LLMEngine:
    def generate(self, ...):
        """
        🎯 断点2：Engine的入口
        
        这里是核心调度循环
        """
        
        # 添加请求到scheduler
        # 继续跟进 self._add_request()
        
    def _add_request(self, ...):
        """
        🎯 断点3：请求被添加到系统
        
        观察：
        - Request如何变成SequenceGroup
        - 如何进入scheduler的waiting queue
        """
        
        # 重点看这一行
        self.scheduler.add_seq_group(seq_group)
    
    def step(self):
        """
        🎯 断点4：核心调度循环 - 每个token generation step
        
        ⭐ 这是最重要的函数！！！
        Continuous Batching的核心逻辑都在这里
        
        观察每个step：
        1. Scheduler如何选择sequences
        2. Batch如何组成
        3. 执行后状态如何更新
        """
        
        # 1️⃣ Scheduler调度 - 决定这一step处理哪些sequences
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        
        # 🔍 在这里打印观察
        print(f"\n{'='*60}")
        print(f"STEP {self.step_count}")
        print(f"{'='*60}")
        print(f"本step要处理的sequences数量: {len(seq_group_metadata_list)}")
        
        # 打印每个sequence的状态
        for i, metadata in enumerate(seq_group_metadata_list):
            seq = metadata.seq_data
            print(f"  Seq {i}: ID={metadata.request_id}")
            print(f"    当前长度: {len(seq)} tokens")
            print(f"    状态: {metadata.is_prompt} (is_prompt)")
        
        # 2️⃣ 执行推理
        output = self.model_executor.execute_model(...)
        
        # 3️⃣ 更新状态
        self._process_model_outputs(output, ...)
        
        # 4️⃣ 检查是否有sequence完成
        # 完成的sequence会从batch中移除
```

---

### 第三步：深入Scheduler核心

**文件**: `vllm/core/scheduler.py`

```python
class Scheduler:
    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        """
        🎯 断点5：Scheduler的核心决策函数
        
        ⭐⭐⭐ 这是Continuous Batching的灵魂！！！
        
        每个step都会调用这个函数来决定：
        1. 哪些sequences进入这个batch
        2. 它们是在做prefill还是decode
        3. 资源是否足够（KV cache blocks）
        """
        
        # 返回值
        scheduled: List[SequenceGroupMetadata] = []
        
        # 1️⃣ 调度正在运行的sequences（RUNNING状态）
        # 这些是已经在做decode的sequences
        running = self._schedule_running(...)
        
        # 🔍 观察点
        print(f"\n📊 Scheduler状态:")
        print(f"  RUNNING sequences: {len(self.running)}")
        print(f"  WAITING sequences: {len(self.waiting)}")
        print(f"  SWAPPED sequences: {len(self.swapped)}")
        
        # 2️⃣ 调度等待中的sequences（WAITING状态）
        # 这些是新到达的请求，需要做prefill
        waiting = self._schedule_waiting(...)
        
        # 3️⃣ 调度被swap out的sequences（如果有）
        swapped = self._schedule_swapped(...)
        
        # 组合成最终的batch
        scheduled = running + waiting + swapped
        
        return scheduled, scheduler_outputs
    
    def _schedule_running(self, ...):
        """
        🎯 断点6：调度RUNNING状态的sequences
        
        这些sequences已经完成了prefill，现在在做decode
        每个step生成1个token
        """
        
        # 遍历所有running sequences
        for seq_group in self.running:
            # 检查是否可以继续（KV cache够不够）
            if self._can_append_slots(seq_group):
                # 可以继续decode
                scheduled.append(seq_group)
            else:
                # 资源不足，可能需要preempt（抢占）
                self._preempt(seq_group)
        
        return scheduled
    
    def _schedule_waiting(self, ...):
        """
        🎯 断点7：调度WAITING状态的sequences
        
        这些是新请求，需要做prefill
        
        关键逻辑：
        - 是否要做chunked prefill
        - 能一次性处理多少个新请求
        """
        
        for seq_group in self.waiting:
            # 检查资源
            if self._can_allocate(seq_group):
                # 分配KV cache blocks
                self._allocate(seq_group)
                scheduled.append(seq_group)
                
                # 从WAITING移到RUNNING
                self.waiting.remove(seq_group)
                self.running.append(seq_group)
            else:
                # 资源不足，留在waiting queue
                break
        
        return scheduled
```

---

## 🎬 实战Debug步骤

### Step 1: 设置断点并运行

```bash
cd /home/benke/Workspace/vLLM/vllm

# 使用Python调试器运行
python -m pdb ben/test/test_continuous_batching.py
```

或使用VSCode调试：
1. 打开 `test_continuous_batching.py`
2. 在关键位置设置断点
3. F5启动调试

### Step 2: 关键断点位置（建议顺序）

**第一轮：宏观理解**
1. ✅ `test_continuous_batching.py:62` - `llm.generate()` 入口
2. ✅ `llm_engine.py:step()` - 核心调度循环
3. ✅ `scheduler.py:schedule()` - Scheduler决策

**第二轮：细节深入**
4. ✅ `scheduler.py:_schedule_running()` - Decode调度
5. ✅ `scheduler.py:_schedule_waiting()` - Prefill调度
6. ✅ `llm_engine.py:_process_model_outputs()` - 状态更新

### Step 3: 每个断点的观察清单

#### 在`llm_engine.step()`处（最重要！）

打印这些信息：
```python
# 在step函数开始处添加
print(f"\n{'='*80}")
print(f"🔄 STEP {self.step_count}")
print(f"{'='*80}")

# 在scheduler.schedule()调用后
print(f"📊 Scheduler输出:")
print(f"  本step要处理: {len(seq_group_metadata_list)} sequences")
print(f"  Scheduled blocks: {scheduler_outputs.num_batched_tokens}")

for i, metadata in enumerate(seq_group_metadata_list):
    print(f"\n  Sequence {i+1}:")
    print(f"    Request ID: {metadata.request_id}")
    print(f"    是否Prefill: {metadata.is_prompt}")
    print(f"    当前长度: {metadata.seq_data.get_len()}")
    print(f"    状态: {metadata.state}")
```

#### 在`scheduler.schedule()`处

```python
print(f"\n📋 Scheduler队列状态:")
print(f"  RUNNING: {len(self.running)} sequences")
print(f"  WAITING: {len(self.waiting)} sequences")
print(f"  SWAPPED: {len(self.swapped)} sequences")
print(f"  Available blocks: {self.block_manager.get_num_free_gpu_blocks()}")
```

---

## 📊 观察重点与预期行为

### 场景1：3个请求同时到达

**Step 1-N (Prefill阶段)**:
```
期望看到：
- 所有3个请求都在WAITING队列
- Scheduler尝试调度它们
- 根据max_num_batched_tokens，可能：
  - 全部一起prefill（如果token总数 < limit）
  - 分批prefill（如果超过limit）

观察：
✓ 哪些请求被选中做prefill
✓ 每个请求的prefill是否被chunk
✓ KV cache block的分配
```

**Step N+1 开始 (Decode阶段)**:
```
期望看到：
- Prefill完成的请求进入RUNNING状态
- 每个step，每个RUNNING sequence生成1个token
- 短请求先完成，从RUNNING队列移除
- Batch size动态减少

关键观察：
✓ 每个step的batch大小变化
✓ Sequence完成顺序（短→中→长）
✓ 完成的sequence何时被移除
```

### 场景2：理解Continuous Batching的优势

**对比传统Static Batching**:
```
Static Batching:
  Step 1: [Req1, Req2, Req3] - prefill
  Step 2: [Req1, Req2, Req3] - decode (等待最长的完成)
  ...
  Step 100: [Req3] - 仍在等待Req3完成
  ❌ Req1, Req2早就完成了，但GPU在浪费

Continuous Batching (vLLM):
  Step 1: [Req1, Req2, Req3] - prefill
  Step 2: [Req1, Req2, Req3] - decode
  Step 12: [Req2, Req3] - Req1完成，移除 ✅
  Step 52: [Req3] - Req2完成，移除 ✅
  Step 100: [] - Req3完成 ✅
  
  ✅ 每个请求完成后立即释放资源
  ✅ 新请求可以随时插入
  ✅ GPU利用率最大化
```

---

## 🎯 自检问题（Debug过程中回答）

### 基础理解
- [ ] **Q1**: 一个`SequenceGroup`包含什么？
  - 提示：看`sequence.py`中的`SequenceGroup`类

- [ ] **Q2**: `is_prompt=True`和`is_prompt=False`分别代表什么？
  - 提示：Prefill vs Decode

- [ ] **Q3**: 为什么需要`max_num_batched_tokens`这个限制？
  - 提示：GPU显存和计算能力的权衡

### 进阶理解
- [ ] **Q4**: Batch大小是如何动态变化的？
  - 在哪个函数中决定？
  - 基于什么条件？

- [ ] **Q5**: 短请求完成后，它的KV cache blocks发生了什么？
  - 提示：观察`block_manager.free()`调用

- [ ] **Q6**: 如果在decode过程中有新请求到达会怎样？
  - 它会等到下一个step吗？
  - 还是会被立即调度？

### 专家级理解
- [ ] **Q7**: 为什么Continuous Batching能降低tail latency？
  - 用你观察到的具体数据支持

- [ ] **Q8**: 如果一个请求特别长（10000 tokens），会发生什么？
  - Prefill会被chunk吗？
  - 会阻塞其他请求吗？

- [ ] **Q9**: 如果要实现fairness-aware scheduling，你会怎么改scheduler？
  - 提示：现在是FIFO，如何改成priority-based？

---

## 🔧 调试技巧

### 技巧1：添加详细日志

在关键位置添加打印（建议创建自己的debug分支）：

```python
# 在 vllm/core/scheduler.py 的 schedule() 函数中
def schedule(self):
    # 在函数开始处
    if os.environ.get('VLLM_DEBUG_SCHEDULER'):
        print(f"\n{'='*80}")
        print(f"🔍 Scheduler.schedule() called")
        print(f"  Running: {len(self.running)}")
        print(f"  Waiting: {len(self.waiting)}")
        print(f"  Free GPU blocks: {self.block_manager.get_num_free_gpu_blocks()}")
```

然后运行时：
```bash
export VLLM_DEBUG_SCHEDULER=1
python ben/test/test_continuous_batching.py
```

### 技巧2：可视化Batch变化

创建一个简单的可视化脚本：

```python
# 在test脚本中添加
class BatchTracker:
    def __init__(self):
        self.steps = []
    
    def record_step(self, step_num, batch_size, seq_ids):
        self.steps.append({
            'step': step_num,
            'batch_size': batch_size,
            'seq_ids': seq_ids
        })
    
    def plot(self):
        # 简单的ASCII可视化
        print("\n📊 Batch Size Over Time:")
        for record in self.steps:
            bar = '█' * record['batch_size']
            print(f"Step {record['step']:3d}: {bar} ({record['batch_size']})")
```

### 技巧3：对比实验

```python
def compare_batch_modes():
    """
    对比不同配置下的行为
    """
    configs = [
        {"max_num_batched_tokens": 512, "name": "Small batch"},
        {"max_num_batched_tokens": 2048, "name": "Large batch"},
    ]
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"测试配置: {config['name']}")
        print(f"{'='*80}")
        
        llm = LLM(
            model="facebook/opt-125m",
            max_num_batched_tokens=config['max_num_batched_tokens']
        )
        
        # 运行相同的测试
        # 对比结果...
```

---

## 📝 学习输出（完成Phase 1后）

### 必须完成的输出

1. **执行时间线图**
   - 画出3个请求从到达到完成的完整timeline
   - 标注每个step的batch组成
   - 标注prefill/decode切换点

2. **Scheduler决策流程图**
   - `schedule()` → `_schedule_running()` → `_schedule_waiting()`
   - 决策条件（资源检查、状态转换）

3. **与Kafka的类比笔记**
   - Consumer group rebalance vs Continuous batching
   - Partition assignment vs Sequence scheduling
   - Offset commit vs Token generation

4. **回答所有自检问题**
   - 用你在debug中观察到的具体数据支持

---

## 🚀 进阶探索（可选）

如果时间充足，可以尝试：

### 1. 修改Scheduler策略
```python
# 实现一个简单的priority-based scheduler
class PriorityScheduler(Scheduler):
    def _schedule_waiting(self):
        # 按priority排序而不是FIFO
        self.waiting.sort(key=lambda x: x.priority, reverse=True)
        # ... 其余逻辑
```

### 2. 模拟真实Serving场景
```python
import threading
import time

def simulate_request_arrival():
    """
    模拟请求随机到达
    """
    requests = []
    
    def send_request(prompt, delay):
        time.sleep(delay)
        # 发送请求
        
    # 请求1：立即
    threading.Thread(target=send_request, args=("Prompt 1", 0)).start()
    # 请求2：1秒后
    threading.Thread(target=send_request, args=("Prompt 2", 1)).start()
    # 请求3：2秒后
    threading.Thread(target=send_request, args=("Prompt 3", 2)).start()
```

### 3. 性能profiling
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# 运行测试
llm.generate(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # 打印top 20耗时函数
```

---

## ✅ Phase 1 完成标志

你可以认为Phase 1完成，当你能够：

- [x] **解释**：为什么vLLM的batch是"动态的"？
- [x] **画出**：3个请求的完整执行timeline
- [x] **指出**：Scheduler在哪些点做决策，基于什么条件
- [x] **对比**：Continuous Batching vs Static Batching的本质区别
- [x] **回答**：如果要实现fairness，需要改哪些代码
- [x] **类比**：用Kafka的概念解释vLLM的调度逻辑

---

## 🎓 Phase 1 → Phase 2 过渡

完成Continuous Batching后，你会自然产生这些疑问：

1. **KV cache是如何分配和管理的？**
   → 这就是Phase 2 (PagedAttention)

2. **长请求的prefill会阻塞其他请求吗？**
   → 这就是Phase 3 (Chunked Prefill)

3. **资源不够时如何决定拒绝请求？**
   → 这就是Phase 5 (Admission Control)

带着这些问题，你就可以进入下一个Phase了！

---

**记住最重要的**：
> "不要只看代码在'做什么'，要理解'为什么这样做'"
> 
> "如果没有Continuous Batching，系统会在哪崩？" → 回答这个问题，你就真正理解了。

祝Debug顺利！🚀
