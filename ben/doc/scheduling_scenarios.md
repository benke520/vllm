# Scheduling Scenarios: Sync/Async × Spec Decode

## Overview

vLLM's scheduler supports different combinations of synchronous/asynchronous scheduling with or without speculative decoding, creating 4 distinct scenarios.

## Scenario Matrix

| Scenario | Sync/Async | Spec Decode | `num_output_placeholders` | Primary Use Case                    |
|:--------:|:----------:|:-----------:|:-------------------------:|-------------------------------------|
| **1**    | Sync       | No          | Always 0                  | Standard autoregressive             |
| **2**    | Sync       | Yes         | Always 0                  | Spec decode (wait for verification) |
| **3**    | Async      | No          | 0 or 1                    | Pipeline Parallelism (PP)           |
| **4**    | Async      | Yes         | 0 to 1+k                  | PP + Spec decode (most complex)     |

## Detailed Behavior

### Scenario 1: Sync + No Spec Decode

**Standard autoregressive generation**

```
Step 1: Schedule → Forward → Output token
Step 2: Schedule → Forward → Output token  (waits for step 1)
Step 3: ...
```

- Generate 1 token per step
- Wait for output before scheduling next step
- `num_computed_tokens` = verified tokens
- `num_output_placeholders` = 0

---

### Scenario 2: Sync + Spec Decode

**Speculative decoding with synchronous verification**

```
Step 1: Draft k tokens → Schedule verification → Verify → Accept/Reject
Step 2: Draft k tokens → ...  (waits for step 1 verification)
```

- Draft k tokens, verify all at once
- Wait for verification before scheduling next step
- `num_computed_tokens` = verified tokens
- `num_output_placeholders` = 0

---

### Scenario 3: Async + No Spec Decode (Pipeline Parallelism)

**Overlapping batches in pipeline stages**

```
Batch 1: [Stage1] → [Stage2] → [Stage3] → Output
Batch 2:           [Stage1] → [Stage2] → [Stage3] → Output
Batch 3:                      [Stage1] → [Stage2] → ...
```

- Generate 1 token per step
- Schedule next **batch** while current is in pipeline
- Single request waits (skip if `placeholders > 0` with PP)
- `num_computed_tokens` = verified + 1 placeholder (if in-flight)
- `num_output_placeholders` = 0 or 1

**Code enforcement:**
```python
# scheduler.py lines 410-414
if self.use_pp and request.num_output_placeholders > 0:
    req_index += 1
    continue  # Skip - PP doesn't support multiple in-flight steps per request
```

---

### Scenario 4: Async + Spec Decode

**Most complex: speculative decoding with async scheduling**

```
Step 1: Draft k=5 → Schedule (1 bonus + 5 spec = 6 placeholders)
        │
        └→ Verification in-flight, can schedule other requests
        
Output: 3 accepted, 2 rejected
        num_computed_tokens -= 2  (rollback rejected)
        num_output_placeholders -= 4  (received tokens)
```

- Draft k tokens, schedule verification
- Can schedule other work while verification in-flight
- `num_computed_tokens` = verified + (1+k) placeholders
- `num_output_placeholders` = 1 + k (bonus + spec tokens)
- Rollback `num_computed_tokens` by `num_rejected` when output arrives

## Key Variables

| Variable                 | Sync Mode             | Async Mode                                      |
|:-------------------------|:----------------------|:------------------------------------------------|
| `num_computed_tokens`    | = verified tokens     | = verified + placeholders                       |
| `num_output_placeholders`| Always 0              | In-flight token count                           |
| True verified position   | `num_computed_tokens` | `num_computed_tokens - num_output_placeholders` |

## The Check for Max Tokens (Async + Spec)

```python
# scheduler.py lines 417-426
if (
    request.num_output_placeholders > 0
    and request.num_computed_tokens + 2 - request.num_output_placeholders
    >= request.num_prompt_tokens + request.max_tokens
):
    # Skip - even in worst case (all spec rejected), we've hit max_tokens
    continue
```

This formula calculates: *"In the worst case where all spec tokens are rejected, will the guaranteed bonus token still reach max_tokens?"*

- `num_computed_tokens - num_output_placeholders` = true verified position
- `+ 1` for the guaranteed bonus from current in-flight step
- `+ 1` for the guaranteed bonus from potential next step
- Result: minimum guaranteed position

## Summary

| Scenario        | Placeholder Behavior | When `num_computed_tokens` Updates   |
|:----------------|:---------------------|:-------------------------------------|
| Sync + No Spec  | None                 | After each token generated           |
| Sync + Spec     | None                 | After verification completes         |
| Async + No Spec | Max 1 per request    | Optimistically, then settled         |
| Async + Spec    | Max 1+k per step     | Optimistically, rollback on rejection |
