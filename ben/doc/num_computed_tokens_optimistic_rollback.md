# num_computed_tokens: Optimistic Advance and Rollback

## Overview

Both sync and async scheduling with speculative decoding follow the same **optimistic advance → rollback** pattern for `num_computed_tokens`. The key difference is when the next `schedule()` call happens relative to the rollback.

## The Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Optimistic Advance → Rollback                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  schedule()                                                                 │
│      │                                                                      │
│      └─→ _update_after_schedule()                                           │
│              │                                                              │
│              └─→ num_computed_tokens += num_scheduled_tokens                │
│                  (OPTIMISTIC: includes unverified spec tokens)              │
│                                                                             │
│  ... forward pass + verification ...                                        │
│                                                                             │
│  update_from_output()                                                       │
│      │                                                                      │
│      └─→ if rejections:                                                     │
│              num_computed_tokens -= num_rejected                            │
│              (ROLLBACK: now equals true verified position)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Code References

### Optimistic Advance (scheduler.py lines 975-989)

```python
def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
    # Advance the number of computed tokens for the request AFTER
    # the request is scheduled.
    # ...
    # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
    #    computed tokens will be adjusted in update_from_output.
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens
    for req_id, num_scheduled_token in num_scheduled_tokens.items():
        request = self.requests[req_id]
        request.num_computed_tokens += num_scheduled_token  # ← Optimistic
```

### Rollback on Rejection (scheduler.py lines 1350-1359)

```python
if scheduled_spec_token_ids:
    num_draft_tokens = len(scheduled_spec_token_ids)
    num_accepted = len(generated_token_ids) - 1
    num_rejected = num_draft_tokens - num_accepted
    # num_computed_tokens represents the number of tokens
    # processed in the current step, considering scheduled
    # tokens and rejections. If some tokens are rejected,
    # num_computed_tokens is decreased by the number of rejected
    # tokens.
    if request.num_computed_tokens > 0:
        request.num_computed_tokens -= num_rejected  # ← Rollback
```

## Sync vs Async: The Key Difference

### Sync + Spec Decode

```
Timeline:
─────────────────────────────────────────────────────────────────────────────→
│                                                                            
│  schedule()                                                                
│      └─→ num_computed_tokens += 6  (1 bonus + 5 spec)                     
│                                                                            
│  ════════════════════════════════════════════════════                      
│  ║  WAIT: Forward pass + Verification in progress  ║                      
│  ════════════════════════════════════════════════════                      
│                                                                            
│  update_from_output()                                                      
│      └─→ 2 rejected: num_computed_tokens -= 2                             
│                                                                            
│  schedule()  ← Only called AFTER rollback completes                       
│      └─→ Sees correct num_computed_tokens                                 
│                                                                            
```

**Result:** The scheduler never "sees" the optimistic value because it waits.

### Async + Spec Decode

```
Timeline:
─────────────────────────────────────────────────────────────────────────────→
│                                                                            
│  schedule()                                                                
│      └─→ num_computed_tokens += 6                                         
│          num_output_placeholders += 6                                      
│                                                                            
│  schedule()  ← Called BEFORE verification completes!                      
│      └─→ Sees optimistic num_computed_tokens                              
│          Uses num_output_placeholders to calculate true position          
│                                                                            
│  update_from_output()                                                      
│      └─→ 2 rejected: num_computed_tokens -= 2                             
│          num_output_placeholders -= 4 (actual received)                   
│                                                                            
```

**Result:** The scheduler may see the optimistic value, so it uses `num_output_placeholders` to track uncertainty.

## Comparison Table

| Aspect | Sync + Spec | Async + Spec |
|--------|-------------|--------------|
| `num_computed_tokens` optimistically advanced? | ✅ Yes | ✅ Yes |
| `num_computed_tokens` rolled back on rejection? | ✅ Yes | ✅ Yes |
| `num_output_placeholders` used? | ❌ No (always 0) | ✅ Yes |
| Next `schedule()` before output? | ❌ No (waits) | ✅ Yes (overlaps) |
| Scheduler sees optimistic value? | ❌ No | ✅ Yes |

## Why Placeholders Only in Async

In **sync mode**, the rollback always completes before the next scheduling decision, so:
- `num_computed_tokens` = true verified tokens (from scheduler's perspective)
- No need for `num_output_placeholders`

In **async mode**, we might schedule again before knowing the verification outcome, so:
- `num_computed_tokens` = verified + unverified (optimistic)
- `num_output_placeholders` = how many are unverified
- True verified position = `num_computed_tokens - num_output_placeholders`

## State Diagram

```
                    ┌─────────────────────────────┐
                    │       schedule()            │
                    │  num_computed_tokens += N   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Optimistic State          │
                    │   (includes unverified)     │
                    └──────────────┬──────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
     [Sync Mode]              [Async Mode]                 │
           │                       │                       │
           │              Can schedule again               │
           │              (uses placeholders)              │
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    update_from_output()     │
                    │  num_computed_tokens -= R   │
                    │  (R = num_rejected)         │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Verified State            │
                    │   (true position)           │
                    └─────────────────────────────┘
```
