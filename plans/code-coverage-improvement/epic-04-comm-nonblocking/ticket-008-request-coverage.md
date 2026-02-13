# Ticket 008: Nonblocking P2P Extras — Request::test() and wait_all()

## Summary

Ensure that `Request::test()`, `Request::wait()`, `Request::wait_all()`, and `Request::drop()` are fully covered. The existing `test_nonblocking.rs` exercises some of these, but verify coverage completeness.

## Dependencies

- ticket-001 (coverage runner script)
- ticket-002 (baseline verification)

## Acceptance Criteria

- [ ] `Request::wait()` FFI success path is covered
- [ ] `Request::test()` FFI success path is covered (both true and false returns)
- [ ] `Request::wait_all()` FFI success path is covered
- [ ] `Request::drop()` path is covered (request that wasn't waited on)
- [ ] `Request::is_completed()` is exercised before and after wait
- [ ] Coverage of `request.rs` reaches >90%

## Files to Create/Modify

- **Modify** (if needed): `examples/test_nonblocking.rs` (add missing test cases)

## Technical Details

### Step 1: Audit existing coverage of request.rs

The existing `test_nonblocking.rs` already covers:

- `Request::wait()` — Test 1 (ring pattern)
- `Request::wait_all()` — Test 2 (multiple outstanding)
- `Request::test()` — Test 7 (polling)

### Step 2: Check for uncovered paths

The main uncovered paths in `request.rs` are likely:

1. `Request::drop()` when `is_completed` is false — this calls `MPI_Request_free`
2. `Request::is_completed()` getter
3. Error paths in `wait()` and `test()`

### Step 3: Add a test that drops a request without waiting

```rust
// Test: Request drop without wait (exercises MPI_Request_free)
{
    let next = (rank + 1) % size;
    let prev = (rank + size - 1) % size;
    let send_data = vec![1.0f64; 5];
    let mut recv_data = vec![0.0f64; 5];

    let _recv_req = world.irecv(&mut recv_data, prev, 900).expect("irecv failed");
    let send_req = world.isend(&send_data, next, 900).expect("isend failed");
    // Drop send_req without waiting — exercises Request::drop with cancel
    drop(send_req);
    // Still need to complete the recv to avoid hanging
    // ... barrier to synchronize
}
```

Note: Be careful with this — dropping a request without completing it can cause issues. We may need to check if this is safe or if the test should be structured differently.

## Definition of Done

- `request.rs` coverage >90%
- All `Request` methods have at least one exercising path in integration tests
