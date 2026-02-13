# Ticket 010: Persistent Request Drop Coverage

## Summary

Ensure that `PersistentRequest::drop()` path is covered. When a `PersistentRequest` is dropped, it must call `MPI_Request_free`. This needs to be exercised in the test to achieve full coverage of `persistent.rs`.

## Dependencies

- ticket-009 (persistent collectives test)

## Acceptance Criteria

- [ ] `PersistentRequest::drop()` path is exercised in a test
- [ ] Drop path when `is_active = true` is covered (if applicable)
- [ ] Drop path when `is_active = false` is covered
- [ ] `persistent.rs` coverage reaches >95%

## Files to Create/Modify

- **Modify**: `examples/test_persistent.rs` (add drop-specific test cases)

## Technical Details

### Step 1: Verify drop coverage

After ticket-009, check if `PersistentRequest::drop()` is covered. It should be — every persistent request created in the tests will eventually be dropped. But verify in the coverage report.

### Step 2: Add explicit drop tests if needed

```rust
// Test: PersistentRequest explicit drop (not active)
{
    let mut data = vec![0.0f64; 5];
    let req = world.bcast_init(&mut data, 0).expect("bcast_init failed");
    drop(req); // Drop without ever starting
}

// Test: PersistentRequest explicit drop (after use)
{
    let mut data = vec![0.0f64; 5];
    let mut req = world.bcast_init(&mut data, 0).expect("bcast_init failed");
    req.start().expect("start failed");
    req.wait().expect("wait failed");
    drop(req); // Drop after completing
}
```

## Definition of Done

- Both drop paths (active and inactive) are covered
- `persistent.rs` shows >95% coverage
