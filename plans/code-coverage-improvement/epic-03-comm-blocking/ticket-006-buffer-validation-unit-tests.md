# Ticket 006: Buffer Validation Error Paths — Additional Unit Tests

## Summary

Add unit tests for buffer validation error paths that aren't covered by existing tests. Many methods in `comm.rs` have buffer size validation checks that return errors before calling FFI — these can be tested without MPI runtime.

## Dependencies

- ticket-002 (baseline verification to identify which error paths are uncovered)

## Acceptance Criteria

- [ ] All buffer size mismatch error paths are tested
- [ ] All count overflow error paths are tested (where applicable)
- [ ] Send/recv size validation error paths are tested
- [ ] Gather/scatter buffer validation at root vs non-root is tested
- [ ] V-variant (gatherv, scatterv, etc.) count/displacement validation is tested
- [ ] Unit tests added to existing `#[cfg(test)]` modules in `comm.rs`

## Files to Create/Modify

- **Modify**: `src/comm.rs` (add tests to `#[cfg(test)]` module)

## Technical Details

### Step 1: Identify uncovered validation paths

Look at the `comm.rs` source for patterns like:

```rust
if send.len() != recv.len() {
    return Err(Error::BufferSizeMismatch { ... });
}
```

Many of these validation checks happen before FFI calls and can be tested in pure unit tests.

### Step 2: Add tests for each validation path

For each method with buffer validation, create a unit test that passes invalid buffers to trigger the error path. Example:

```rust
#[test]
fn allreduce_buffer_size_mismatch() {
    // Create a fake communicator for testing (handle = 0, won't be used)
    let comm = Communicator { handle: 0 };
    let send = vec![1.0f64; 5];
    let mut recv = vec![0.0f64; 3]; // Wrong size
    let result = comm.allreduce(&send, &mut recv, ReduceOp::Sum);
    assert!(matches!(result, Err(Error::BufferSizeMismatch { .. })));
}
```

Note: This requires `Communicator` fields to be accessible in tests, which they are since the tests are in the same module.

### Step 3: Organize tests

Group the new tests by operation category:

- P2P validation tests
- Collective validation tests
- V-variant validation tests

## Definition of Done

- All buffer validation error paths in `comm.rs` have unit tests
- `cargo test` passes
- These paths show as covered in `cargo llvm-cov --lib`
