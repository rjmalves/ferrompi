# Ticket 005: Point-to-Point Coverage — send, recv, sendrecv, probe, iprobe

## Summary

Verify that all point-to-point methods (blocking send/recv, sendrecv, probe, iprobe) and their error validation paths are covered by existing tests or new tests.

## Dependencies

- ticket-001 (coverage runner script)
- ticket-002 (baseline verification)

## Acceptance Criteria

- [ ] `send()` FFI success path is covered
- [ ] `recv()` FFI success path is covered
- [ ] `sendrecv()` FFI success path is covered
- [ ] `probe()` FFI success path is covered
- [ ] `iprobe()` FFI success path is covered (both found and not-found cases)
- [ ] Buffer validation error paths for send/recv are tested (empty buffer, etc.)
- [ ] Tests exercise multiple data types

## Files to Create/Modify

- **Modify** (if needed): `examples/test_nonblocking.rs` (or create `examples/test_p2p_extra.rs`)
- **Modify**: `Cargo.toml` (if new example)
- **Modify**: `tests/run_mpi_coverage.sh` (add to test list)
- **Modify**: `tests/run_mpi_tests.sh` (add to test list)

## Technical Details

### Step 1: Audit existing coverage

`test_nonblocking.rs` already covers:

- isend/irecv with wait
- isend/irecv with wait_all
- sendrecv (two tests)
- blocking send/recv (even/odd pattern)
- probe (blocking)
- iprobe (polling)
- isend/irecv with test() polling

So the FFI success paths for all P2P operations should already be covered. The main gaps would be:

- Multiple data types (existing tests only use f64)
- Buffer validation error paths in the unit tests (may already exist)

### Step 2: Add type diversity tests if needed

Add a small section testing send/recv with i32, u8, or f32 types.

### Step 3: Verify with coverage report

Check that `send()`, `recv()`, `sendrecv()`, `probe()`, `iprobe()` methods in `comm.rs` show coverage.

## Definition of Done

- All P2P method FFI paths are covered
- Coverage report confirms the P2P section of `comm.rs` is green
