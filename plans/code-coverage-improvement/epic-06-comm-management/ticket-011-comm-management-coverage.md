# Ticket 011: Communicator Management — split, duplicate, processor_name

## Summary

Verify that all communicator management methods are covered by existing tests or add new tests. This includes `split()`, `split_type()`, `split_shared()`, `duplicate()`, `processor_name()`, `barrier()`, `rank()`, `size()`, `raw_handle()`.

## Dependencies

- ticket-001 (coverage runner script)
- ticket-002 (baseline verification)

## Acceptance Criteria

- [ ] `Communicator::split()` with valid color is covered
- [ ] `Communicator::split()` with `UNDEFINED` color is covered
- [ ] `Communicator::split_type()` with `Shared` is covered
- [ ] `Communicator::split_shared()` is covered
- [ ] `Communicator::duplicate()` is covered
- [ ] `Communicator::processor_name()` is covered
- [ ] `Communicator::barrier()` is covered
- [ ] `Communicator::rank()` is covered
- [ ] `Communicator::size()` is covered
- [ ] `Communicator::raw_handle()` is covered
- [ ] `Communicator::world()` is covered
- [ ] `Communicator::drop()` for split/duplicated communicators is covered

## Files to Create/Modify

- **Modify** (if needed): `examples/test_comm_split.rs` (add missing cases)
- **Modify** (if needed): `examples/test_lifecycle.rs` (add raw_handle test)

## Technical Details

### Step 1: Audit existing coverage

`test_comm_split.rs` already covers:

- split() with even/odd color (Test 1)
- split() with mod-3 color (Test 3)
- split() with UNDEFINED (Test 4)
- split_type(Shared) (Test 6)
- split_shared() (Test 7)
- duplicate() (Test 8)
- barrier() (used throughout)
- rank() and size() (used throughout)

`test_lifecycle.rs` (from ticket-003) covers:

- processor_name()
- world()

### Step 2: Add raw_handle() test if missing

In `test_lifecycle.rs`:

```rust
// Test raw_handle
let handle = world.raw_handle();
assert!(handle >= 0, "raw_handle should be non-negative");
println!("PASS: raw_handle = {}", handle);
```

### Step 3: Verify Communicator::drop() for derived communicators

When a sub-communicator from split/duplicate goes out of scope, `Communicator::drop()` must call `MPI_Comm_free`. Verify this is covered by the existing test_comm_split.rs (the `sub` variables drop at end of each block).

### Step 4: Verify with coverage report

```bash
./tests/run_mpi_coverage.sh
# Check comm.rs coverage for the management section (lines ~80-230)
```

## Definition of Done

- All communicator management methods show coverage in the report
- Coverage of the management section of `comm.rs` reaches >95%
