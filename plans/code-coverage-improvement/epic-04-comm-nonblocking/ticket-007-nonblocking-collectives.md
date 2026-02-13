# Ticket 007: Nonblocking Collective Operations — test_nonblocking_collectives

## Summary

Create a new test example that exercises ALL nonblocking collective operations (`i`-prefixed methods) in `comm.rs`. The existing `test_nonblocking.rs` only covers P2P nonblocking ops (isend/irecv), not the nonblocking collectives.

## Dependencies

- ticket-001 (coverage runner script)

## Acceptance Criteria

- [ ] New example `examples/test_nonblocking_collectives.rs` exists
- [ ] Example registered in `Cargo.toml`
- [ ] Tests cover all 15 nonblocking collective methods:
  - [ ] `ibroadcast`
  - [ ] `iallreduce`
  - [ ] `ireduce`
  - [ ] `igather`
  - [ ] `iallgather`
  - [ ] `iscatter`
  - [ ] `ibarrier`
  - [ ] `iscan`
  - [ ] `iexscan`
  - [ ] `ialltoall`
  - [ ] `igatherv`
  - [ ] `iscatterv`
  - [ ] `iallgatherv`
  - [ ] `ialltoallv`
  - [ ] `ireduce_scatter_block`
- [ ] Each operation uses `Request::wait()` and verifies results with assertions
- [ ] Example runs successfully with `mpiexec -n 4`

## Files to Create/Modify

- **Create**: `examples/test_nonblocking_collectives.rs`
- **Modify**: `Cargo.toml` (add `[[example]]` entry)
- **Modify**: `tests/run_mpi_coverage.sh` (add to test list)
- **Modify**: `tests/run_mpi_tests.sh` (add to test list)

## Technical Details

### Step 1: Create the test binary

Structure: one test block per nonblocking collective, following the same pattern as `test_collectives.rs` but using the `i`-prefixed methods and waiting on the returned `Request`.

Example pattern:

```rust
// Test: ibroadcast
{
    let mut data = vec![0.0f64; 10];
    if rank == 0 {
        for (i, x) in data.iter_mut().enumerate() {
            *x = (i + 1) as f64;
        }
    }
    let req = world.ibroadcast(&mut data, 0).expect("ibroadcast failed");
    req.wait().expect("ibroadcast wait failed");
    // Verify data...
    if rank == 0 { println!("PASS: ibroadcast"); }
}
```

### Step 2: Follow the same verification patterns as test_collectives.rs

Each nonblocking operation should use the same mathematical verification as its blocking counterpart:

- `iallreduce`: Sum of ranks
- `ireduce`: Sum/Max to root
- `igather`/`iscatter`: Index-based verification
- `iscan`/`iexscan`: Prefix sum verification
- etc.

### Step 3: Register and test

```toml
[[example]]
name = "test_nonblocking_collectives"
path = "examples/test_nonblocking_collectives.rs"
```

```bash
cargo build --example test_nonblocking_collectives
mpiexec -n 4 ./target/debug/examples/test_nonblocking_collectives
```

## Definition of Done

- All 15 nonblocking collective methods have their FFI paths covered
- Example passes with `mpiexec -n 4`
- Coverage of the nonblocking section of `comm.rs` reaches >90%
