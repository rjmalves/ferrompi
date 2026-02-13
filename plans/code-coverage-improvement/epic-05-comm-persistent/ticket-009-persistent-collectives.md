# Ticket 009: Persistent Collective Operations — test_persistent

## Summary

Create a comprehensive test example that exercises ALL persistent collective operations (`_init` methods) and the `PersistentRequest` lifecycle (start, wait, test, start_all, wait_all, drop).

## Dependencies

- ticket-001 (coverage runner script)

## Acceptance Criteria

- [ ] New example `examples/test_persistent.rs` exists
- [ ] Example registered in `Cargo.toml`
- [ ] Tests cover all persistent collective init methods:
  - [ ] `bcast_init`
  - [ ] `allreduce_init`
  - [ ] `allreduce_init_inplace`
  - [ ] `gather_init`
  - [ ] `reduce_init`
  - [ ] `scatter_init`
  - [ ] `allgather_init`
  - [ ] `scan_init`
  - [ ] `exscan_init`
  - [ ] `alltoall_init`
  - [ ] `gatherv_init`
  - [ ] `scatterv_init`
  - [ ] `allgatherv_init`
  - [ ] `alltoallv_init`
  - [ ] `reduce_scatter_block_init`
- [ ] Tests exercise `PersistentRequest::start()`, `wait()`, `test()` lifecycle
- [ ] Tests exercise `PersistentRequest::start_all()`, `wait_all()` for batch operations
- [ ] Tests exercise `PersistentRequest::is_active()` before and after start
- [ ] Each operation verifies results with assertions
- [ ] Example runs successfully with `mpiexec -n 4`

## Files to Create/Modify

- **Create**: `examples/test_persistent.rs`
- **Modify**: `Cargo.toml` (add `[[example]]` entry)
- **Modify**: `tests/run_mpi_coverage.sh` (add to test list)
- **Modify**: `tests/run_mpi_tests.sh` (add to test list)

## Technical Details

### Step 1: Create the test binary

Structure: each persistent init method is called, then the persistent request is started, waited on, and results verified.

Example pattern:

```rust
// Test: bcast_init + start + wait
{
    let mut data = vec![0.0f64; 10];
    if rank == 0 {
        for (i, x) in data.iter_mut().enumerate() {
            *x = (i + 1) as f64;
        }
    }
    let mut req = world.bcast_init(&mut data, 0).expect("bcast_init failed");

    // Test is_active before start
    assert!(!req.is_active(), "should not be active before start");

    req.start().expect("start failed");

    // Test is_active after start
    assert!(req.is_active(), "should be active after start");

    req.wait().expect("wait failed");

    // Verify data
    for (i, &x) in data.iter().enumerate() {
        assert!((x - (i + 1) as f64).abs() < f64::EPSILON);
    }

    // Persistent requests can be reused
    if rank == 0 {
        data.fill(99.0);
    }
    req.start().expect("restart failed");
    req.wait().expect("rewait failed");
    for &x in &data {
        assert!((x - 99.0).abs() < f64::EPSILON);
    }

    if rank == 0 { println!("PASS: bcast_init"); }
}
```

### Step 2: Test start_all and wait_all

```rust
// Test: start_all + wait_all with multiple persistent requests
{
    let mut data1 = vec![0.0f64; 5];
    let mut data2 = vec![0i32; 3];

    let req1 = world.bcast_init(&mut data1, 0).expect("bcast_init 1 failed");
    let req2 = world.bcast_init(&mut data2, 0).expect("bcast_init 2 failed");

    let mut reqs = vec![req1, req2];
    PersistentRequest::start_all(&mut reqs).expect("start_all failed");
    PersistentRequest::wait_all(&mut reqs).expect("wait_all failed");
    // ... verify
}
```

### Step 3: Test the test() polling pattern

```rust
// Test: PersistentRequest::test() polling
{
    let mut data = vec![0.0f64; 100];
    let mut req = world.bcast_init(&mut data, 0).expect("bcast_init failed");
    req.start().expect("start failed");
    while !req.test().expect("test failed") {
        std::hint::spin_loop();
    }
    // ... verify
}
```

### Step 4: Handle MPI 4.0+ requirement

Persistent collectives require MPI 4.0+. The test should gracefully handle cases where the MPI implementation doesn't support them:

```rust
match world.bcast_init(&mut data, 0) {
    Ok(req) => { /* test normally */ },
    Err(e) => {
        if rank == 0 {
            println!("SKIP: bcast_init not available ({e})");
        }
    }
}
```

### Step 5: Register and test

```toml
[[example]]
name = "test_persistent"
path = "examples/test_persistent.rs"
```

## Definition of Done

- All 15 persistent init methods have their FFI paths covered
- `PersistentRequest` lifecycle (start, wait, test, start_all, wait_all, is_active, drop) is fully covered
- Coverage of `persistent.rs` reaches >90%
- Coverage of the persistent section of `comm.rs` reaches >90%
