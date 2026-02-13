# Ticket 013: RMA SharedWindow Coverage — test_rma_window

## Summary

Create a comprehensive test example for the `SharedWindow<T>` type, exercising all its methods under the `rma` feature flag.

## Dependencies

- ticket-001 (coverage runner script)

## Acceptance Criteria

- [ ] New example `examples/test_rma_window.rs` exists with `required-features = ["rma"]`
- [ ] Example registered in `Cargo.toml`
- [ ] Tests cover:
  - [ ] `SharedWindow::allocate()`
  - [ ] `local_slice()` / `local_slice_mut()`
  - [ ] `remote_slice()`
  - [ ] `fence()`
  - [ ] `lock()` / `LockGuard`
  - [ ] `lock_all()` / `LockAllGuard`
  - [ ] `LockGuard::flush()`
  - [ ] `LockAllGuard::flush()` / `flush_all()`
  - [ ] `raw_handle()`
  - [ ] `comm_size()`
  - [ ] `SharedWindow::drop()`
- [ ] Tests verify data correctness through window operations
- [ ] Example runs successfully with `mpiexec -n 4`
- [ ] Coverage script runs with `--features rma` to capture this

## Files to Create/Modify

- **Create**: `examples/test_rma_window.rs`
- **Modify**: `Cargo.toml` (add `[[example]]` entry with `required-features = ["rma"]`)
- **Modify**: `tests/run_mpi_coverage.sh` (add RMA tests when features include `rma`)
- **Modify**: `tests/run_mpi_tests.sh` (update RMA test section)

## Technical Details

### Step 1: Create the test binary

```rust
//! Integration test for SharedWindow (RMA) operations.
//!
//! Exercises SharedWindow allocate, local/remote slices, fence,
//! lock/lock_all with RAII guards, and flush operations.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_rma_window
//! Build with: cargo build --example test_rma_window --features rma

use ferrompi::{Mpi, SharedWindow, LockType};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(size >= 2, "test_rma_window requires at least 2 processes");

    // ========================================================================
    // Test 1: Allocate and basic properties
    // ========================================================================
    {
        let window: SharedWindow<f64> = SharedWindow::allocate(&world, 10)
            .expect("allocate failed");

        assert!(window.raw_handle() >= 0);
        assert_eq!(window.comm_size(), size);

        // local_slice_mut: initialize local data
        let local = window.local_slice_mut();
        assert_eq!(local.len(), 10);
        for (i, x) in local.iter_mut().enumerate() {
            *x = (rank * 100 + i as i32) as f64;
        }

        // local_slice: verify
        let local_ro = window.local_slice();
        assert_eq!(local_ro.len(), 10);
        for (i, &x) in local_ro.iter().enumerate() {
            assert!((x - (rank * 100 + i as i32) as f64).abs() < f64::EPSILON);
        }

        if rank == 0 { println!("PASS: allocate, local_slice, local_slice_mut"); }
    }

    world.barrier().expect("barrier 1 failed");

    // ========================================================================
    // Test 2: Fence synchronization
    // ========================================================================
    {
        let mut window: SharedWindow<f64> = SharedWindow::allocate(&world, 5)
            .expect("allocate failed");

        let local = window.local_slice_mut();
        for (i, x) in local.iter_mut().enumerate() {
            *x = (rank * 10 + i as i32) as f64;
        }

        window.fence().expect("fence 1 failed");

        // Read remote slice from rank 0
        let remote = window.remote_slice(0).expect("remote_slice failed");
        assert_eq!(remote.len(), 5);
        for (i, &x) in remote.iter().enumerate() {
            assert!((x - (0 * 10 + i as i32) as f64).abs() < f64::EPSILON);
        }

        window.fence().expect("fence 2 failed");

        if rank == 0 { println!("PASS: fence and remote_slice"); }
    }

    world.barrier().expect("barrier 2 failed");

    // ========================================================================
    // Test 3: Lock/unlock with LockGuard
    // ========================================================================
    {
        let mut window: SharedWindow<f64> = SharedWindow::allocate(&world, 5)
            .expect("allocate failed");

        let local = window.local_slice_mut();
        for x in local.iter_mut() { *x = rank as f64; }

        window.fence().expect("fence failed");

        // Lock rank 0's window with shared lock
        {
            let guard = window.lock(LockType::Shared, 0).expect("lock failed");
            guard.flush().expect("flush failed");
            // Guard drops here, releasing lock
        }

        // Lock rank 0's window with exclusive lock (only one at a time)
        if rank == 0 {
            let _guard = window.lock(LockType::Exclusive, 0).expect("exclusive lock failed");
            // Guard drops here
        }

        world.barrier().expect("barrier after locks failed");
        if rank == 0 { println!("PASS: lock/unlock with LockGuard"); }
    }

    world.barrier().expect("barrier 3 failed");

    // ========================================================================
    // Test 4: lock_all / LockAllGuard
    // ========================================================================
    {
        let mut window: SharedWindow<f64> = SharedWindow::allocate(&world, 5)
            .expect("allocate failed");

        let local = window.local_slice_mut();
        for (i, x) in local.iter_mut().enumerate() { *x = (rank + i as i32) as f64; }

        window.fence().expect("fence failed");

        {
            let guard = window.lock_all().expect("lock_all failed");
            guard.flush(0).expect("flush rank 0 failed");
            guard.flush_all().expect("flush_all failed");
            // Guard drops here, releasing all locks
        }

        world.barrier().expect("barrier after lock_all failed");
        if rank == 0 { println!("PASS: lock_all with LockAllGuard"); }
    }

    // ========================================================================
    // Final summary
    // ========================================================================
    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All RMA window tests passed!");
        println!("========================================");
    }
}
```

### Step 2: Register in Cargo.toml

```toml
[[example]]
name = "test_rma_window"
path = "examples/test_rma_window.rs"
required-features = ["rma"]
```

### Step 3: Update runner scripts

In `run_mpi_coverage.sh`, when features include `rma`:

```bash
if [[ "$FEATURES" == *"rma"* ]] || [[ "$FEATURES" == *"numa"* ]]; then
    run_test test_rma_window
fi
```

## Definition of Done

- `window.rs` appears in coverage report with >80% coverage
- All SharedWindow methods and guard types are exercised
- Test passes with `mpiexec -n 4`
