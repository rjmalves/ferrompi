# Ticket 003: MPI Lifecycle Coverage — test_lifecycle example

## Summary

Create a new test example binary that exercises all `Mpi` lifecycle functions: `init()`, `init_thread()`, `Drop`, `version()`, `wtime()`, `is_initialized()`, `is_finalized()`, `thread_level()`, and `world()`.

## Dependencies

- ticket-001 (coverage runner script)

## Acceptance Criteria

- [ ] New example `examples/test_lifecycle.rs` exists
- [ ] Example is registered in `Cargo.toml` as an `[[example]]`
- [ ] Example tests `Mpi::init()` (used implicitly since `init_thread` is the underlying call)
- [ ] Example tests `Mpi::version()` and asserts it returns a non-empty string
- [ ] Example tests `Mpi::wtime()` and asserts it returns a positive value
- [ ] Example tests `Mpi::is_initialized()` returns `true` after init
- [ ] Example tests `Mpi::is_finalized()` returns `false` before drop
- [ ] Example tests `mpi.thread_level()` returns a valid `ThreadLevel`
- [ ] Example tests `mpi.world()` returns a communicator with valid rank/size
- [ ] Example runs successfully with `mpiexec -n 2`
- [ ] Coverage runner script updated to include this example

## Files to Create/Modify

- **Create**: `examples/test_lifecycle.rs`
- **Modify**: `Cargo.toml` (add `[[example]]` entry)
- **Modify**: `tests/run_mpi_coverage.sh` (add to test list)
- **Modify**: `tests/run_mpi_tests.sh` (add to test list)

## Technical Details

### Step 1: Create the test binary

```rust
//! Integration test for MPI lifecycle functions.
//!
//! Exercises Mpi::init, version, wtime, is_initialized, is_finalized,
//! thread_level, and world.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_lifecycle

use ferrompi::{Mpi, ThreadLevel};

fn main() {
    // Test is_initialized before init (should be false if this is first init)
    // Note: can't test this reliably since we need MPI to not be initialized yet

    // Test init_thread (which init() delegates to)
    let mpi = Mpi::init().expect("MPI init failed");

    // Test is_initialized after init
    assert!(Mpi::is_initialized(), "is_initialized should be true after init");
    println!("PASS: is_initialized");

    // Test is_finalized before drop
    assert!(!Mpi::is_finalized(), "is_finalized should be false before drop");
    println!("PASS: is_finalized (false before drop)");

    // Test thread_level
    let level = mpi.thread_level();
    assert!(level >= ThreadLevel::Single, "thread_level should be >= Single");
    println!("PASS: thread_level = {:?}", level);

    // Test version
    let version = Mpi::version().expect("version() failed");
    assert!(!version.is_empty(), "version string should not be empty");
    println!("PASS: version = {}", version);

    // Test wtime
    let t1 = Mpi::wtime();
    assert!(t1 > 0.0, "wtime should return positive value");
    let t2 = Mpi::wtime();
    assert!(t2 >= t1, "wtime should be monotonic");
    println!("PASS: wtime = {}", t1);

    // Test world
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();
    assert!(rank >= 0 && rank < size, "rank should be in [0, size)");
    assert!(size >= 1, "size should be >= 1");
    println!("PASS: world rank={} size={}", rank, size);

    // Test processor_name
    let name = world.processor_name().expect("processor_name failed");
    assert!(!name.is_empty(), "processor_name should not be empty");
    println!("PASS: processor_name = {}", name);

    // Mpi drops here, which calls finalize
    drop(mpi);

    if rank == 0 {
        println!("\n========================================");
        println!("All lifecycle tests passed!");
        println!("========================================");
    }
}
```

### Step 2: Register in Cargo.toml

```toml
[[example]]
name = "test_lifecycle"
path = "examples/test_lifecycle.rs"
```

### Step 3: Add to test runner scripts

Add `run_test test_lifecycle 2` to `tests/run_mpi_tests.sh` in the core tests section.
Add `test_lifecycle` to the test list in `tests/run_mpi_coverage.sh`.

### Step 4: Verify

```bash
cargo build --example test_lifecycle
mpiexec -n 2 ./target/debug/examples/test_lifecycle
```

## Definition of Done

- Example runs cleanly with `mpiexec -n 2`
- `lib.rs` coverage increases significantly (target: >80%)
- `processor_name()` in `comm.rs` is now covered
