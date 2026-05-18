//! Integration test for Win::create and Win::allocate (distributed RMA windows).
//!
//! Verifies that `Win<T>` can be constructed via both constructors, that
//! `raw_handle()` and `comm_size()` return expected values, and that the RAII
//! `Drop` implementation does not produce MPI errors.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_win_create

use ferrompi::{Mpi, ReduceOp, Win};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_win_create requires at least 2 processes, got {size}"
    );

    // ========================================================================
    // Probe: Win::create and Win::allocate require MPI >= 3.
    // Check the MPI version and skip gracefully on older implementations.
    // ========================================================================
    let version_str = Mpi::version().unwrap_or_default();
    // version_str is like "MPI 3.1" or "MPI 4.0"
    let major: u32 = version_str
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.split('.').next())
        .and_then(|m| m.parse().ok())
        .unwrap_or(0);

    if major < 3 {
        if rank == 0 {
            println!("SKIP: Win::create and Win::allocate require MPI >= 3 (got {version_str})");
        }
        return;
    }

    let mut local_ok = true;

    // ========================================================================
    // Test 1: Win::create with a caller-supplied i32 buffer
    // ========================================================================
    {
        let mut buf = vec![0i32; 16];
        let win = Win::create(&world, &mut buf).expect("Win::create failed");

        let handle = win.raw_handle();
        if handle < 0 {
            if rank == 0 {
                eprintln!("FAIL: Win::create raw_handle() = {handle}, expected >= 0");
            }
            local_ok = false;
        }

        let cs = win.comm_size();
        if cs != size {
            if rank == 0 {
                eprintln!("FAIL: Win::create comm_size() = {cs}, expected world size = {size}");
            }
            local_ok = false;
        }

        // Verify local_slice / local_slice_mut round-trip
        {
            let slice = win.local_slice();
            if slice.len() != 16 {
                if rank == 0 {
                    eprintln!(
                        "FAIL: Win::create local_slice len = {}, expected 16",
                        slice.len()
                    );
                }
                local_ok = false;
            }
        }

        // Win dropped here — exercises MPI_Win_free for WinKind::Created
    }

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 {
        println!("PASS: Win::create (i32, 16 elements, raw_handle, comm_size, local_slice)");
    }

    // ========================================================================
    // Test 2: Win::allocate with MPI-managed memory
    // ========================================================================
    {
        let local_count: usize = 32;
        let mut win = Win::<f64>::allocate(&world, local_count).expect("Win::allocate failed");

        let handle = win.raw_handle();
        if handle < 0 {
            if rank == 0 {
                eprintln!("FAIL: Win::allocate raw_handle() = {handle}, expected >= 0");
            }
            local_ok = false;
        }

        let cs = win.comm_size();
        if cs != size {
            if rank == 0 {
                eprintln!("FAIL: Win::allocate comm_size() = {cs}, expected world size = {size}");
            }
            local_ok = false;
        }

        // Verify local_slice_mut write-then-read round-trip
        {
            let slice = win.local_slice_mut();
            if slice.len() != local_count {
                if rank == 0 {
                    eprintln!(
                        "FAIL: Win::allocate local_slice_mut len = {}, expected {local_count}",
                        slice.len()
                    );
                }
                local_ok = false;
            } else {
                for (i, x) in slice.iter_mut().enumerate() {
                    *x = (rank as f64) * 100.0 + i as f64;
                }
            }
        }

        // Read back via local_slice
        {
            let slice = win.local_slice();
            for (i, &x) in slice.iter().enumerate() {
                let expected = (rank as f64) * 100.0 + i as f64;
                if (x - expected).abs() > f64::EPSILON {
                    if rank == 0 {
                        eprintln!(
                            "FAIL: Win::allocate local_slice[{i}] = {x}, expected {expected}"
                        );
                    }
                    local_ok = false;
                    break;
                }
            }
        }

        // Win dropped here — exercises MPI_Win_free for WinKind::Allocated
    }

    world.barrier().expect("barrier after test 2 failed");
    if rank == 0 {
        println!("PASS: Win::allocate (f64, 32 elements, raw_handle, comm_size, local_slice_mut, local_slice)");
    }

    // ========================================================================
    // Test 3: Sentinel allreduce(Min) — confirms no rank diverged silently
    // (per epic-04 invariant)
    // ========================================================================
    let global_ok = world
        .allreduce_scalar(local_ok as i32, ReduceOp::Min)
        .expect("sentinel allreduce failed");

    assert!(
        global_ok != 0,
        "test_rma_win_create: one or more ranks reported failure"
    );

    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All Win create/allocate tests passed! (2 tests)");
        println!("========================================");
    }
}
