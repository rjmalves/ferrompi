//! Integration test for Win::fence active-target epoch synchronization.
//!
//! Verifies that `Win::fence` opens and closes an epoch correctly on all
//! ranks, and that `WinFenceAssert` composition works as expected.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_win_fence

use ferrompi::{Mpi, ReduceOp, Win, WinFenceAssert};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_win_fence requires at least 2 processes, got {size}"
    );

    // ========================================================================
    // Probe: Win::fence requires MPI >= 3. Skip gracefully on older builds.
    // ========================================================================
    let version_str = Mpi::version().unwrap_or_default();
    let major: u32 = version_str
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.split('.').next())
        .and_then(|m| m.parse().ok())
        .unwrap_or(0);

    if major < 3 {
        if rank == 0 {
            println!("SKIP: Win::fence requires MPI >= 3 (got {version_str})");
        }
        return;
    }

    let mut local_ok = true;

    // ========================================================================
    // Test 1: Win::fence with WinFenceAssert::default() (no assertion)
    //
    // Allocate a window, call fence twice (open epoch, close epoch).
    // This is the minimal valid fence-pair required before and after RMA ops.
    // ========================================================================
    {
        let win = Win::<f64>::allocate(&world, 16).expect("Win::allocate failed");

        // Open the access/exposure epoch
        if let Err(e) = win.fence(WinFenceAssert::default()) {
            if rank == 0 {
                eprintln!("FAIL: first Win::fence(default) returned error: {e}");
            }
            local_ok = false;
        }

        // Close the epoch
        if let Err(e) = win.fence(WinFenceAssert::default()) {
            if rank == 0 {
                eprintln!("FAIL: second Win::fence(default) returned error: {e}");
            }
            local_ok = false;
        }
    }

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::fence with no assertion");
    }

    // ========================================================================
    // Test 2: Win::fence with WinFenceAssert::none() — explicit zero
    // ========================================================================
    {
        let win = Win::<i32>::allocate(&world, 8).expect("Win::allocate (i32) failed");

        if let Err(e) = win.fence(WinFenceAssert::none()) {
            if rank == 0 {
                eprintln!("FAIL: Win::fence(none) open returned error: {e}");
            }
            local_ok = false;
        }
        if let Err(e) = win.fence(WinFenceAssert::none()) {
            if rank == 0 {
                eprintln!("FAIL: Win::fence(none) close returned error: {e}");
            }
            local_ok = false;
        }
    }

    world.barrier().expect("barrier after test 2 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::fence with explicit none()");
    }

    // ========================================================================
    // Test 3: WinFenceAssert bitflag composition (non-MPI, compile+logic only)
    //
    // Verify that the OR of two non-zero flags has at least two bits set.
    // We use no_store() | no_put() as an example. On real MPI installations
    // these constants are non-zero and distinct, so their OR has >=2 bits set.
    // If the MPI < 3 fallback is active, both equal 0 and the OR is also 0
    // (which is the safe "no assertion" value).
    // ========================================================================
    {
        let combined = WinFenceAssert::no_store() | WinFenceAssert::no_put();
        let no_store_bits = WinFenceAssert::no_store().bits();
        let no_put_bits = WinFenceAssert::no_put().bits();

        // If both are non-zero they must be distinct (MPI guarantees that) and
        // their combination must differ from each individually.
        if no_store_bits != 0
            && no_put_bits != 0
            && (combined.bits() == no_store_bits || combined.bits() == no_put_bits)
        {
            if rank == 0 {
                eprintln!(
                    "FAIL: WinFenceAssert OR result ({}) is not a proper superset \
                     of no_store ({}) | no_put ({})",
                    combined.bits(),
                    no_store_bits,
                    no_put_bits
                );
            }
            local_ok = false;
        }
    }

    world.barrier().expect("barrier after test 3 failed");
    if rank == 0 && local_ok {
        println!("PASS: WinFenceAssert bitflag composition");
    }

    // ========================================================================
    // Sentinel allreduce(Min) — confirms no rank diverged silently
    // ========================================================================
    let global_ok = world
        .allreduce_scalar(local_ok as i32, ReduceOp::Min)
        .expect("sentinel allreduce failed");

    assert!(
        global_ok != 0,
        "test_rma_win_fence: one or more ranks reported failure"
    );

    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All Win::fence tests passed! (3 tests)");
        println!("========================================");
    }
}
