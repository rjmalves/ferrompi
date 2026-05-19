//! Integration test for the `MPI_COMM_NULL` guard in
//! `ferrompi_comm_create_from_group` (MPI-4 path).
//!
//! Calls `ferrompi_comm_create_from_group` directly via `extern "C"` with a
//! group that excludes some ranks, verifying that those ranks receive a
//! non-`MPI_SUCCESS` return code (previously they would receive a corrupt
//! handle because the NULL guard was missing).
//!
//! The test is skipped gracefully when MPI_VERSION < 4 (the underlying
//! `MPI_Comm_create_from_group` is not available on older MPI runtimes and
//! the C shim returns `MPI_ERR_OTHER` unconditionally in that case).
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_create_from_group_null_handle

use ferrompi::{Error, Mpi, MpiErrorClass, ReduceOp};

// Raw FFI declaration for the C-side shim under test.
//
// We use the Rust API to build the MPI_Group (to keep the test simple), then
// pass its raw handle directly to the C shim to bypass the Rust-side
// Result<Communicator> wrapping and observe the raw return code.
//
// SAFETY invariants for the call below:
//   - group_h is a valid ferrompi group handle obtained from the Rust API.
//   - stringtag is a valid null-terminated C string on the stack.
//   - out_h points to a valid i32 on the stack.
//   - On MPI < 4 the shim returns MPI_ERR_OTHER immediately; no group state
//     is modified.
#[allow(dead_code)]
extern "C" {
    fn ferrompi_comm_create_from_group(
        group_h: i32,
        stringtag: *const std::ffi::c_char,
        out_h: *mut i32,
    ) -> std::ffi::c_int;
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 2,
        "test_create_from_group_null_handle requires exactly 2 processes, got {size}"
    );

    // ========================================================================
    // Build a group containing only rank 0.  Rank 1 is excluded and will
    // receive MPI_COMM_NULL from MPI_Comm_create_from_group.
    // ========================================================================
    let world_group = world.group().expect("world.group() failed");
    let rank0_group = world_group
        .include(&[0])
        .expect("group.include(&[0]) failed");

    let group_h = rank0_group.raw_handle();
    let tag = b"test_null_guard\0";
    let mut out_h: i32 = -2;

    let raw_ret = unsafe {
        // SAFETY: see the invariant comment on the extern "C" block above.
        ferrompi_comm_create_from_group(
            group_h,
            tag.as_ptr().cast::<std::ffi::c_char>(),
            std::ptr::addr_of_mut!(out_h),
        )
    };

    if rank == 1 {
        // Rank 1 is not in the group; MPI_Comm_create_from_group returns
        // MPI_COMM_NULL for this rank.  The new guard must convert that to a
        // non-SUCCESS return code rather than storing MPI_COMM_NULL in the
        // handle table.
        if raw_ret == 0 {
            // raw_ret == MPI_SUCCESS — the guard failed to fire; the
            // out_h slot may now hold MPI_COMM_NULL (corrupt state).
            //
            // On MPI < 4 the shim returns MPI_ERR_OTHER immediately (the
            // #else branch), so a SUCCESS here on rank 1 with MPI >= 4
            // is definitely a bug.
            eprintln!(
                "rank {rank}: FAIL: create_from_group returned MPI_SUCCESS for excluded rank — \
                 MPI_COMM_NULL guard is missing or not compiled in"
            );
            // Participate in the sentinel barrier before exiting.
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            std::process::exit(1);
        }

        let err = Error::from_code(raw_ret);
        println!("PASS rank {rank}: create_from_group for excluded rank returned: {err:?}");
    } else {
        // Rank 0 is in the group; it must either succeed or return MPI_ERR_OTHER
        // (on MPI < 4 the shim returns MPI_ERR_OTHER immediately).
        if raw_ret != 0 {
            let err = Error::from_code(raw_ret);
            match &err {
                Error::Mpi {
                    class: MpiErrorClass::Other,
                    ..
                } => {
                    // MPI < 4: the shim returns MPI_ERR_OTHER for the
                    // unsupported path.  Skip gracefully.
                    println!(
                        "rank {rank}: SKIP: MPI_Comm_create_from_group not available \
                         (MPI_ERR_OTHER returned — MPI < 4.0 runtime)"
                    );
                }
                _ => {
                    eprintln!(
                        "rank {rank}: FAIL: create_from_group for member rank returned \
                         unexpected error: {err:?}"
                    );
                    let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
                    std::process::exit(1);
                }
            }
        } else {
            println!("PASS rank {rank}: create_from_group succeeded for member rank");
        }
    }

    // Sentinel barrier: all ranks must reach this point.
    world
        .allreduce_scalar(1i32, ReduceOp::Min)
        .expect("sentinel allreduce failed");

    if rank == 0 {
        println!("\n========================================");
        println!("All create_from_group_null_handle tests passed!");
        println!("========================================");
    }
}
