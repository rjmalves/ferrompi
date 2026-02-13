//! Minimal diagnostic binary with fn main() (no Result).
//! Named WITHOUT the test_ prefix to check if binary name matters.

use ferrompi::Mpi;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    eprintln!(
        "[DIAG] diag_minimal: rank={rank}, size={size}, pid={}",
        std::process::id()
    );

    assert!(
        size >= 2,
        "diag_minimal requires at least 2 processes, got {size}"
    );

    if rank == 0 {
        println!("PASS: diag_minimal (size={size})");
    }
}
