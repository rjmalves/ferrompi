//! Minimal test to diagnose MPICH 4.2.0 singleton init issue.
//! This mirrors the test binary pattern (fn main + expect) vs
//! the example pattern (fn main() -> Result<()> + ?).

use ferrompi::Mpi;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    eprintln!(
        "[DIAG] test_minimal: rank={rank}, size={size}, pid={}",
        std::process::id()
    );

    assert!(
        size >= 2,
        "test_minimal requires at least 2 processes, got {size}"
    );

    if rank == 0 {
        println!("PASS: test_minimal (size={size})");
    }
}
