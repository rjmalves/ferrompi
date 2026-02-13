//! Minimal test using Result pattern (like passing examples).
//! This mirrors the example pattern (fn main() -> Result<()> + ?).

use ferrompi::{Mpi, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    eprintln!(
        "[DIAG] test_minimal_result: rank={rank}, size={size}, pid={}",
        std::process::id()
    );

    assert!(
        size >= 2,
        "test_minimal_result requires at least 2 processes, got {size}"
    );

    if rank == 0 {
        println!("PASS: test_minimal_result (size={size})");
    }

    Ok(())
}
