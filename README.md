# FerroMPI

**Lightweight Rust bindings for MPI 4.x with persistent collectives support.**

[![Crates.io](https://img.shields.io/crates/v/ferrompi.svg)](https://crates.io/crates/ferrompi)
[![Documentation](https://docs.rs/ferrompi/badge.svg)](https://docs.rs/ferrompi)
[![License](https://img.shields.io/crates/l/ferrompi.svg)](LICENSE)
[![CI](https://github.com/rjmalves/ferrompi/actions/workflows/test.yml/badge.svg)](https://github.com/rjmalves/ferrompi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/rjmalves/ferrompi/branch/main/graph/badge.svg)](https://codecov.io/gh/rjmalves/ferrompi)
[![Security](https://github.com/rjmalves/ferrompi/actions/workflows/security.yml/badge.svg)](https://github.com/rjmalves/ferrompi/actions/workflows/security.yml)

FerroMPI provides Rust bindings to MPI through a thin C wrapper layer, enabling access to MPI 4.0+ features like **persistent collectives** that are not available in other Rust MPI bindings.

## Features

- 🚀 **MPI 4.0+ support**: Persistent collectives, large-count operations
- 🪶 **Lightweight**: Minimal C wrapper (~700 lines), focused API
- 🔒 **Safe**: Rust-idiomatic API with proper error handling and RAII
- 🔧 **Flexible**: Works with MPICH, OpenMPI, Intel MPI, and Cray MPI
- ⚡ **Fast**: Zero-cost abstractions, direct FFI calls

## Why FerroMPI?

| Feature                | FerroMPI         | rsmpi                  |
| ---------------------- | ---------------- | ---------------------- |
| MPI Version            | 4.1              | 3.1                    |
| Persistent Collectives | ✅               | ❌                     |
| Large Count (>2³¹)     | ✅               | ❌                     |
| API Style              | Minimal, focused | Comprehensive          |
| C Wrapper              | ~700 lines       | None (direct bindings) |

FerroMPI is ideal for:

- Iterative algorithms benefiting from persistent collectives (10-30% speedup)
- Applications with large data transfers (>2GB)
- Users who want a simple, focused MPI API

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ferrompi = "0.1"
```

### Requirements

- **Rust 1.74+**
- **MPICH 4.0+** (recommended) or **OpenMPI 5.0+**

**Ubuntu/Debian:**

```bash
sudo apt install mpich libmpich-dev
```

**macOS:**

```bash
brew install mpich
```

### Hello World

```rust
use ferrompi::{Mpi, ReduceOp};

fn main() -> ferrompi::Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    println!("Hello from rank {} of {}", rank, size);

    // Sum across all ranks
    let sum = world.allreduce_scalar(rank as f64, ReduceOp::Sum)?;
    println!("Rank {}: sum = {}", rank, sum);

    Ok(())
}
```

```bash
cargo build --release
mpiexec -n 4 ./target/release/my_program
```

## Examples

### Blocking Collectives

```rust
use ferrompi::{Mpi, ReduceOp};

let mpi = Mpi::init()?;
let world = mpi.world();

// Broadcast
let mut data = vec![0.0; 100];
if world.rank() == 0 {
    data.fill(42.0);
}
world.broadcast_f64(&mut data, 0)?;

// All-reduce
let send = vec![1.0; 100];
let mut recv = vec![0.0; 100];
world.allreduce_f64(&send, &mut recv, ReduceOp::Sum)?;

// Gather
let my_data = vec![world.rank() as f64];
let mut gathered = vec![0.0; world.size() as usize];
world.gather_f64(&my_data, &mut gathered, 0)?;
```

### Nonblocking Collectives

```rust
use ferrompi::{Mpi, ReduceOp, Request};

let mpi = Mpi::init()?;
let world = mpi.world();

let send = vec![1.0; 1000];
let mut recv = vec![0.0; 1000];

// Start nonblocking operation
let request = world.iallreduce_f64(&send, &mut recv, ReduceOp::Sum)?;

// Do other work while communication proceeds...
expensive_computation();

// Wait for completion
request.wait()?;
// recv now contains the result
```

### Persistent Collectives (MPI 4.0+)

```rust
use ferrompi::{Mpi, ReduceOp};

let mpi = Mpi::init()?;
let world = mpi.world();

// Buffer used for all iterations
let mut data = vec![0.0f64; 1000];

// Initialize ONCE
let mut persistent = world.bcast_init_f64(&mut data, 0)?;

// Use MANY times - amortizes setup cost!
for iter in 0..10000 {
    if world.rank() == 0 {
        data.fill(iter as f64);
    }

    persistent.start()?;
    persistent.wait()?;

    // data contains broadcast result on all ranks
}
// Cleanup on drop
```

## API Reference

### Core Types

| Type                | Description                            |
| ------------------- | -------------------------------------- |
| `Mpi`               | MPI environment handle (init/finalize) |
| `Communicator`      | MPI communicator wrapper               |
| `Request`           | Nonblocking operation handle           |
| `PersistentRequest` | Persistent operation handle (MPI 4.0+) |

### Collective Operations

| Operation | Blocking        | Nonblocking      | Persistent           |
| --------- | --------------- | ---------------- | -------------------- |
| Broadcast | `broadcast_f64` | `ibroadcast_f64` | `bcast_init_f64`     |
| Reduce    | `reduce_f64`    | -                | -                    |
| Allreduce | `allreduce_f64` | `iallreduce_f64` | `allreduce_init_f64` |
| Gather    | `gather_f64`    | -                | -                    |
| Allgather | `allgather_f64` | -                | -                    |
| Scatter   | `scatter_f64`   | -                | -                    |

### Reduction Operations

```rust
pub enum ReduceOp {
    Sum,   // MPI_SUM
    Max,   // MPI_MAX
    Min,   // MPI_MIN
    Prod,  // MPI_PROD
}
```

## Running Tests

```bash
# Build examples
cargo build --release --examples

# Run hello world
mpiexec -n 4 ./target/release/examples/hello_world

# Run all examples
mpiexec -n 4 ./target/release/examples/allreduce
mpiexec -n 4 ./target/release/examples/nonblocking
mpiexec -n 4 ./target/release/examples/persistent_bcast
mpiexec -n 4 ./target/release/examples/pi_monte_carlo
```

## Configuration

### Environment Variables

| Variable         | Description           | Example                     |
| ---------------- | --------------------- | --------------------------- |
| `MPI_PKG_CONFIG` | pkg-config name       | `mpich`, `ompi`             |
| `MPICC`          | MPI compiler wrapper  | `/opt/mpich/bin/mpicc`      |
| `CRAY_MPICH_DIR` | Cray MPI installation | `/opt/cray/pe/mpich/8.1.25` |

### Build Configuration

FerroMPI automatically detects MPI installations via:

1. `MPI_PKG_CONFIG` environment variable
2. pkg-config (`mpich`, `ompi`, `mpi`)
3. `mpicc -show` output
4. `CRAY_MPICH_DIR` (for Cray systems)
5. Common installation paths

## Troubleshooting

### "Could not find MPI installation"

```bash
# Check if MPI is installed
which mpiexec
mpiexec --version

# Set pkg-config name explicitly
export MPI_PKG_CONFIG=mpich
cargo build
```

### "Persistent collectives not available"

Persistent collectives require MPI 4.0+. Check your MPI version:

```bash
mpiexec --version
# MPICH Version: 4.2.0  ✓
# Open MPI 5.0.0        ✓
# MPICH Version: 3.4.2  ✗ (too old)
```

### macOS linking issues

```bash
export DYLD_LIBRARY_PATH=$(brew --prefix mpich)/lib:$DYLD_LIBRARY_PATH
```

## Architecture

```
┌─────────────────────────┐
│    Rust Application     │
├─────────────────────────┤
│  ferrompi (Safe Rust)   │
├─────────────────────────┤
│     ffi.rs (bindings)   │
├─────────────────────────┤
│   ferrompi.c (C layer)  │  ← ~700 lines
├─────────────────────────┤
│   MPICH / OpenMPI       │
└─────────────────────────┘
```

The C layer provides:

- Handle tables for MPI opaque objects
- Automatic large-count operation selection
- Thread-safe request management
- Graceful degradation for MPI <4.0

## License

Licensed under:

- MIT license ([LICENSE](LICENSE))

## Contributing

Contributions welcome! Please ensure:

- All examples pass with `mpiexec -n 4`
- New features include tests and documentation
- Code follows Rust style guidelines (`cargo fmt`, `cargo clippy`)

## Acknowledgments

FerroMPI was inspired by:

- [rsmpi](https://github.com/rsmpi/rsmpi) - Comprehensive MPI bindings for Rust
- The MPI Forum for the excellent MPI 4.0 specification
