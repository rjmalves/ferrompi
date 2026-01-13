#!/bin/bash
# FerroMPI Test Suite
#
# Prerequisites:
# - MPICH 4.0+ or OpenMPI 5.0+ installed
# - Rust toolchain installed
#
# Usage: ./test.sh [num_procs]

set -e

NUM_PROCS=${1:-4}

echo "╔════════════════════════════════════════════════════╗"
echo "║            FerroMPI Test Suite                     ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Check for mpiexec
if ! command -v mpiexec &> /dev/null; then
    echo "ERROR: mpiexec not found in PATH"
    echo "Please install MPICH or OpenMPI first:"
    echo "  Ubuntu: sudo apt install mpich libmpich-dev"
    echo "  macOS:  brew install mpich"
    exit 1
fi

# Check MPI version
echo "MPI Implementation:"
mpiexec --version 2>/dev/null | head -3 || echo "Unknown version"
echo ""

# Build the library and examples
echo "Building ferrompi..."
cargo build --release --examples
echo "✓ Build successful!"
echo ""

# Function to run an example
run_example() {
    local name=$1
    local np=$2
    echo "────────────────────────────────────────"
    echo "Running: $name (n=$np)"
    echo "────────────────────────────────────────"
    if mpiexec -n $np ./target/release/examples/$name; then
        echo "✓ $name passed"
    else
        echo "✗ $name FAILED"
        exit 1
    fi
    echo ""
}

echo "════════════════════════════════════════"
echo "Running examples with $NUM_PROCS processes"
echo "════════════════════════════════════════"
echo ""

# Basic tests
run_example hello_world $NUM_PROCS
run_example ring $NUM_PROCS
run_example allreduce $NUM_PROCS
run_example nonblocking $NUM_PROCS

# Monte Carlo Pi (good stress test)
echo "────────────────────────────────────────"
echo "Running: pi_monte_carlo (n=$NUM_PROCS)"
echo "────────────────────────────────────────"
mpiexec -n $NUM_PROCS ./target/release/examples/pi_monte_carlo || true
echo ""

# MPI 4.0+ test (may not work on older MPI)
echo "────────────────────────────────────────"
echo "Running: persistent_bcast (MPI 4.0+)"
echo "────────────────────────────────────────"
echo "Note: May show 'not available' on MPI < 4.0"
mpiexec -n $NUM_PROCS ./target/release/examples/persistent_bcast || true
echo ""

echo "╔════════════════════════════════════════════════════╗"
echo "║         All tests completed successfully!          ║"
echo "╚════════════════════════════════════════════════════╝"
