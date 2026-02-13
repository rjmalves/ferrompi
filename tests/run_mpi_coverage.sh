#!/bin/bash
# ==========================================================================
# FerroMPI MPI Coverage Runner
#
# Runs MPI integration test examples under cargo llvm-cov instrumentation,
# collecting coverage data from all example binaries and unit tests,
# then produces merged LCOV and HTML reports.
#
# Usage:
#   ./tests/run_mpi_coverage.sh               # Run with default settings
#   ./tests/run_mpi_coverage.sh rma           # Run with rma feature (includes shared_memory)
#   ./tests/run_mpi_coverage.sh numa          # Run with numa feature (implies rma)
#   MPI_NP=8 ./tests/run_mpi_coverage.sh      # Run with 8 processes
#
# Environment:
#   MPI_NP      — Number of MPI processes (default: 4)
#   MPIEXEC     — Path to mpiexec (default: mpiexec)
#
# Prerequisites:
#   - MPICH 4.0+ or OpenMPI 5.0+ installed
#   - Rust toolchain installed
#   - cargo-llvm-cov installed (cargo install cargo-llvm-cov)
#
# Output:
#   - lcov.info                        — LCOV coverage data
#   - target/llvm-cov/html/index.html  — HTML coverage report
#   - Summary printed to stdout
# ==========================================================================
set -euo pipefail

# Configuration
FEATURES="${1:-}"
NP="${MPI_NP:-4}"
MPIEXEC="${MPIEXEC:-mpiexec}"

# Auto-detect OpenMPI and add --oversubscribe to avoid binding errors in CI
MPIEXEC_ARGS=""
if "$MPIEXEC" --version 2>&1 | grep -q "Open MPI"; then
    MPIEXEC_ARGS="--oversubscribe"
fi

PASSED=0
FAILED=0
SKIPPED=0
FAILED_TESTS=()

# Color codes (if terminal supports it)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    RESET='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    CYAN=''
    BOLD=''
    RESET=''
fi

# ========================================================================
# Helper functions
# ========================================================================

print_header() {
    echo ""
    echo -e "${BOLD}╔════════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}║       FerroMPI MPI Coverage Runner                ║${RESET}"
    echo -e "${BOLD}╚════════════════════════════════════════════════════╝${RESET}"
    echo ""
    echo -e "  Processes:   ${CYAN}${NP}${RESET}"
    echo -e "  Features:    ${CYAN}${FEATURES:-default}${RESET}"
    echo -e "  mpiexec:     ${CYAN}${MPIEXEC}${RESET}"
    echo -e "  mpiexec args:${CYAN}${MPIEXEC_ARGS:- (none)}${RESET}"
    echo ""
}

check_prerequisites() {
    if ! command -v "$MPIEXEC" &> /dev/null; then
        echo -e "${RED}ERROR: ${MPIEXEC} not found in PATH${RESET}"
        echo "Please install MPICH or OpenMPI first:"
        echo "  Ubuntu: sudo apt install mpich libmpich-dev"
        echo "  macOS:  brew install mpich"
        exit 1
    fi

    if ! command -v cargo &> /dev/null; then
        echo -e "${RED}ERROR: cargo not found in PATH${RESET}"
        echo "Please install the Rust toolchain first."
        exit 1
    fi

    if ! cargo llvm-cov --version &> /dev/null; then
        echo -e "${RED}ERROR: cargo-llvm-cov not found${RESET}"
        echo "Please install it first:"
        echo "  cargo install cargo-llvm-cov"
        exit 1
    fi
}

# Run a single test example under instrumentation
# Arguments: $1 = test name, $2 = number of processes (optional, defaults to NP)
run_test() {
    local name="$1"
    local procs="${2:-$NP}"
    local binary="./target/debug/examples/${name}"

    if [ ! -f "$binary" ]; then
        echo -e "  ${YELLOW}SKIP${RESET}  ${name} (binary not found)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    echo -n -e "  Running ${BOLD}${name}${RESET} (n=${procs})... "

    local output
    local exit_code=0
    # shellcheck disable=SC2086
    output=$("$MPIEXEC" $MPIEXEC_ARGS -n "$procs" "$binary" 2>&1) || exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}PASS${RESET}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}FAIL${RESET} (exit code: ${exit_code})"
        FAILED=$((FAILED + 1))
        FAILED_TESTS+=("$name")
        # Print output for failed tests to aid debugging
        echo "    --- output ---"
        echo "$output" | sed 's/^/    /'
        echo "    --- end ---"
    fi
}

# ========================================================================
# Main
# ========================================================================

print_header
check_prerequisites

# ========================================================================
# Step 1: Set up cargo-llvm-cov instrumentation environment
# ========================================================================
echo -e "${BOLD}Setting up coverage instrumentation...${RESET}"
source <(cargo llvm-cov show-env --export-prefix)
echo -e "${GREEN}Instrumentation environment configured.${RESET}"
echo ""

# ========================================================================
# Step 2: Clean previous coverage data
# ========================================================================
echo -e "${BOLD}Cleaning previous coverage data...${RESET}"
cargo llvm-cov clean --workspace
echo -e "${GREEN}Coverage data cleaned.${RESET}"
echo ""

# ========================================================================
# Step 3: Build instrumented examples
# ========================================================================
echo -e "${BOLD}Building instrumented examples...${RESET}"
BUILD_ARGS=(--examples)
if [ -n "$FEATURES" ]; then
    BUILD_ARGS+=(--features "$FEATURES")
fi
if ! cargo build "${BUILD_ARGS[@]}" 2>&1; then
    echo -e "${RED}ERROR: Build failed${RESET}"
    exit 1
fi
echo -e "${GREEN}Build successful!${RESET}"
echo ""

# ========================================================================
# Step 4: Run unit tests under instrumentation
# ========================================================================
echo -e "${BOLD}Running unit tests under instrumentation...${RESET}"
TEST_ARGS=(--lib)
if [ -n "$FEATURES" ]; then
    TEST_ARGS+=(--features "$FEATURES")
fi
if ! cargo test "${TEST_ARGS[@]}" 2>&1; then
    echo -e "${RED}ERROR: Unit tests failed${RESET}"
    exit 1
fi
echo -e "${GREEN}Unit tests passed!${RESET}"
echo ""

# ========================================================================
# Step 5: Run MPI test examples under instrumentation
# ========================================================================
echo -e "${BOLD}════════════════════════════════════════${RESET}"
echo -e "${BOLD}Running MPI integration tests (instrumented)${RESET}"
echo -e "${BOLD}════════════════════════════════════════${RESET}"

# Core test examples
run_test test_lifecycle 2
run_test test_collectives
run_test test_blocking_extra
run_test test_nonblocking
run_test test_p2p_extra
run_test test_nonblocking_collectives
run_test test_persistent
run_test test_comm_split 4

# RMA / shared memory tests (only if rma or numa feature is enabled)
if [[ "$FEATURES" == *"rma"* ]] || [[ "$FEATURES" == *"numa"* ]]; then
    echo ""
    echo -e "${BOLD}────────────────────────────────────────${RESET}"
    echo -e "${BOLD}Running RMA/shared memory tests${RESET}"
    echo -e "${BOLD}────────────────────────────────────────${RESET}"
    run_test shared_memory
fi

# ========================================================================
# Step 6: Generate coverage reports
# ========================================================================
echo ""
echo -e "${BOLD}════════════════════════════════════════${RESET}"
echo -e "${BOLD}Generating coverage reports${RESET}"
echo -e "${BOLD}════════════════════════════════════════${RESET}"

echo -e "  Generating LCOV report..."
cargo llvm-cov report --lcov --output-path lcov.info
echo -e "  ${GREEN}LCOV report written to lcov.info${RESET}"

echo -e "  Generating HTML report..."
cargo llvm-cov report --html
echo -e "  ${GREEN}HTML report written to target/llvm-cov/html/index.html${RESET}"

echo ""
echo -e "${BOLD}Coverage summary:${RESET}"
cargo llvm-cov report

# ========================================================================
# Summary
# ========================================================================
echo ""
echo -e "${BOLD}╔════════════════════════════════════════════════════╗${RESET}"
TOTAL=$((PASSED + FAILED + SKIPPED))
if [ "$FAILED" -eq 0 ]; then
    echo -e "${BOLD}║${RESET}  ${GREEN}Results: ${PASSED} passed, ${FAILED} failed, ${SKIPPED} skipped (${TOTAL} total)${RESET}"
    echo -e "${BOLD}║${RESET}  ${GREEN}All tests passed!${RESET}"
else
    echo -e "${BOLD}║${RESET}  ${RED}Results: ${PASSED} passed, ${FAILED} failed, ${SKIPPED} skipped (${TOTAL} total)${RESET}"
    echo -e "${BOLD}║${RESET}  ${RED}Failed tests: ${FAILED_TESTS[*]}${RESET}"
fi
echo -e "${BOLD}╚════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  Coverage reports:"
echo -e "    LCOV: ${CYAN}lcov.info${RESET}"
echo -e "    HTML: ${CYAN}target/llvm-cov/html/index.html${RESET}"

exit "$FAILED"
