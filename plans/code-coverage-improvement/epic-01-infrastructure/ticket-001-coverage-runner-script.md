# Ticket 001: Create MPI Coverage Runner Script

## Summary

Create a shell script that runs MPI integration test examples under `cargo llvm-cov` instrumentation, collecting coverage data from all example binaries and producing a merged report.

## Dependencies

- None (first ticket)

## Acceptance Criteria

- [ ] Script `tests/run_mpi_coverage.sh` exists and is executable
- [ ] Script builds instrumented example binaries using `cargo llvm-cov` tooling
- [ ] Script runs each test example with `mpiexec` under instrumentation
- [ ] Script produces both LCOV and HTML coverage reports
- [ ] Script supports feature flags (rma, numa) as arguments
- [ ] Script supports configurable process count via `MPI_NP` env var
- [ ] Running the script shows coverage from FFI paths (not just unit tests)
- [ ] Script returns nonzero if any test example fails

## Files to Create/Modify

- **Create**: `tests/run_mpi_coverage.sh`
- **Modify**: (none)

## Technical Details

### Step 1: Research `cargo llvm-cov` usage for examples

`cargo llvm-cov` can instrument example binaries with:

```bash
# Build instrumented binaries
cargo llvm-cov run --no-report --example <name>
```

However, since MPI programs must be launched with `mpiexec`, we need an alternative approach. The `cargo llvm-cov` tool sets environment variables (`LLVM_PROFILE_FILE`, `CARGO_LLVM_COV`, etc.) to control instrumentation. We can:

1. Use `cargo llvm-cov show-env --export-prefix` to get the environment variables
2. Build the instrumented binaries with those env vars set
3. Run the binaries with `mpiexec` (the instrumentation is baked into the binary)
4. Merge the profile data and generate the report

### Step 2: Script structure

```bash
#!/bin/bash
set -euo pipefail

# Configuration
FEATURES="${1:-}"
NP="${MPI_NP:-4}"
MPIEXEC="${MPIEXEC:-mpiexec}"

# Step 1: Get instrumentation environment from cargo-llvm-cov
source <(cargo llvm-cov show-env --export-prefix)

# Step 2: Clean previous coverage data
cargo llvm-cov clean --workspace

# Step 3: Build instrumented examples (and run unit tests for baseline coverage)
CARGO_BUILD_ARGS="--examples"
if [ -n "$FEATURES" ]; then
    CARGO_BUILD_ARGS="$CARGO_BUILD_ARGS --features $FEATURES"
fi
cargo build $CARGO_BUILD_ARGS

# Step 4: Also run unit tests under instrumentation to capture those paths too
cargo test --lib

# Step 5: Run each test example under mpiexec
for example in test_collectives test_nonblocking test_comm_split test_lifecycle test_info_request test_persistent test_blocking_extra test_nonblocking_extra; do
    binary="./target/debug/examples/$example"
    if [ -f "$binary" ]; then
        echo "Running $example with $NP processes..."
        $MPIEXEC -n $NP $binary || { echo "FAIL: $example"; exit 1; }
    fi
done

# Step 6: Generate reports
cargo llvm-cov report --lcov --output-path lcov.info
cargo llvm-cov report --html
cargo llvm-cov report  # Print summary to stdout

echo "Coverage reports generated:"
echo "  LCOV: lcov.info"
echo "  HTML: target/llvm-cov/html/index.html"
```

### Step 3: Handle OpenMPI oversubscribe

Same as `run_mpi_tests.sh`, detect OpenMPI and add `--oversubscribe`.

### Step 4: Test the script

```bash
chmod +x tests/run_mpi_coverage.sh
./tests/run_mpi_coverage.sh
# Verify it produces lcov.info and HTML report
# Verify coverage numbers include FFI paths
```

## Definition of Done

- Script runs successfully and produces merged coverage from unit tests + MPI integration tests
- Coverage report shows non-zero coverage for `lib.rs`, `comm.rs`, and other modules' FFI paths
- Script is documented with usage instructions in its header
