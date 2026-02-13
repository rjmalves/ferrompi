# Master Plan: Maximize Code Coverage for ferrompi

## Goal

Increase `ferrompi` code coverage from ~42.57% to >90% line coverage by creating MPI integration tests that run under `cargo llvm-cov` and systematically covering all uncovered code paths.

## Current State

> **Baseline measured**: 64.84% total line coverage (unit tests + MPI examples via `run_mpi_coverage.sh`)

| File            | Line Coverage | Lines Missed | Notes                                       |
| --------------- | ------------- | ------------ | ------------------------------------------- |
| `datatype.rs`   | 100.00%       | 0            | Complete                                    |
| `status.rs`     | 100.00%       | 0            | Complete                                    |
| `request.rs`    | 92.50%        | 6            | Nearly complete; minor FFI gaps             |
| `error.rs`      | 86.31%        | 33           | Missing `from_code()` FFI path              |
| `info.rs`       | 70.34%        | 35           | Missing FFI success paths                   |
| `lib.rs`        | 69.30%        | 35           | MPI lifecycle partially covered by examples |
| `persistent.rs` | 65.12%        | 45           | Missing FFI success paths                   |
| `comm.rs`       | 56.38%        | 591          | Biggest gap — many MPI ops still untested   |
| `window.rs`     | Not measured  | N/A          | Behind `rma` feature gate                   |
| `slurm.rs`      | Not measured  | N/A          | Behind `numa` feature gate                  |

## Uncovered Functions by Module (Baseline)

### `lib.rs` — 6 missed functions

- `Mpi::init`, `Mpi::init_thread`, `Mpi::world`, `Mpi::version`, `Mpi::wtime`
- `Mpi::is_initialized`, `Mpi::is_finalized`, `Mpi::thread_level`
- `Mpi::drop` (MPI_Finalize)

### `comm.rs` — 23 missed functions (generic instantiations excluded)

- **Point-to-point**: `send`, `recv`, `isend`, `irecv`, `sendrecv`, `probe`, `iprobe`
- **Blocking collectives**: `broadcast`, `reduce`, `gather`, `scatter`, `allgather`, `alltoall`, `scan`, `exscan`, `gatherv`, `scatterv`, `allgatherv`, `alltoallv`, `reduce_scatter_block`
- **Scalar/inplace variants**: `reduce_scalar`, `reduce_inplace`, `allreduce_scalar`, `allreduce_inplace`, `scan_scalar`, `exscan_scalar`
- **Nonblocking**: `ibroadcast`, `iallreduce`, `ireduce`, `igather`, `iallgather`, `iscatter`, `ialltoall`, `iscan`, `iexscan`, `igatherv`, `iscatterv`, `iallgatherv`, `ialltoallv`, `ireduce_scatter_block`, `ibarrier`
- **Persistent init**: `bcast_init`, `reduce_init`, `allreduce_init`, `allreduce_init_inplace`, `gather_init`, `scatter_init`, `allgather_init`, `alltoall_init`, `scan_init`, `exscan_init`, `gatherv_init`, `scatterv_init`, `allgatherv_init`, `alltoallv_init`, `reduce_scatter_block_init`
- **Comm management**: `rank`, `size`, `world`, `split`, `split_type`, `split_shared`, `duplicate`, `processor_name`, `barrier`, `abort`, `drop`

### `error.rs` — 1 missed function

- `Error::from_code` (FFI path from MPI error codes)

### `info.rs` — 2 missed functions

- `Info::new`, `Info::drop` (FFI paths for `MPI_Info_create`/`MPI_Info_free`)
- `Info::set`, `Info::get` inner FFI closures

### `persistent.rs` — 2 missed functions

- `PersistentRequest::wait` (FFI path)
- `PersistentRequest::drop` (FFI path for `MPI_Request_free`)

### `request.rs` — 0 missed functions (all executed, 6 lines missed)

## Core Technical Challenge

Most uncovered code calls FFI functions that require an MPI runtime. Standard `cargo test` cannot exercise these paths because MPI is not initialized. The existing integration tests (`examples/test_*.rs`) run via `mpiexec` but are example binaries — they don't get captured by `cargo llvm-cov --lib`.

## Architecture Decision: MPI Integration Test Harness

**Decision**: Create proper `tests/*.rs` integration tests that can be run under `cargo llvm-cov run` with `mpiexec`.

**Rationale**:

- `cargo llvm-cov run --example <name>` can instrument example binaries and capture their coverage into the library report
- This avoids needing to refactor the FFI layer or introduce mocks
- The tests already exist as examples; we need to either (a) run them with `cargo llvm-cov run`, or (b) create new comprehensive test binaries
- We'll create new test example binaries that systematically cover ALL methods, organized by module

**Coverage measurement approach**:

```bash
# Run all test examples under coverage and merge
cargo llvm-cov clean --workspace
cargo llvm-cov run --no-report --example test_collectives -- [mpiexec wrapper]
cargo llvm-cov run --no-report --example test_nonblocking
# ... etc
cargo llvm-cov report --lcov --output-path lcov.info

# OR: Create a shell script that runs mpiexec with cargo-llvm-cov-instrumented binaries
```

**Alternative considered**: A mock/trait-based FFI abstraction layer. Rejected because it would be a massive refactor changing the entire library API, and the FFI calls are the actual behavior we want to test.

## Epic Breakdown

### Epic 1: Coverage Infrastructure (2 tickets)

Set up tooling to measure coverage from MPI integration tests. Create a coverage runner script that uses `cargo llvm-cov` to instrument example binaries run under `mpiexec`.

### Epic 2: MPI Lifecycle Coverage — `lib.rs` (1 ticket)

Cover `Mpi::init()`, `init_thread()`, `Drop`, `version()`, `wtime()`, `is_initialized()`, `is_finalized()`.

### Epic 3: Blocking Collectives Coverage — `comm.rs` (3 tickets)

Cover all blocking collective operations: broadcast, reduce, allreduce, gather, scatter, allgather, alltoall, scan, exscan, gatherv, scatterv, allgatherv, alltoallv, reduce_scatter_block, and their scalar/inplace variants. Plus point-to-point: send, recv, sendrecv, probe, iprobe.

### Epic 4: Nonblocking Operations Coverage — `comm.rs` (2 tickets)

Cover all `i`-prefixed nonblocking operations: ibroadcast, iallreduce, ireduce, igather, iallgather, iscatter, ibarrier, iscan, iexscan, ialltoall, igatherv, iscatterv, iallgatherv, ialltoallv, ireduce_scatter_block.

### Epic 5: Persistent Operations Coverage — `comm.rs` + `persistent.rs` (2 tickets)

Cover all `_init` persistent operations and `PersistentRequest` lifecycle (start, wait, test, start_all, wait_all, drop).

### Epic 6: Communicator Management Coverage — `comm.rs` (1 ticket)

Cover `processor_name()`, `duplicate()`, `split()`, `split_type()`, `split_shared()`, `barrier()`.

### Epic 7: Support Module Coverage — `error.rs`, `info.rs`, `request.rs` (1 ticket)

Cover remaining FFI paths in `error.rs` (`from_code`), `info.rs` (new, set, get, drop), `request.rs` (wait, test, wait_all, drop).

### Epic 8: RMA Window Coverage — `window.rs` (2 tickets)

Cover `SharedWindow` operations with the `rma` feature flag: allocate, local/remote slices, fence, lock/lock_all, flush, drop.

## Execution Order

```
Epic 1 (Infrastructure)
  |
  +---> Epic 2 (lib.rs lifecycle)
  |       |
  |       +---> Epic 3 (blocking collectives) ---> Epic 4 (nonblocking) ---> Epic 5 (persistent)
  |       |
  |       +---> Epic 6 (comm management)
  |       |
  |       +---> Epic 7 (support modules)
  |
  +---> Epic 8 (RMA window) [can run in parallel with Epics 3-7]
```

## Success Criteria

- [ ] Line coverage > 90% as measured by `cargo llvm-cov`
- [ ] All MPI integration tests pass with `mpiexec -n 4`
- [ ] Coverage script runs in CI and produces an HTML/LCOV report
- [ ] No existing tests are broken
- [ ] Feature-gated code (`rma`, `numa`) is included in coverage measurement
