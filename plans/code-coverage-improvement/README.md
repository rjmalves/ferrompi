# Code Coverage Improvement Plan

**Goal**: Increase `ferrompi` line coverage from ~42% to >90%

**Strategy**: Create MPI integration test examples that run under `cargo llvm-cov` instrumentation via `mpiexec`, systematically covering all FFI code paths.

## Status Table

| Ticket | Epic           | Description                       | Status      | Dependencies |
| ------ | -------------- | --------------------------------- | ----------- | ------------ |
| 001    | Infrastructure | Coverage runner script            | `completed` | —            |
| 002    | Infrastructure | Verify baseline coverage          | `completed` | 001          |
| 003    | Lifecycle      | MPI lifecycle test example        | `completed` | 001          |
| 004    | Blocking       | Blocking collective extras        | `completed` | 001, 002     |
| 005    | Blocking       | Point-to-point coverage           | `completed` | 001, 002     |
| 006    | Blocking       | Buffer validation unit tests      | `completed` | 002          |
| 007    | Nonblocking    | Nonblocking collective operations | `completed` | 001          |
| 008    | Nonblocking    | Request coverage extras           | `completed` | 001, 002     |
| 009    | Persistent     | Persistent collective operations  | `completed` | 001          |
| 010    | Persistent     | Persistent request drop coverage  | `completed` | 009          |
| 011    | Comm Mgmt      | Communicator management coverage  | `completed` | 001, 002     |
| 012    | Support        | Error and Info module coverage    | `completed` | 001          |
| 013    | RMA            | SharedWindow coverage             | `completed` | 001          |
| 014    | RMA            | SLURM module coverage             | `completed` | 001, 013     |

## Dependency Graph

```
ticket-001 (Coverage runner script)
├── ticket-002 (Verify baseline)
│   ├── ticket-004 (Blocking collective extras)
│   ├── ticket-005 (Point-to-point coverage)
│   ├── ticket-006 (Buffer validation unit tests)
│   ├── ticket-008 (Request coverage extras)
│   └── ticket-011 (Comm management coverage)
├── ticket-003 (MPI lifecycle test)
├── ticket-007 (Nonblocking collectives)
├── ticket-009 (Persistent collectives)
│   └── ticket-010 (Persistent drop coverage)
├── ticket-012 (Error/Info coverage)
└── ticket-013 (RMA SharedWindow)
    └── ticket-014 (SLURM coverage)
```

## Recommended Execution Order

**Phase 1** — Infrastructure:

1. ticket-001: Coverage runner script
2. ticket-002: Verify baseline

**Phase 2** — Core coverage (can be parallelized): 3. ticket-003: MPI lifecycle 4. ticket-006: Buffer validation unit tests 5. ticket-007: Nonblocking collectives 6. ticket-009: Persistent collectives 7. ticket-012: Error/Info coverage

**Phase 3** — Gap-filling (depends on Phase 2 baseline): 8. ticket-004: Blocking collective extras 9. ticket-005: Point-to-point coverage 10. ticket-008: Request coverage extras 11. ticket-010: Persistent drop coverage 12. ticket-011: Comm management coverage

**Phase 4** — Feature-gated: 13. ticket-013: RMA SharedWindow 14. ticket-014: SLURM module

## Coverage Target by File

> **Baseline**: 64.84% total line coverage (measured with unit tests + MPI examples)

| File            | Baseline | Target | Primary Ticket(s)            |
| --------------- | -------- | ------ | ---------------------------- |
| `comm.rs`       | 56.38%   | >85%   | 004, 005, 006, 007, 009, 011 |
| `lib.rs`        | 69.30%   | >90%   | 003                          |
| `persistent.rs` | 65.12%   | >95%   | 009, 010                     |
| `info.rs`       | 70.34%   | >95%   | 012                          |
| `error.rs`      | 86.31%   | >95%   | 012                          |
| `request.rs`    | 92.50%   | >95%   | 008                          |
| `datatype.rs`   | 100.00%  | 100%   | —                            |
| `status.rs`     | 100.00%  | 100%   | —                            |
| `window.rs`     | N/A      | >80%   | 013                          |
| `slurm.rs`      | N/A      | >80%   | 014                          |

## How to Run

```bash
# Run coverage with default features
./tests/run_mpi_coverage.sh

# Run coverage with RMA features
./tests/run_mpi_coverage.sh rma

# Run coverage with all features
./tests/run_mpi_coverage.sh numa

# Override process count
MPI_NP=8 ./tests/run_mpi_coverage.sh
```
