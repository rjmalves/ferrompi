# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-04-24

### Breaking Changes

- **`Error::Mpi` gains an `operation` field.** A new field
  `operation: Option<&'static str>` is appended to the `Error::Mpi`
  variant. Consumers pattern-matching `Error::Mpi { class, code, message }`
  without a trailing `..` will fail to compile. Recommended migrations:
  `Error::Mpi { class, code, message, .. }` if the operation tag is
  not needed, or `Error::Mpi { class, code, message, operation }` to
  read it. This bundles the breaking-change boundary that motivates
  the v0.4.0 minor bump.
- **`Display` format change.** When `operation.is_some()`, the error
  message now reads `"MPI error in {op}: ..."` (e.g.
  `"MPI error in bcast: invalid rank (class=ERR_RANK, code=6)"`)
  instead of `"MPI error: ..."`. Consumers parsing the human-readable
  `Display` output (not recommended, but extant) must account for the
  new prefix.

### Added

#### Groups & Custom Datatypes (epic-06)

- **`Group`** -- RAII handle to an `MPI_Group` with automatic free on drop.
  Obtain from `Communicator::group()` or by calling the set-operation
  methods below.
- **`GroupComparison`** -- `#[repr(i32)]` enum with ferrompi-stable
  discriminants (`Identical = 0`, `Similar = 1`, `Unequal = 2`)
  returned by `Group::compare`.
- **`RankRange`** -- Plain `{ first, last, stride }` struct for compact
  specification of arithmetic rank progressions; used by
  `Group::range_include` and `Group::range_exclude`.
- **`Group::size()`** -- Returns the number of processes in the group.
- **`Group::rank()`** -- Returns the calling rank within the group, or
  `Group::undefined()` (`-1`) when not a member.
- **`Group::undefined()`** -- Returns the normalised `MPI_UNDEFINED`
  sentinel (`-1`) without requiring an active MPI session.
- **`Group::include(ranks)`** -- Creates a sub-group from the given rank
  indices.
- **`Group::exclude(ranks)`** -- Creates a sub-group omitting the given
  rank indices.
- **`Group::union(other)`** -- Set-union of two groups.
- **`Group::intersection(other)`** -- Set-intersection of two groups.
- **`Group::difference(other)`** -- Set-difference (`self` minus `other`).
- **`Group::range_include(ranges)`** -- Sub-group formed from arithmetic
  rank progressions.
- **`Group::range_exclude(ranges)`** -- Sub-group formed by removing
  arithmetic rank progressions.
- **`Group::compare(other)`** -- Structural comparison of two groups;
  returns `GroupComparison`.
- **`Group::translate_ranks(ranks, other)`** -- Translates ranks from
  this group's rank space into `other`'s; returns
  `Vec<Option<i32>>` (`None` for non-members).
- **`Communicator::group()`** -- Returns the `Group` associated with this
  communicator.
- **`Mpi::create_from_group(group, tag)`** -- Creates a new communicator
  from a `Group` using `MPI_Comm_create_from_group` (MPI 4.0+). Returns
  `Err(Error::NotSupported)` on older MPI runtimes detected at runtime
  via a cached `OnceLock<bool>` version probe.
- **`CustomDatatype`** -- RAII handle to a committed derived `MPI_Datatype`;
  always committed on construction, freed automatically on drop. Not
  `Clone`; share via `&CustomDatatype`.
- **`StructField`** -- Field descriptor (`blocklength`, `displacement`,
  `basetype`) used as input to `CustomDatatype::create_struct`.
- **`BytePermutable`** -- Sealed trait marking types whose byte
  representation may be reinterpreted by custom-datatype operations
  (in `src/datatype.rs`).
- **`CustomDatatype::contiguous(count, basetype)`** -- Wraps
  `MPI_Type_contiguous` + `MPI_Type_commit`; creates a block of `count`
  identical base elements.
- **`CustomDatatype::vector(count, blocklength, stride, basetype)`** --
  Wraps `MPI_Type_vector` + `MPI_Type_commit`; creates a strided
  block type.
- **`CustomDatatype::create_struct(fields)`** -- Wraps
  `MPI_Type_create_struct` + `MPI_Type_commit`; creates a
  heterogeneous struct type from a slice of `StructField` descriptors.
- **`CustomDatatype::resized(lb, extent)`** -- Wraps
  `MPI_Type_create_resized` + `MPI_Type_commit`; produces a new type
  with adjusted lower bound and extent, leaving the original intact.
- **`Communicator::send_custom` / `recv_custom` / `isend_custom` / `irecv_custom`** --
  Point-to-point methods that accept a `&CustomDatatype` handle for
  user-defined derived types.

Reference: plans/ferrompi-gap-closure/learnings/epic-06-summary.md

#### Full RMA + Op_create + Persistent P2P (epic-07)

- **`UserOp<T>`** -- Safe wrapper around a user-supplied
  `Fn(&[T], &mut [T]) + Send + Sync + 'static` closure registered with
  `MPI_Op_create`. Backed by 16 pre-compiled trampolines; at most 16
  `UserOp` instances may be live concurrently per process.
- **`UserOp::new(f)`** -- Creates a commutative user-defined reduction op.
- **`UserOp::new_noncommutative(f)`** -- Creates a non-commutative
  user-defined reduction op; MPI will not reorder operands.
- **`Communicator::allreduce_with_op(send, recv, op)`** -- Allreduce
  driven by a `&UserOp<T>`.
- **`Win<T>`** -- Full-featured RMA window (`Win<'a, T: MpiDatatype>`)
  owning either a `Created` or `Allocated` window kind; freed
  automatically on drop. Complements the pre-existing
  `SharedWindow<T>`.
- **`WinFenceAssert`** -- Bitflag type for active-target fence assertions
  (`MPI_MODE_NOSTORE`, `MPI_MODE_NOPUT`, etc.); mode values cached via
  `OnceLock` at first use.
- **`WinPscwAssert`** -- Bitflag type for PSCW (`post`/`start`/
  `complete`/`wait_exposure`) epoch assertions.
- **`LockType`** -- Enum (`Exclusive`, `Shared`) for passive-target
  `Win::lock`.
- **`PendingFetchResult<T>`** -- Wraps `MaybeUninit<T>`; produced by
  `fetch_and_op` and `compare_and_swap` and resolved to `T` after the
  epoch completes.
- **`Win::fence(assert)`** -- Active-target epoch delimiter.
- **`Win::post(group, assert)`** -- PSCW: starts an exposure epoch on
  the target side.
- **`Win::start(group, assert)`** -- PSCW: starts an access epoch on
  the origin side.
- **`Win::complete()`** -- PSCW: ends the origin access epoch.
- **`Win::wait_exposure()`** -- PSCW: waits for the exposure epoch to
  finish (`MPI_Win_wait`; named to avoid collision with `Request::wait`).
- **`Win::lock(lock_type, rank)`** -- Passive-target: acquires a lock on
  a remote window and returns a `WinLockGuard` RAII guard.
- **`Win::lock_all()`** -- Passive-target: locks all ranks and returns a
  `WinLockAllGuard` RAII guard.
- **`Win::flush_local(rank)`** -- Completes locally-initiated RMA
  operations to `rank` without synchronising the remote side.
- **`Win::flush_local_all()`** -- `flush_local` across all ranks.
- **`Win::sync()`** -- Memory synchronisation fence for passive-target
  epochs.
- **`Win::put(origin, target_rank, target_disp)`** -- Blocking RMA put.
- **`Win::get(target_rank, target_disp, count)`** -- Blocking RMA get.
- **`Win::accumulate(origin, op, target_rank, target_disp)`** -- RMA
  accumulate with a predefined `ReduceOp`.
- **`Win::get_accumulate(origin, op, target_rank, target_disp)`** -- RMA
  fetch-and-accumulate; atomic on types implementing `AtomicMpiDatatype`.
- **`Win::fetch_and_op(value, op, target_rank, target_disp)`** -- Atomic
  fetch-and-op on a single element; returns `PendingFetchResult<T>`.
- **`Win::compare_and_swap(compare, value, target_rank, target_disp)`** --
  Atomic compare-and-swap; restricted to `AtomicMpiDatatype` (i32, i64,
  u32, u64, u8 — no f32/f64).
- **`Win::rput(origin, target_rank, target_disp)`** -- Non-blocking RMA
  put; returns a `Request`.
- **`Win::rget(target_rank, target_disp, count)`** -- Non-blocking RMA
  get; returns a `Request`.
- **`Win::raccumulate(origin, op, target_rank, target_disp)`** -- Non-blocking
  RMA accumulate; returns a `Request`.
- **`WinLockGuard`** -- RAII passive-target lock guard on a single rank;
  exposes `flush()` (flushes the locked rank captured at `Win::lock`).
- **`WinLockAllGuard`** -- RAII passive-target lock-all guard; exposes
  `flush(rank)` and `flush_all()`.
- **`AtomicMpiDatatype`** -- Sealed trait restricting atomic RMA
  operations (`compare_and_swap`, `get_accumulate`) to safe integer
  types: i32, i64, u32, u64, u8. f32 and f64 are excluded.
- **`Mpi::buffer_attach(buffer)`** -- Registers a `Box<[u8]>` with MPI
  for buffered-send mode; returns `Error::InvalidOp` on double-attach.
- **`Mpi::buffer_detach()`** -- Unregisters the attached buffer and
  returns ownership of the `Box<[u8]>`; blocks until all buffered sends
  complete.
- **`Communicator::send_init(buf, dest, tag)`** -- Creates a persistent
  send request (standard mode); returns `Result<PersistentRequest>`.
- **`Communicator::bsend_init(buf, dest, tag)`** -- Persistent buffered
  send request; returns `Result<PersistentRequest>`.
- **`Communicator::rsend_init(buf, dest, tag)`** -- Persistent ready-send
  request; returns `Result<PersistentRequest>`.
- **`Communicator::ssend_init(buf, dest, tag)`** -- Persistent
  synchronous-send request; returns `Result<PersistentRequest>`.
- **`Communicator::recv_init(buf, source, tag)`** -- Persistent receive
  request; returns `Result<PersistentRequest>`.

Reference: plans/ferrompi-gap-closure/learnings/epic-07-summary.md

#### Documentation (epic-08)

- **`docs/architecture.md`** -- Nine-section contributor reference
  (3 078 words): six-layer Mermaid architecture diagram, handle-table
  catalog for all seven tables, thread-safety model, C layer scope,
  FFI/ABI invariants, sealed-trait families, error handling model, and
  ADR cross-reference index.
- **`docs/migrating-from-rsmpi.md`** -- Function-for-function migration
  guide from rsmpi (3 238 words): 96-row pipe-table mapping rsmpi
  expressions to ferrompi equivalents, three side-by-side code samples,
  "Not supported" section, and a migration checklist.
- **`docs/mpi-compatibility.md`** -- Feature compatibility matrix
  (3 697 words): 82 five-column rows covering MPICH 3.x/4.x, Open MPI
  4/5, Intel MPI, and Cray MPI with footnoted MPICH 4.2.x quirks.
- **`docs/adr/0001-why-c-wrapper.md`** -- ADR justifying the hand-written
  C shim layer over `bindgen` (ABI portability) and Boost.MPI/MPL
  (dependency weight).
- **`docs/adr/0003-generic-mpi-datatype.md`** -- ADR documenting the
  four sealed-trait families (`MpiDatatype`, `AtomicMpiDatatype`,
  `MpiIndexedDatatype`, `BytePermutable`) and the hybrid
  `CustomDatatype` model for user types.
- **`docs/adr/0004-persistent-collective-approach.md`** -- ADR covering
  `PersistentRequest<'a, T>` lifetime enforcement and the unified design
  shared by persistent collectives and all five P2P `_init` variants.
- **`docs/adr/0005-mpi-op-create.md`** -- ADR covering the seven design
  decisions for the `MPI_Op_create` FFI-callback trampoline (closure
  storage, `Send+Sync+'static`, lifetime ordering, commutativity,
  dispatch, panic handling, datatype contract).
- **`docs/README.md`** -- Landing page for the `docs/` directory with
  GitHub-relative links and rustdoc/Mermaid rendering caveats.
- **`src/doc.rs`** -- Eight `#[doc = include_str!(...)]` modules that
  publish all long-form documentation files into rustdoc under
  `ferrompi::doc::*`.

Reference: plans/ferrompi-gap-closure/learnings/epic-08-summary.md

- **`Error::from_code_with_op(code, op)`** -- Constructs `Error::Mpi`
  with the operation tag pre-populated. Replaces the pattern of
  constructing `Error::from_code` and patching `operation` separately.
- **`Error::check_with_op(code, op)`** -- Mirror of `check` that
  propagates the operation tag on the error path.
- **`examples/test_error_context.rs`** -- Integration example that
  triggers an out-of-range broadcast root and asserts the new
  operation-tagged error format end-to-end.
- **`examples/test_request_table_concurrency.rs`** -- Multi-threaded
  isend/irecv stress test (4 threads × 100 iterations) that exercises
  the request table under `MPI_THREAD_MULTIPLE`.
- **`docs/adr/0002-handle-tables.md`** -- Architecture Decision Record
  explaining the C11-atomics-with-CAS strategy chosen for the request
  handle table and the rejected alternatives (pthread mutex, Treiber
  stack).

### Changed

- **All 101 internal FFI call sites now tag errors with the underlying
  C function name** (e.g., `"bcast"`, `"allreduce"`,
  `"allreduce_init"`, `"wait"`, `"isend"`). When an MPI call fails,
  the resulting `Error::Mpi.operation` is populated with the tag,
  giving downstream code (notably cobre) structured error context
  without needing to invent sentinel values.

### Fixed

- **Request handle table is now safe under `MPI_THREAD_MULTIPLE`.**
  Previously, concurrent `alloc_request` calls could observe the same
  `request_used[i] == 0` slot and both write `1`, with the second
  thread's `request_table[i]` write clobbering the first -- a silent
  lost-request data race that TSan would flag. The C wrapper now uses
  `atomic_compare_exchange_strong_explicit` on `request_used[i]` with
  `memory_order_acq_rel` semantics, paired with an
  `atomic_store_explicit(..., memory_order_release)` on free and an
  `atomic_load_explicit(..., memory_order_acquire)` on read. The
  comm/win/info tables retain their existing implementations and are
  scheduled for hardening in a later release.

## [0.3.0] - 2026-04-10

### Added

- **Topology reporting** -- New `Communicator::topology(&mpi)` collective that
  gathers rank-to-host mapping across all processes and returns a `TopologyInfo`
  struct. The `Display` implementation produces a human-readable report showing
  MPI library version, standard version, thread level, process distribution
  across nodes, and (with the `numa` feature) SLURM job metadata.
- **`Mpi::library_version()`** -- Returns the MPI implementation version string
  (e.g. "Open MPI v4.1.6") by wrapping `MPI_Get_library_version`.
- **`TopologyInfo`**, **`HostEntry`**, and **`SlurmInfo`** public types with
  accessors for programmatic inspection of job topology.
- **`topology` example** -- Demonstrates one-liner topology reporting and
  programmatic access to host/rank mapping.

### Fixed

- 34 findings from security/correctness assessment addressed.

## [0.2.2] - 2026-03-27

### Fixed

- **aarch64 compatibility** -- All `c_char` casts now use `std::ffi::c_char`
  instead of hardcoded `i8`. On aarch64 (ARM), `c_char` is `u8` (unsigned),
  while on x86_64 it is `i8` (signed). The previous `.cast::<i8>()` calls
  caused type mismatches on ARM targets. Affected: `get_version`,
  `get_processor_name`, `error_info`, `info_get`.

## [0.2.1] - 2026-03-27

### Fixed

- **Remove RPATH from linked binaries** -- Build script no longer embeds
  `-Wl,-rpath` with the build machine's library paths. Pre-built release
  binaries were failing on HPC clusters and containers where MPI is installed
  in a different path. Users must ensure `libmpi` is discoverable at runtime
  via `LD_LIBRARY_PATH`, `ldconfig`, or their cluster's module system.

### Changed

- **Repository URLs** -- Updated all URLs after migration to `cobre-rs` org.

## [0.2.0] - 2026-02-13

### Breaking Changes

- Removed all type-specific methods (`_f64`, `_i32`, etc.) in favor of generic `MpiDatatype` API
- `Communicator` is now `Send + Sync` for hybrid MPI+threads programs
- Error type restructured: `Error::MpiError(i32)` → `Error::Mpi { class, code, message }` with `MpiErrorClass` enum providing rich error categorization
- Removed `InvalidRank`, `InvalidCommunicator`, etc. (now covered by `MpiErrorClass` variants)

### Added

- Generic `MpiDatatype` trait for `f32`, `f64`, `i32`, `i64`, `u8`, `u32`, `u64`
- Communicator management: `split()`, `split_type()`, `split_shared()`, `duplicate()`
- MPI_Info object support with RAII lifecycle (`Info` type)
- Complete nonblocking point-to-point: `isend`, `irecv`, `sendrecv`
- Probe/Iprobe: `probe<T>`, `iprobe<T>` with `Status` struct (source, tag, count)
- Scalar reduce variants: `reduce_scalar`, `allreduce_scalar`
- In-place reduce variants: `reduce_inplace`, `allreduce_inplace`
- Scan/Exscan: `scan`, `exscan`, `scan_scalar`, `exscan_scalar` (blocking, nonblocking, persistent)
- V-collectives: `gatherv`, `scatterv`, `allgatherv`, `alltoallv` (blocking, nonblocking, persistent)
- Alltoall: `alltoall` (blocking, nonblocking, persistent)
- Reduce-scatter-block: `reduce_scatter_block` (blocking, nonblocking, persistent)
- All 15 nonblocking collective variants: `ibroadcast`, `iallreduce`, `ireduce`, `igather`, `iallgather`, `iscatter`, `ibarrier`, `iscan`, `iexscan`, `ialltoall`, `igatherv`, `iscatterv`, `iallgatherv`, `ialltoallv`, `ireduce_scatter_block`
- All 15 persistent collective variants (MPI 4.0+): `bcast_init`, `allreduce_init`, `allreduce_init_inplace`, `reduce_init`, `gather_init`, `scatter_init`, `allgather_init`, `scan_init`, `exscan_init`, `alltoall_init`, `gatherv_init`, `scatterv_init`, `allgatherv_init`, `alltoallv_init`, `reduce_scatter_block_init`
- Shared memory windows: `SharedWindow<T>` with RAII lock guards (`LockGuard`, `LockAllGuard`) (feature: `rma`)
- Window synchronization: `fence`, `lock`, `lock_all`, `flush`, `flush_all`
- SLURM environment helpers: `is_slurm_job`, `job_id`, `local_rank`, `local_size`, `num_nodes`, `cpus_per_task`, `node_name`, `node_list` (feature: `numa`)
- Comprehensive test suite: unit tests for `slurm` module, MPI integration test runner (`tests/run_mpi_tests.sh`)
- CI matrix: MPICH × OpenMPI × default/rma feature combinations
- New examples: `comm_split`, `scan`, `gatherv`, `shared_memory`, `hybrid_openmp`

### Changed

- C handle table limits expanded: 256 communicators, 16384 requests, 256 windows, 64 infos
- Rich error messages via `MPI_Error_class` + `MPI_Error_string`
- C wrapper layer significantly expanded (~2400 lines, up from ~700)

## [0.1.0] - 2026-01-13

### Added

- Initial release
- MPI 4.0+ support with persistent collectives
- Safe Rust API for MPI operations
- Examples: hello_world, ring, allreduce, nonblocking, persistent_bcast, pi_monte_carlo
- Comprehensive documentation
- Initial CI/CD setup with GitHub Actions

[Unreleased]: https://github.com/cobre-rs/ferrompi/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/cobre-rs/ferrompi/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cobre-rs/ferrompi/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/cobre-rs/ferrompi/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/cobre-rs/ferrompi/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cobre-rs/ferrompi/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cobre-rs/ferrompi/releases/tag/v0.1.0
