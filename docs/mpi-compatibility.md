# MPI Implementation Compatibility

> **Audience:** Users selecting an MPI implementation to deploy ferrompi, and
> contributors interpreting test failures against a non-CI-tested runtime.
>
> **Testing disclaimer:** Only two implementations are part of CI: MPICH
> (using the Ubuntu Noble package, hotfixed to 4.2.1) and Open MPI (from
> `libopenmpi-dev` on Ubuntu Noble, which ships Open MPI 4.x). All other
> implementations — Open MPI 5.x, Intel MPI, and Cray MPI — are **not CI-tested**.
> Their entries in the matrix below are based on known MPI standard version
> support and user reports. Do not assume that an unmarked cell indicates a
> problem; it indicates an absence of evidence. Where CI evidence exists, cells
> are marked ✓ or ⚠. Where no evidence exists, cells are marked ?.

---

## CI vs User-Reported Status

ferrompi's CI pipeline (`.github/workflows/test.yml`) runs on Ubuntu Noble
(`ubuntu-latest`) and installs MPI via `apt-get`. The matrix covers two
implementations:

| Implementation | CI status     | Version tested      | Install package              |
| -------------- | ------------- | ------------------- | ---------------------------- |
| MPICH          | CI-tested     | 4.2.1 (hotfixed)    | `mpich libmpich-dev`         |
| Open MPI       | CI-tested     | 4.x (Noble default) | `libopenmpi-dev openmpi-bin` |
| Open MPI 5.x   | user-reported | 5.0.0+              | —                            |
| Intel MPI      | user-reported | 2021.x / 2024.x     | —                            |
| Cray MPI / MPT | untested      | 8.x (MPICH-derived) | via `CRAY_MPICH_DIR`         |

The CI workflow installs MPICH, applies a hotfix for the Ubuntu Noble
`libmpich12` package (see the
[upstream bug report](https://bugs.launchpad.net/ubuntu/+source/mpich/+bug/2072338)),
and runs all integration tests under both the default feature set and the `rma`
feature set. Open MPI is also installed from the Noble archive but requires
environment overrides (`OMPI_MCA_rmaps_base_oversubscribe=1`,
`OMPI_ALLOW_RUN_AS_ROOT=1`) to run correctly in the GitHub Actions sandbox.

MPICH 3.x is **not tested in CI**. The Ubuntu Noble default provides MPICH
4.2.x; there is no matrix entry for an older MPICH. User reports from MPICH 3.x
deployments are the only signal. The same applies to Open MPI 5.x, Intel MPI,
and Cray MPI.

User-reported compatibility evidence is collected via GitHub Issues. When a
report is verified (a reproducible test run with version output attached), the
relevant cells in this document are updated and the issue is linked from the
corresponding footnote. See [How to Report Compatibility](#how-to-report-compatibility)
for what to include.

---

## Implementation Matrix

Column ordering is consistent across all sub-tables:
**MPICH 3.x | MPICH 4.x | Open MPI 4 | Open MPI 5 | Cray MPI**

Legend:

- ✓ — works; confirmed by CI pass or a verified user report
- ⚠ — works with a caveat; see the numbered footnote below the table
- ✗ — does not work; feature is absent or known to fail
- ? — untested; no CI evidence and no verified user report

### Blocking Collectives

These functions map to the `Communicator::barrier`, `broadcast`, `reduce`,
`allreduce`, `gather`, `allgather`, `scatter`, `alltoall`, `scan`, `exscan`,
and `reduce_scatter_block` methods in ferrompi's public API. All are MPI 1.0+
features and have been available in every MPI implementation for decades. The
MPICH 4.x ✓ cells reflect CI passes on MPICH 4.2.1.

| Feature                | MPICH 3.x | MPICH 4.x | Open MPI 4 | Open MPI 5 | Cray MPI |
| ---------------------- | :-------: | :-------: | :--------: | :--------: | :------: |
| `barrier`              |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `broadcast`            |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `reduce`               |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `allreduce`            |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `gather`               |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `allgather`            |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `scatter`              |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `alltoall`             |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `scan`                 |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `exscan`               |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `reduce_scatter_block` |     ✓     |     ✓     |     ✓      |     ?      |    ?     |

Open MPI 4 ✓ reflects CI passes on the Ubuntu Noble `libopenmpi-dev` package.

### Nonblocking Collectives

These are the `i*` variants (`ibarrier`, `ibcast`, `ireduce`, etc.). They are
MPI 3.0+ features. MPICH 3.x supports MPI 3.1 and provides these; Open MPI 3+
does as well. The ✓ entries for MPICH 3.x are based on MPI standard conformance
(MPICH 3.x implements MPI 3.1); CI evidence exists only for MPICH 4.x and Open
MPI 4.

| Feature                 | MPICH 3.x | MPICH 4.x | Open MPI 4 | Open MPI 5 | Cray MPI |
| ----------------------- | :-------: | :-------: | :--------: | :--------: | :------: |
| `ibarrier`              |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `ibcast`                |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `ireduce`               |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `iallreduce`            |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `igather`               |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `iallgather`            |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `iscatter`              |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `ialltoall`             |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `iscan`                 |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `iexscan`               |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `ireduce_scatter_block` |     ✓     |     ✓     |     ✓      |     ?      |    ?     |

### Persistent Collectives

Persistent collectives (`*_init` / `start` / `wait` pattern) require **MPI
4.0+**. MPICH 4.0 was the first MPICH release to support MPI 4.0. Open MPI
4.x implements MPI 3.1 only; Open MPI 5.0 is the first Open MPI release to
implement MPI 4.0. Consequently:

- MPICH 3.x: ✗ — the `#if MPI_VERSION >= 4` stub in `csrc/ferrompi.c` returns
  `MPI_ERR_UNSUPPORTED_OPERATION`; the Rust layer maps this to
  `Error::NotSupported`.
- Open MPI 4: ✗ — Open MPI 4.x implements MPI 3.1, not MPI 4.0. The MPI
  version number of the _implementation_ (4.x) does not equal the MPI
  _standard_ version (3.1). See note on version number confusion in
  [Open MPI 4 notes](#open-mpi-4).

| Feature                     | MPICH 3.x | MPICH 4.x | Open MPI 4 | Open MPI 5 | Cray MPI |
| --------------------------- | :-------: | :-------: | :--------: | :--------: | :------: |
| `barrier_init`              |     ✗     |     ✓     |     ✗      |     ?      |    ?     |
| `bcast_init`                |     ✗     |     ✓     |     ✗      |     ?      |    ?     |
| `reduce_init`               |     ✗     |     ✓     |     ✗      |     ?      |    ?     |
| `allreduce_init`            |     ✗     |     ✓     |     ✗      |     ?      |    ?     |
| `gather_init`               |     ✗     |     ✓     |     ✗      |     ?      |    ?     |
| `allgather_init`            |     ✗     |     ✓     |     ✗      |     ?      |    ?     |
| `scatter_init`              |     ✗     |  ⚠ [^1]   |     ✗      |     ?      |    ?     |
| `alltoall_init`             |     ✗     |     ✓     |     ✗      |     ?      |    ?     |
| `scan_init`                 |     ✗     |     ✓     |     ✗      |     ?      |    ?     |
| `exscan_init`               |     ✗     |     ✓     |     ✗      |     ?      |    ?     |
| `reduce_scatter_block_init` |     ✗     |     ✓     |     ✗      |     ?      |    ?     |

[^1]:
    `scatter_init` with `MPI_IN_PLACE` (`scatter_init_inplace`) triggers a
    deterministic deadlock in MPICH 4.2.x when called on the root rank.
    Reproduced on MPICH 4.2.0, 4.2.1 (Ubuntu Noble hotfix), and 4.2.3 (Fedora).
    Fixed in MPICH 4.3+. The `test_scatter_init_inplace` integration test is
    conditionally skipped on any MPICH version string containing `"4.2."`. The
    non-inplace `scatter_init` (distinct buffer send and receive) is unaffected.

### Persistent Point-to-Point

Persistent P2P functions (`send_init`, `recv_init`, `bsend_init`, `rsend_init`,
`ssend_init`) were introduced in MPI 1.1 but ferrompi gates them at
`#if MPI_VERSION >= 3` per project policy. No runtime version probe is needed;
the gate is compile-time only. All five are available on every MPI 3.x and 4.x
implementation.

| Feature      | MPICH 3.x | MPICH 4.x | Open MPI 4 | Open MPI 5 | Cray MPI |
| ------------ | :-------: | :-------: | :--------: | :--------: | :------: |
| `send_init`  |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `recv_init`  |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `bsend_init` |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `rsend_init` |     ✓     |  ✓ [^2]   |     ✓      |     ?      |    ?     |
| `ssend_init` |     ✓     |     ✓     |     ✓      |     ?      |    ?     |

[^2]:
    `rsend_init` requires strict ordering: the receiving rank must post and
    start its `recv_init`/`start` before a barrier, and only after that barrier
    may the sending rank start the ready-send request. Omitting the barrier
    produces undefined behavior on strict implementations (including MPICH 4.x).
    The `test_rsend_init` integration test enforces this ordering explicitly.

### RMA / Shared Memory Windows

RMA operations require the `rma` feature flag (`cargo build --features rma`).
`Win_allocate_shared` requires MPI 3.0+. All other RMA functions below are MPI
2.0+ (`MPI_Put`, `MPI_Get`, `MPI_Accumulate`) or MPI 3.0+ (`MPI_Rput`,
`MPI_Rget`, `MPI_Raccumulate`, `MPI_Get_accumulate`, `MPI_Fetch_and_op`,
`MPI_Compare_and_swap`, `MPI_Win_flush*`, `MPI_Win_sync`, `MPI_Win_lock_all`).
All are available on MPICH 3.x/4.x and Open MPI 4+.

| Feature               | MPICH 3.x | MPICH 4.x | Open MPI 4 | Open MPI 5 | Cray MPI |
| --------------------- | :-------: | :-------: | :--------: | :--------: | :------: |
| `win_fence`           |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `win_lock`            |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `win_lock_all`        |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `win_unlock`          |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `win_unlock_all`      |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `win_flush`           |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `win_flush_all`       |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `win_flush_local`     |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `win_flush_local_all` |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `win_sync`            |     ✓     |  ⚠ [^3]   |     ✓      |     ?      |    ?     |
| `put`                 |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `get`                 |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `accumulate`          |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `get_accumulate`      |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `fetch_and_op`        |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `compare_and_swap`    |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `rput`                |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `rget`                |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `raccumulate`         |     ✓     |     ✓     |     ✓      |     ?      |    ?     |

[^3]:
    `MPI_Win_sync` is documented by the MPI standard as valid at any point
    within an epoch, but MPICH 4.2.x enforces a stricter interpretation and
    requires that `MPI_Win_sync` be called only within a passive-target epoch
    (inside a `lock`/`unlock` or `lock_all`/`unlock_all` pair). ferrompi
    documents this in the `src/window.rs` rustdoc for `Win::sync`. For
    portability across all MPICH 4.x releases, always call `sync` via a
    `WinLockGuard` or `WinLockAllGuard` rather than as a bare inherent method.

### Groups and Communicators

| Feature                   | MPICH 3.x | MPICH 4.x | Open MPI 4 | Open MPI 5 | Cray MPI |
| ------------------------- | :-------: | :-------: | :--------: | :--------: | :------: |
| `Group::union`            |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `Group::intersection`     |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `Group::difference`       |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `Communicator::split`     |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `Communicator::duplicate` |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `create_from_group`       |  ✗ [^4]   |     ✓     |   ✗ [^4]   |     ?      |    ?     |

[^4]:
    `Mpi::create_from_group` calls `MPI_Comm_create_from_group`, which is
    a MPI 4.0+ function. On implementations that provide only MPI 3.x
    (`MPI_VERSION < 4`), the C shim at `csrc/ferrompi.c:651-668` returns
    `MPI_ERR_UNSUPPORTED_OPERATION` via a compile-time `#if MPI_VERSION >= 4`
    stub. The Rust layer maps this to `Error::NotSupported`. A runtime version
    probe via `OnceLock<bool>` in `src/lib.rs` prevents the call from reaching
    the C layer at all on older runtimes.

### User-Defined Reduction Operations

`UserOp` uses the `MPI_Op_create` trampoline mechanism, which is MPI 1.0+
and available everywhere. The `_noncommutative` variant uses `commute=0` in
`MPI_Op_create`; this is standard MPI and supported by all implementations.
Only `allreduce_with_op` is currently exposed; the full `_with_op` sweep
(reduce, scan, exscan) is deferred to a follow-up.

| Feature                      | MPICH 3.x | MPICH 4.x | Open MPI 4 | Open MPI 5 | Cray MPI |
| ---------------------------- | :-------: | :-------: | :--------: | :--------: | :------: |
| `UserOp::new`                |     ✓     |     ✓     |     ✓      |     ?      |    ?     |
| `UserOp::new_noncommutative` |     ✓     |     ✓     |     ✓      |     ?      |    ?     |

### Large-Count `_c` Variants

All collective `_c` variants (e.g., `MPI_Allreduce_c`, `MPI_Bcast_c`) require
**MPI 4.0+**. The `#if MPI_VERSION >= 4` gate in `csrc/ferrompi.c` covers
every nonblocking collective shim. Persistent shims deliberately omit the `_c`
branch (standing epic-04 convention: persistent shims are MPI 4.0+ themselves,
and large-count interaction with persistent initialisers is deferred to MPI 5.x
clarification).

| Feature                                   | MPICH 3.x | MPICH 4.x | Open MPI 4 | Open MPI 5 | Cray MPI |
| ----------------------------------------- | :-------: | :-------: | :--------: | :--------: | :------: |
| All collective `_c` variants (>2³¹ elems) |     ✗     |     ✓     |     ✗      |     ?      |    ?     |

---

## Implementation-Specific Notes

### MPICH 4.x

MPICH 4.x is the primary CI-tested implementation. ferrompi is developed and
tested against MPICH 4.2.1 (the Ubuntu Noble hotfix package).

**Known issues on MPICH 4.2.x:**

- **`scatter_init_inplace` deadlock.** `MPI_Scatter_init` with `MPI_IN_PLACE`
  on the root rank produces a deterministic deadlock on all MPICH 4.2.x
  releases (4.2.0, 4.2.1, 4.2.3). The fix is present in MPICH 4.3+. ferrompi's
  `test_scatter_init_inplace` example detects the affected versions at runtime
  and prints `SKIP: MPICH 4.2.x has a known MPI_Scatter_init+MPI_IN_PLACE
deadlock`. The non-inplace `scatter_init` (separate send and receive buffers)
  is unaffected.

- **`MPI_Win_sync` passive-epoch requirement.** MPICH 4.2.x enforces that
  `MPI_Win_sync` must be called within a passive-target epoch (inside a lock
  pair), even though the MPI standard does not impose this restriction in all
  contexts. Always call `Win::sync` via a lock guard. This is documented in the
  `src/window.rs` rustdoc and captured in footnote [^3] above.

- **Ubuntu Noble MPICH package.** The `mpich` and `libmpich12` packages in
  Ubuntu 24.04 LTS (Noble) ship a broken MPICH 4.2.0. CI applies a hotfix that
  replaces the shared library with MPICH 4.2.1 from the Ubuntu archive. See the
  `Hotfix MPICH on Ubuntu Noble` step in `.github/workflows/test.yml` and the
  [upstream Launchpad bug](https://bugs.launchpad.net/ubuntu/+source/mpich/+bug/2072338).

**`MPI_UNDEFINED` value.** MPICH uses `MPI_UNDEFINED = -32766`; ferrompi
normalises this to `-1` in the C shim layer (`csrc/ferrompi.c`) so callers
always see `-1`. Do not hardcode either value in application code; use
`Group::undefined()` (which calls `ferrompi_mpi_undefined()`) to obtain the
sentinel.

**`MPI_MODE_*` constants.** Window fence and PSCW assert constants
(`MPI_MODE_NOSTORE`, `MPI_MODE_NOPUT`, etc.) are implementation-defined
integers. ferrompi queries them at runtime via `OnceLock`-cached C shims and
exposes them through the `WinFenceAssert` and `WinPscwAssert` bitflag types.
Never hardcode mode constant values.

### MPICH 3.x

MPICH 3.x implements MPI 3.1. It is **not tested in CI**; support is based on
known MPI standard conformance. The following MPI 4.0+ features are absent:

- All persistent collectives (`barrier_init`, `bcast_init`, etc.)
- `create_from_group` (`MPI_Comm_create_from_group`)
- All large-count `_c` variants (`MPI_Allreduce_c`, etc.)

The C shim layer in `csrc/ferrompi.c` provides `#if MPI_VERSION >= 4` stubs
for every MPI 4.0+ entry point. These stubs return
`MPI_ERR_UNSUPPORTED_OPERATION` unconditionally when compiled against
MPICH 3.x headers. Rust code maps this to `Error::NotSupported` and integration
tests use the `SKIP` pattern: check `Mpi::version()` at runtime, print
`"SKIP: ..."`, and return `0`.

All blocking collectives, nonblocking collectives, persistent P2P, and
group/communicator operations (excluding `create_from_group`) are expected to
work on MPICH 3.4.x.

### Open MPI 4

> **Important version number note.** "Open MPI 4" refers to the _implementation_
> version (the release number on the Open MPI website). Open MPI 4.x implements
> the MPI **3.1** standard, not MPI 4.0. Do not confuse the Open MPI release
> version with the MPI standard version.

Open MPI 4.x is CI-tested via the Ubuntu Noble `libopenmpi-dev` package. All
blocking collectives, nonblocking collectives, persistent P2P, RMA operations,
group operations, and user-defined ops are expected to work.

**Absent features (MPI 3.1 only):**

- Persistent collectives (require MPI 4.0+)
- `create_from_group` (requires MPI 4.0+)
- Large-count `_c` variants (require MPI 4.0+)

**Known CI caveats.** Open MPI requires oversubscription and root-permission
flags when running in GitHub Actions:
`OMPI_MCA_rmaps_base_oversubscribe=1`, `OMPI_ALLOW_RUN_AS_ROOT=1`,
`OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`. These are set in `.github/workflows/test.yml`
and are not required outside the CI sandbox.

**`MPI_UNDEFINED` value.** Open MPI uses `-1` for `MPI_UNDEFINED`, matching
ferrompi's normalised value. The normalisation in the C shim layer is still
applied for defensive correctness.

**`Request::cancel` portability.** `MPI_Cancel` on a send request is silently
ignored by Open MPI (the cancel is accepted but the send completes normally).
This is standard-conformant. The `cancel` method in ferrompi rustdoc documents
this explicitly.

### Open MPI 5

Open MPI 5.0 is the first Open MPI release to implement MPI 4.0. It is
**not CI-tested** in ferrompi; support is based on user reports.

With Open MPI 5.x, the following features expected to be available (in addition
to everything in Open MPI 4):

- Persistent collectives (`bcast_init`, `allreduce_init`, etc.)
- `create_from_group`
- Large-count `_c` variants

The `scatter_init_inplace` MPICH 4.2.x deadlock is absent on Open MPI 5.x (the
`test_scatter_init_inplace` integration test documents this explicitly).

If you run ferrompi on Open MPI 5.x, please report your experience via a GitHub
Issue. See [How to Report Compatibility](#how-to-report-compatibility).

### Intel MPI

Intel MPI is **not CI-tested**. Intel MPI is based on MPICH and tracks MPICH
releases closely. Intel MPI 2021.x is built on MPICH 3.4; Intel MPI 2024.x is
built on MPICH 4.x.

**Expected compatibility:**

- Intel MPI 2021.x: equivalent to MPICH 3.4. MPI 4.0 features unavailable.
- Intel MPI 2024.x: equivalent to MPICH 4.x. All MPI 4.0 features expected to
  work. The MPICH 4.2.x `scatter_init_inplace` deadlock may also affect Intel
  MPI 2024 releases that incorporate MPICH 4.2.x; check `mpiexec --version`
  for the embedded MPICH version string.

To build against Intel MPI, set `MPICC=/opt/intel/oneapi/mpi/latest/bin/mpicc`
and either set `MPI_PKG_CONFIG` or ensure that Intel MPI's `pkg-config` path
is on `PKG_CONFIG_PATH`.

### Cray MPI / MPT

Cray MPI (also referred to as Cray MPT or MPICH for Cray) is **untested**.
It is a custom MPICH derivative maintained by HPE/Cray for Cray XC, XE, and EX
systems. Compatibility with ferrompi is expected to be similar to the
underlying MPICH release, but Cray-specific compiler and linker requirements
create additional configuration complexity:

- The `CRAY_MPICH_DIR` environment variable must be set to the Cray MPI
  installation directory (e.g., `/opt/cray/pe/mpich/8.1.25`). ferrompi's
  `build.rs` checks this variable before falling back to pkg-config.
- Cray systems typically use the Cray compiler wrapper (`cc`) rather than
  `mpicc`. Set `MPICC=$(which cc)` or the appropriate Cray wrapper.
- pthread linkage may differ from upstream MPICH. The handle-table ADR
  (`docs/adr/0002-handle-tables.md`, Decision Driver 5) notes Cray-specific
  pthread linkage as a known concern.
- The Cray ICC backend may emit warnings that ferrompi's CI does not test
  (`-D warnings` is enforced in CI under GCC/Clang only).

If you successfully build and test ferrompi on a Cray system, please report
your experience including the `CRAY_MPICH_DIR` path and the embedded MPICH
version number.

---

## Required Environment Variables and Build Flags

ferrompi's `build.rs` detects the MPI installation through a probe chain.
The table below summarises the three environment variables that override the
auto-detection, plus the feature flags that gate optional functionality.

| Variable / Flag   | Type          | Effect                                              | When Required                                                    |
| ----------------- | ------------- | --------------------------------------------------- | ---------------------------------------------------------------- |
| `MPI_PKG_CONFIG`  | env var       | Sets the `pkg-config` service name to query         | When pkg-config returns the wrong MPI (e.g., multiple installed) |
| `MPICC`           | env var       | Sets the MPI compiler wrapper path                  | When `mpicc` is not on `$PATH` or the wrong wrapper is found     |
| `CRAY_MPICH_DIR`  | env var       | Points `build.rs` to the Cray MPI installation root | Required on all Cray PE systems                                  |
| `--features rma`  | Cargo feature | Enables RMA and shared-memory window API            | Optional; adds `Win`, `WinLockGuard`, and all RMA methods        |
| `--features numa` | Cargo feature | Enables NUMA-aware allocation (implies `rma`)       | Optional; requires `libhwloc-dev` at build time                  |

**Which implementations need explicit overrides:**

- **MPICH** (non-system install): set `MPICC=/path/to/mpicc` and
  `MPI_PKG_CONFIG=mpich`.
- **Open MPI**: if both MPICH and Open MPI are installed, set
  `MPI_PKG_CONFIG=ompi` to select Open MPI.
- **Intel MPI**: set `MPICC=/opt/intel/oneapi/mpi/latest/bin/mpicc`. The
  `intel-mpi` pkg-config name is not universally available; using `MPICC` is
  more reliable.
- **Cray MPI**: set `CRAY_MPICH_DIR`. All other detection is skipped when this
  variable is present.
- **Ubuntu system MPICH**: auto-detected via `pkg-config mpich`. No override
  needed if `mpich libmpich-dev` are the only MPI packages installed.

---

## How to Report Compatibility

Compatibility reports from non-CI implementations are the primary mechanism for
improving this document. When filing a GitHub Issue, include the following
checklist so the report can be verified and the matrix updated:

```text
## Compatibility Report

- ferrompi version: (output of `grep '^version' Cargo.toml`)
- MPI implementation and version: (output of `mpiexec --version`)
- OS and distribution: (e.g., Ubuntu 22.04, RHEL 9.3, macOS 14.4)
- Compiler: (output of `$(MPICC) --version` or `mpicc --version`)
- Cargo features tested: (e.g., `default`, `rma`, `numa`)
- Test command: (e.g., `MPI_NP=4 ./tests/run_mpi_tests.sh rma`)
- Test results: (pass/fail counts; paste the summary line)
- Failing examples (if any): (names of failing `run_test` entries)
- Any environment variable overrides used: (MPI_PKG_CONFIG, MPICC, etc.)
```

Once a report is verified (reproducible by a maintainer or confirmed by a
second independent report), the relevant cells in the matrix above are updated
from ? to ✓ or ⚠, and the issue number is linked from the corresponding
footnote. Reports that do not include the `mpiexec --version` output cannot be
verified and will not be used to update the matrix.
