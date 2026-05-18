# ferrompi Architecture

## Overview

This document is the canonical reference for ferrompi's internal architecture.
Its audience is contributors investigating implementation details, not end users
seeking usage guidance (which lives in `README.md`). It explains why a
hand-written C wrapper exists, how handle tables are organised and made
thread-safe, which invariants must be preserved when adding new MPI entry
points, and how the sealed-trait type system enforces datatype safety across
the Rust/C boundary.

## Layer Diagram

ferrompi sits between application Rust code and an underlying MPI runtime
through a deliberate six-layer stack. Each layer has a clearly bounded
responsibility; crossing a layer boundary requires following the invariants
described in the sections below.

```mermaid
graph TD
    A["Rust application code\n(user crate, examples/)"]
    B["Public ferrompi API\n(src/lib.rs, src/comm/*, src/window.rs,\nsrc/group.rs, src/datatype_builder.rs, src/op.rs)"]
    C["FFI declarations\n(src/ffi.rs — extern \"C\" blocks)"]
    D["C wrapper layer\n(csrc/ferrompi.c — 4629 LOC\ncsrc/ferrompi.h — 2171 LOC)"]
    E["MPI implementation\n(MPICH / Open MPI / Cray MPT)"]
    F["MPI runtime\n(process manager, network fabric, RDMA HW)"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
```

| Layer                    | Responsibility                                                                                                 |
| ------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Rust application         | Calls ferrompi's public API; owns all live data.                                                               |
| Public ferrompi API      | Type safety, RAII drop semantics, `Result` error mapping, sealed-trait enforcement.                            |
| `src/ffi.rs`             | Thin `extern "C"` declarations — no logic, only signatures.                                                    |
| `csrc/ferrompi.c` / `.h` | Handle tables, large-count branching, `MPI_UNDEFINED` normalisation, op trampolines, runtime constant queries. |
| MPI implementation       | Collective algorithms, point-to-point transport, window coherence.                                             |
| MPI runtime              | Process launch, rank assignment, network fabric, memory registration.                                          |

The split between the C layer and the public Rust API is deliberate and
explained in full in `adr/0001-why-c-wrapper.md`. The short answer: several MPI
idioms (opaque handle tables, `_c` large-count branching, implementation-defined
constant values) require C11 constructs that cannot be expressed portably in
Rust FFI without replicating the C layer anyway.

## Handle Tables

The C layer in `csrc/ferrompi.c` owns seven fixed-size static tables that map
compact integer handles — passed over the FFI boundary as `int32_t` or
`int64_t` — to the opaque MPI handle types that the MPI library returns from
its constructor functions. MPI handles (`MPI_Comm`, `MPI_Request`, etc.) are
not stable integers; their values are implementation-defined and may change
across library versions. Storing them in a table and exposing only the table
index to Rust insulates the Rust layer from this instability.

| Table            | Capacity | Element type   | Occupancy tracking               |
| ---------------- | -------- | -------------- | -------------------------------- |
| `comm_table`     | 256      | `MPI_Comm`     | `MPI_COMM_NULL` sentinel         |
| `request_table`  | 16384    | `MPI_Request`  | `atomic_int request_used[16384]` |
| `win_table`      | 256      | `MPI_Win`      | `int win_used[256]`              |
| `info_table`     | 64       | `MPI_Info`     | `int info_used[64]`              |
| `group_table`    | 64       | `MPI_Group`    | `int group_used[64]`             |
| `datatype_table` | 64       | `MPI_Datatype` | `int datatype_used[64]`          |
| `op_table`       | 16       | `MPI_Op`       | `atomic_int op_used[16]`         |

Each table has a corresponding `alloc_*` / `free_*` / `get_*` function triple.
Allocation scans from a cached hint index and returns the first available slot;
freeing clears the occupancy marker and resets the handle to its null sentinel.
The hint is advisory (not correctness-critical) and advances after each
successful allocation to amortise scan cost.

### Request table — C11 atomic slot-claim

The `request_table` is by far the most active table: every non-blocking send,
receive, and collective operation allocates a slot. Under
`MPI_THREAD_MULTIPLE`, multiple threads may call `ferrompi_isend`,
`ferrompi_irecv`, and the wait/test family concurrently. A plain read-modify-
write on `request_used` would be a data race, allowing two threads to claim the
same slot silently.

`alloc_request` uses `atomic_compare_exchange_strong_explicit` with
`memory_order_acq_rel` on success: exactly one thread wins the CAS per slot per
attempt; the loser advances to the next index. `free_request` uses
`atomic_store_explicit` with `memory_order_release`; `get_request_ptr` uses
`atomic_load_explicit` with `memory_order_acquire`. This acquire/release pairing
ensures a thread that observes `request_used[i] != 0` also observes the
`request_table[i]` value written by the allocating thread. The full analysis,
including rejection of a pthread-mutex approach and a lock-free Treiber stack,
is in `adr/0002-handle-tables.md`.

The same C11 atomic pattern applies to `op_table` (parallel `atomic_int
op_used[MAX_OPS]` with the same CAS strategy), because user-defined ops may
also be created concurrently under `MPI_THREAD_MULTIPLE`.

### Slot 0 reservations and drop guards

`group_table[0]` is reserved for `MPI_GROUP_EMPTY` and is initialised once in
`init_tables`. The `Drop` implementation for `Group` guards `handle > 0` before
calling `ferrompi_group_free`, preventing a free of the reserved slot.

`datatype_table` has no reserved slot; `Drop` for `CustomDatatype` guards
`handle >= 0`. `op_table` slot allocation begins at index 0 with no reservation;
`Drop` for `UserOp` guards `op_handle >= 0` and calls `MPI_Op_free` before
clearing the slot (see "Thread-Safety Model" below for the ordering requirement).

### Open hardening work

`win_table`, `info_table`, and `comm_table` currently use plain `int used[]`
arrays with non-atomic read-modify-write. These have the same structural data
race under `MPI_THREAD_MULTIPLE` that was fixed in the request table. Hardening
them follows the ticket-023 pattern documented in `adr/0002-handle-tables.md`
and is tracked as an open issue in `plans/ferrompi-gap-closure/learnings/epic-07-summary.md`.

## Thread-Safety Model

ferrompi maps its `ThreadLevel` enum to MPI's four standard thread levels:

| `ThreadLevel` variant | MPI constant            | Meaning                                                     |
| --------------------- | ----------------------- | ----------------------------------------------------------- |
| `Single`              | `MPI_THREAD_SINGLE`     | Only one thread in the process.                             |
| `Funneled`            | `MPI_THREAD_FUNNELED`   | Only the thread that called `MPI_Init_thread` may call MPI. |
| `Serialized`          | `MPI_THREAD_SERIALIZED` | Multiple threads may call MPI, but not concurrently.        |
| `Multiple`            | `MPI_THREAD_MULTIPLE`   | Multiple threads may call MPI concurrently.                 |

`ThreadLevel` is selected at init time via `Mpi::init_thread(level)`. The
requested level is advisory; the MPI implementation may grant a lower level,
which ferrompi reflects in the returned `ThreadLevel`.

### `Communicator` is `Send + Sync`

`Communicator` implements both `Send` and `Sync`. The underlying `MPI_Comm`
handle is an integer index into `comm_table`; it carries no thread-affinity and
is valid to use from any thread, provided the MPI implementation was initialised
with at least `MPI_THREAD_FUNNELED`. Under `MPI_THREAD_MULTIPLE`, any thread
may call any `Communicator` method concurrently.

### Synchronisation in the C layer under `Multiple`

Under `MPI_THREAD_MULTIPLE`, the slot-claim CAS in `alloc_request` (and the
parallel CAS in `alloc_op`) is the **only** synchronisation primitive in the C
wrapper layer. All other correctness guarantees rely on the MPI implementation's
own thread-safety contract: if MPI reports `MPI_THREAD_MULTIPLE`, it guarantees
that concurrent calls to its own functions on the same communicator are safe.
ferrompi does not add mutexes, condition variables, or barriers beyond the
handle-table atomics.

### `UserOp` closure thread-safety contract

`UserOp<T>` wraps a user-supplied Rust closure that MPI will invoke during
reduction operations. The closure is stored in the C-side `op_closure_data` /
`op_closure_vtbl` fat-pointer pair and may be called from any thread under
`MPI_THREAD_MULTIPLE` — including an MPI-internal thread-pool thread that the
application did not create.

The closure must satisfy `F: Fn(&[T], &mut [T]) + Send + Sync + 'static`. All
three bounds are mandatory and enforced at compile time (see
`adr/0005-mpi-op-create.md` Decision 2):

- `Send` — the closure is moved into a global static table accessible from any thread.
- `Sync` — concurrent invocations on the same `MPI_Op` (e.g., two concurrent `allreduce` calls on different communicators) must not produce data races on the closure's captured state.
- `'static` — the closure is stored in a raw pointer in the C static segment; any borrow shorter than `'static` could be invalidated while the `MPI_Op` handle is still live.

Captures using `Rc<T>`, `Cell<T>`, `RefCell<T>`, raw pointers, or non-`'static`
references are rejected at compile time. Callers requiring shared mutable
closure state must use `Arc<Mutex<T>>` or `Arc<RwLock<T>>`.

### Drop ordering for `UserOp`

`Drop for UserOp<T>` calls `MPI_Op_free` first, then releases the closure slot.
This ordering is mandatory: after `MPI_Op_free` returns, the MPI library
guarantees it will never invoke the user function pointer again; only then may
the closure storage be freed. Reversing the order — dropping the closure before
freeing the op — is a use-after-free if any in-flight collective is still
dispatching the trampoline. The full rationale is in `adr/0005-mpi-op-create.md`
Decision 3.

## C Layer Scope

The C wrapper is intentionally narrow. The table below distinguishes what
belongs in C from what belongs in Rust.

### What goes in C (`csrc/ferrompi.c`)

- **Handle tables** — `comm_table`, `request_table`, `win_table`, `info_table`,
  `group_table`, `datatype_table`, `op_table` and their `alloc_*` / `free_*` /
  `get_*` triples. MPI opaque handles cannot be stored in Rust without copying
  the entire allocation strategy anyway.
- **Large-count branching** — shims for non-blocking and custom p2p operations
  inspect `MPI_VERSION` at compile time and dispatch to `MPI_*_c` variants
  (which accept `MPI_Count` rather than `int`) when `MPI_VERSION >= 4`. This
  branching requires C preprocessor guards that would be unreadable as inline
  Rust assembly or build-script code generation.
- **`MPI_UNDEFINED` normalisation** — MPI does not standardise the integer value
  of `MPI_UNDEFINED`. MPICH uses `-32766`; Open MPI uses `-1`. Every shim that
  may return `MPI_UNDEFINED` (e.g., `ferrompi_group_translate_ranks`) normalises
  it to `-1` before returning to Rust, so the Rust layer can use a single
  sentinel value.
- **`install_errors_return`** — called on every newly-created communicator handle
  to set the error handler to `MPI_ERRORS_RETURN`, converting MPI errors from
  process-aborting signals into return codes that ferrompi can translate to
  `Err(Error::Mpi { .. })`.
- **Op trampolines** — 16 distinct C functions `ferrompi_user_op_trampoline_0`
  through `ferrompi_user_op_trampoline_15`, generated by a preprocessor macro.
  Each bakes its slot index into the function body so that the closure pointer
  can be retrieved from `op_closure_data[N]` without a side channel. See
  `adr/0005-mpi-op-create.md` Decision 5 for why a single trampoline with
  thread-local dispatch is unsafe under `MPI_THREAD_MULTIPLE`.
- **Runtime implementation-defined constants** — `MPI_MODE_NOSTORE`,
  `MPI_MODE_NOPUT`, `MPI_MODE_NOPRECEDE`, `MPI_MODE_NOSUCCEED`, and the
  analogous PSCW assert constants are queried once via C shims and cached in
  Rust via `OnceLock<[i32; N]>`. Hardcoding them in Rust would be incorrect
  because their values are implementation-defined.

### What stays in Rust (`src/`)

- **Type safety** — the sealed-trait families `MpiDatatype`,
  `AtomicMpiDatatype`, `MpiIndexedDatatype`, and `BytePermutable` (see
  "Generic-over-`MpiDatatype` Design") ensure that only valid Rust types reach
  MPI entry points. The C layer accepts raw integers and cannot enforce this.
- **RAII drop semantics** — `Communicator`, `Request`, `PersistentRequest`,
  `Group`, `CustomDatatype`, `Win`, `UserOp`, and the RMA lock guards all
  implement `Drop`. Callers cannot forget to free handles; the Rust borrow
  checker enforces lifetime containment.
- **`Error` mapping** — `Error::check_with_op(ret, "<tag>")` converts every
  non-zero MPI return code into a structured `Error::Mpi { class, code,
message, operation }`. The `operation` tag is the C function name with the
  `ferrompi_` prefix stripped, enabling precise error attribution.
- **`OnceLock`-cached version probes** — MPI 4.0+ features (e.g.,
  `Mpi::create_from_group`) are gated by `OnceLock<bool>` probes that parse
  `Mpi::version()` at first call and cache the result. The probe lives in Rust
  because version queries are pure Rust logic; the underlying MPI version
  integer is available without FFI.
- **`catch_unwind + abort` panic fence** — the `rust_user_op_invoke`
  `extern "C"` entry point in `src/op.rs` wraps every closure invocation in
  `std::panic::catch_unwind`. If the closure panics, the process aborts
  immediately. Panicking across the FFI boundary is undefined behaviour; silent
  data corruption in a collective result is worse than a loud process abort for
  HPC use cases. See `adr/0005-mpi-op-create.md` Decision 6.
- **`WinLockGuard` / `WinLockAllGuard` RAII epoch guards** — passive-target
  RMA epochs (lock/unlock) are represented as RAII guards that carry `flush`
  and `flush_all` as inherent methods. This prevents calling `flush` outside a
  passive-target epoch, which is a misuse MPI does not check at runtime.

The rationale for this split — rather than writing pure Rust FFI without any C
intermediary — is documented in `adr/0001-why-c-wrapper.md`.

## FFI / ABI Invariants

The following invariants are established across the seven epics and must be
preserved by every new MPI entry point added to ferrompi.

### `#[repr(i32)]` enums with explicit discriminants

`ReduceOp` and `DatatypeTag` carry `#[repr(i32)]` with explicit `= N`
discriminants. These integers cross the FFI boundary raw (cast with `op as i32`
or `tag as i32`) and are decoded by `get_op()` and `get_datatype()` in
`csrc/ferrompi.c`. The discriminant values are a **semver contract**: renumbering
any variant silently corrupts any caller that stored the integer across a
version boundary. New variants must append at the next free integer; gaps are
forbidden.

### Unconditional C switch

The C switch that decodes op and datatype tags is always compiled
unconditionally, with no `#ifdef` guards. This is required because the Rust
layer may legally produce any tag value that the `ReduceOp` or `DatatypeTag`
enum can represent, including variants that are only reachable when specific
Cargo features (`rma`) are enabled. The C layer must accept all of them.

### Six-layer entry-point pattern

Every new MPI entry point follows this pattern in order:

1. C declaration in `csrc/ferrompi.h`.
2. C implementation in `csrc/ferrompi.c`.
3. `extern "C"` declaration in `src/ffi.rs`.
4. Safe Rust wrapper in the appropriate `src/` module.
5. Integration example in `examples/`.
6. `run_test` line in `tests/run_mpi_tests.sh`.

Omitting any layer is a scope violation. The test runner entry is particularly
easy to forget; it must be declared explicitly in each ticket's "Key Files to
Modify" list.

### `Error::check_with_op` at every call site

Every `ferrompi_*` FFI result must pass through
`Error::check_with_op(ret, "<tag>")`, where `<tag>` is the C function name with
the `ferrompi_` prefix stripped. Bare `Error::check(ret)` calls are forbidden in
production code and were eliminated in epic-05 (101 migration sites).

### `install_errors_return` on comm-creating shims

Every shim that creates a new communicator (or may return one from MPI) calls
`install_errors_return(newcomm)` before returning the handle to Rust. This sets
`MPI_ERRORS_RETURN` as the error handler, converting MPI library aborts into
return codes.

### Two-phase comm handle guard

Every shim that takes a `comm_handle` parameter applies a two-phase guard:
first `if (comm_handle < 0) return MPI_ERR_COMM;`, then
`if (comm == MPI_COMM_NULL) return MPI_ERR_COMM;` after resolving through
`comm_table`. This defends against both invalid indices and handles that were
freed but whose slot was not yet reclaimed.

### `MPI_UNDEFINED` normalised to `-1`

Every shim that may return `MPI_UNDEFINED` normalises it to `-1` before
returning to Rust. The Rust layer uses `-1` as the sentinel; `MPI_UNDEFINED`'s
value is implementation-defined (MPICH: `-32766`; Open MPI: `-1`).

### `MPI_MODE_*` constants queried at runtime

RMA fence and PSCW assert constants (`MPI_MODE_NOSTORE`, `MPI_MODE_NOPUT`,
`MPI_MODE_NOPRECEDE`, `MPI_MODE_NOSUCCEED`, and their PSCW equivalents) are
never hardcoded in Rust. They are queried once via C shims at first use and
cached via `OnceLock<[i32; N]>` in `src/window.rs`. This is required because
their values are implementation-defined.

## Generic-over-`MpiDatatype` Design

All ferrompi communication APIs are generic over the element type `T`. Four
sealed-trait families in `src/datatype.rs` define which Rust types are valid
for which MPI operations.

| Trait                | Sealed module        | Types                                                                   | Operations                                                        |
| -------------------- | -------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `MpiDatatype`        | `mod sealed`         | `f32`, `f64`, `i32`, `i64`, `u8`, `u32`, `u64`                          | All point-to-point, collectives, RMA                              |
| `AtomicMpiDatatype`  | `mod sealed_atomic`  | `i32`, `i64`, `u32`, `u64`, `u8`                                        | RMA atomic ops (`compare_and_swap`, `fetch_and_op`, `accumulate`) |
| `MpiIndexedDatatype` | `mod sealed_indexed` | `FloatInt`, `DoubleInt`, `LongInt`, `Int2`, `ShortInt`, `LongDoubleInt` | `allreduce_indexed` (MaxLoc/MinLoc)                               |
| `BytePermutable`     | `mod sealed_byte`    | `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `[T; N]`          | `allreduce_bytes` (bitwise ops only)                              |

Each trait uses a private `mod sealed` module containing a `Sealed` marker
trait. External crates cannot implement `sealed::Sealed` and therefore cannot
add new `MpiDatatype` (or any other) implementations. This prevents misuse at
compile time rather than at runtime.

`AtomicMpiDatatype` excludes `f32` and `f64`. Floating-point atomics (compare-
and-swap, fetch-and-op) are not defined by the MPI standard for floating-point
types; admitting `f64` into `Win::compare_and_swap` would silently produce
undefined behaviour on most MPI implementations. A `compile_fail` doctest in
`src/datatype.rs` verifies that `Win<f64>::compare_and_swap` is rejected by the
compiler.

Each trait carries a `const TAG: DatatypeTag` associated constant (for
`MpiDatatype`) or an equivalent tag mechanism, allowing the C layer to identify
the element type at runtime via the stable `#[repr(i32)]` discriminant. At
runtime, the `debug_assert_eq!(T::TAG as i32, mapped_tag(*dt))` assertion in
`UserOp` trampolines detects mismatches in debug builds without a hot-path
branch in release builds.

The authoritative design record for this trait family is
`adr/0003-generic-mpi-datatype.md`.

## Error Handling Model

All fallible ferrompi operations return `Result<T, Error>`. The `Error` type
has two primary variants: `Error::Mpi` for errors originating from the MPI
library, and validation variants (`Error::InvalidOp`, `Error::InvalidBuffer`,
etc.) for errors detected in Rust before any MPI call.

### `Error::Mpi` structure

`Error::Mpi` carries four fields:

| Field       | Type                   | Source                                                        |
| ----------- | ---------------------- | ------------------------------------------------------------- |
| `class`     | `MpiErrorClass`        | `MPI_Error_class(code)` — normalised to a stable Rust enum.   |
| `code`      | `i32`                  | Raw MPI error code as returned by the failing function.       |
| `message`   | `String`               | `MPI_Error_string(code)` — human-readable description.        |
| `operation` | `Option<&'static str>` | The C function name (without `ferrompi_` prefix) that failed. |

Pattern matches on `Error::Mpi` must bind all four fields or use `..`.
This is a compiler-enforced exhaustiveness requirement introduced as a breaking
change in version 0.4.0. Using `_` to suppress the `operation` field silently
discards attribution information; the correct pattern for ignoring a field is
`..`.

### `Error::check_with_op` pattern

```rust,ignore
Error::check_with_op(ret, "allreduce")?;
```

This is the standard idiom at every FFI call site. It returns `Ok(())` when
`ret == MPI_SUCCESS` and constructs `Err(Error::Mpi { operation: Some("allreduce"), .. })`
otherwise. The tag string must match the C function name with `ferrompi_`
stripped; a grep over `csrc/ferrompi.c` confirms each tag.

### `Display` implementation

`Display` for `Error` is hand-rolled in `src/error.rs`. The `thiserror` crate
was removed from the dependency list in version 0.4.0 because `operation`
requires conditional formatting that `#[derive(thiserror::Error)]` cannot
express cleanly. The `[dependencies]` table in `Cargo.toml` is currently empty
as a result.

## Cross-References to ADRs

Five Architecture Decision Records provide the authoritative rationale for
ferrompi's major design choices. These records are immutable once accepted;
if a decision changes, a new ADR supersedes the old one.

- `adr/0001-why-c-wrapper.md` — Why ferrompi uses a hand-written C shim layer
  rather than direct Rust FFI against `libmpi.so`. Covers handle table
  portability, `_c` large-count dispatch, and `MPI_UNDEFINED` normalisation.
- `adr/0002-handle-tables.md` — Concurrency strategy for the request table
  under `MPI_THREAD_MULTIPLE`. Justifies C11 atomics with `atomic_compare_exchange_strong_explicit`
  over a pthread mutex (rejected: serialises the hot read path) and a lock-free
  Treiber stack (rejected: ABA complexity, Cray toolchain gaps).
- `adr/0003-generic-mpi-datatype.md` — Design of the sealed-trait type family
  (`MpiDatatype`, `MpiIndexedDatatype`, `BytePermutable`, `AtomicMpiDatatype`)
  and the `#[repr(i32)]` discriminant ABI contract for `DatatypeTag`.
- `adr/0004-persistent-collective-approach.md` — How ferrompi exposes MPI 4.0
  persistent collective operations (`*_init` / `start` / `wait` lifecycle),
  including the decision to omit `_c` large-count variants from persistent
  shims.
- `adr/0005-mpi-op-create.md` — Seven decisions covering the
  `MPI_Op_create` trampoline safety model: closure storage (per-op static slot
  table), `Send + Sync + 'static` bounds, `MPI_Op_free`-before-slot-release drop
  ordering, default commutativity, per-slot baked-index C trampolines, and
  `catch_unwind + abort` panic handling.
