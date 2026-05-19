//! Long-form documentation for ferrompi.
//!
//! This module embeds the Markdown files from the `docs/` directory directly
//! into rustdoc via `#[doc = include_str!(...)]`. Each sub-module corresponds
//! to one documentation artifact. The same content is available as plain
//! Markdown on GitHub at `docs/` and `docs/adr/`; see [`docs/README.md`] for
//! a navigable index with relative-link support.
//!
//! [`docs/README.md`]: https://github.com/cobre-rs/ferrompi/blob/main/docs/README.md

/// Architecture overview for ferrompi contributors.
///
/// Describes the six-layer stack, handle tables, thread-safety model, C layer
/// scope, FFI/ABI invariants, the generic `MpiDatatype` trait family, and the
/// error handling model.
#[doc = include_str!("../docs/architecture.md")]
pub mod architecture {}

/// Migration guide from rsmpi to ferrompi.
///
/// Covers the quick-comparison table, a function-for-function API mapping,
/// migration cookbook examples, unsupported features, and API ergonomic
/// differences.
#[doc = include_str!("../docs/migrating-from-rsmpi.md")]
pub mod migrating_from_rsmpi {}

/// MPI implementation compatibility matrix.
///
/// Documents which features are available on MPICH 3.x/4.x, Open MPI 4/5,
/// Intel MPI, and Cray MPI, including known issues and how to report new
/// compatibility data.
#[doc = include_str!("../docs/mpi-compatibility.md")]
pub mod mpi_compatibility {}

/// ADR-0001: Why ferrompi uses a hand-written C wrapper layer.
///
/// Explains the ABI portability problem with `bindgen`, the handle-table
/// pattern, large-count version gating, and the op-trampoline infrastructure.
#[doc = include_str!("../docs/adr/0001-why-c-wrapper.md")]
pub mod adr_0001_why_c_wrapper {}

/// ADR-0002: Handle-table concurrency strategy for the request table.
///
/// Justifies C11 atomic `compare_exchange_strong` over pthread mutex and a
/// lock-free Treiber stack for safe concurrent slot allocation under
/// `MPI_THREAD_MULTIPLE`.
#[doc = include_str!("../docs/adr/0002-handle-tables.md")]
pub mod adr_0002_handle_tables {}

/// ADR-0003: Sealed generic `MpiDatatype` trait family.
///
/// Documents the design of the sealed-trait type family (`MpiDatatype`,
/// `MpiIndexedDatatype`, `BytePermutable`, `AtomicMpiDatatype`) and the
/// `#[repr(i32)]` discriminant ABI contract for `DatatypeTag`.
#[doc = include_str!("../docs/adr/0003-generic-mpi-datatype.md")]
pub mod adr_0003_generic_mpi_datatype {}

/// ADR-0004: `PersistentRequest` lifecycle and buffer-lifetime invariants.
///
/// Covers the `*_init` / `start` / `wait` lifecycle, the decision to omit
/// `_c` large-count variants from persistent shims, and the buffer-borrow
/// safety model.
#[doc = include_str!("../docs/adr/0004-persistent-collective-approach.md")]
pub mod adr_0004_persistent_collective_approach {}

/// ADR-0005: `MPI_Op_create` safety model.
///
/// Seven decisions covering closure storage (per-op static slot table),
/// `Send + Sync + 'static` bounds, `MPI_Op_free`-before-slot-release drop
/// ordering, default commutativity, per-slot baked-index C trampolines, and
/// `catch_unwind + abort` panic handling.
#[doc = include_str!("../docs/adr/0005-mpi-op-create.md")]
pub mod adr_0005_mpi_op_create {}
