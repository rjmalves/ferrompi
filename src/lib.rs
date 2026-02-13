//! # ferrompi
//!
//! Safe, generic Rust bindings for MPI (Message Passing Interface).
//!
//! This crate wraps MPI functionality through a thin C layer, providing:
//! - Type-safe generic API for all MPI datatypes
//! - Blocking, nonblocking, and persistent (MPI 4.0+) collectives
//! - Communicator management (split, duplicate)
//! - RMA shared memory windows (with `rma` feature)
//! - SLURM environment helpers (with `numa` feature)
//! - Large count support (MPI 4.0+ `_c` variants)
//!
//! ## Supported Types
//!
//! All communication operations are generic over [`MpiDatatype`]:
//! `f32`, `f64`, `i32`, `i64`, `u8`, `u32`, `u64`
//!
//! ## Quick Start
//!
//! ```no_run
//! use ferrompi::{Mpi, ReduceOp};
//!
//! fn main() -> Result<(), ferrompi::Error> {
//!     let mpi = Mpi::init()?;
//!     let world = mpi.world();
//!
//!     let rank = world.rank();
//!     let size = world.size();
//!     println!("Hello from rank {} of {}", rank, size);
//!
//!     // Generic broadcast — works with any MpiDatatype
//!     let mut data = vec![0.0f64; 100];
//!     if rank == 0 {
//!         data.fill(42.0);
//!     }
//!     world.broadcast(&mut data, 0)?;
//!
//!     // Generic all-reduce
//!     let sum = world.allreduce_scalar(rank as f64, ReduceOp::Sum)?;
//!     println!("Rank {rank}: sum of all ranks = {sum}");
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Description | Dependencies |
//! |---------|-------------|--------------|
//! | `rma`   | RMA shared memory window operations | — |
//! | `numa`  | NUMA-aware windows and SLURM helpers | `rma` |
//! | `debug` | Detailed debug output | — |
//!
//! ## Capabilities
//!
//! - **Generic API**: All operations work with any [`MpiDatatype`] (`f32`, `f64`, `i32`, `i64`, `u8`, `u32`, `u64`)
//! - **Basic collectives**: barrier, broadcast, reduce, allreduce, gather, scatter
//! - **Nonblocking collectives**: ibcast, iallreduce with request handles
//! - **Persistent collectives** (MPI 4.0+): `bcast_init`, `allreduce_init`, etc.
//! - **Rich error handling**: [`MpiErrorClass`] categorization with messages from the MPI runtime

#![warn(missing_docs)]
#![warn(clippy::all)]
// Allow certain pedantic lints for existing code
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::similar_names)]

mod comm;
mod datatype;
mod error;
mod ffi;
mod info;
mod persistent;
mod request;
mod status;

pub use comm::{Communicator, SplitType};
pub use datatype::{DatatypeTag, MpiDatatype};
pub use error::{Error, MpiErrorClass, Result};
pub use info::Info;
pub use persistent::PersistentRequest;
pub use request::Request;
pub use status::Status;

use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

/// Global flag tracking whether MPI has been initialized
static MPI_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// MPI thread support levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum ThreadLevel {
    /// Only single-threaded execution
    Single = 0,
    /// Multi-threaded, but MPI calls only from main thread
    Funneled = 1,
    /// Multi-threaded, but MPI calls serialized by user
    Serialized = 2,
    /// Full multi-threaded support
    Multiple = 3,
}

/// Reduction operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ReduceOp {
    /// Sum of values
    Sum = 0,
    /// Maximum value
    Max = 1,
    /// Minimum value
    Min = 2,
    /// Product of values
    Prod = 3,
}

/// MPI environment handle.
///
/// This type represents an initialized MPI environment. There can only be one
/// instance of this type at a time. When dropped, it finalizes MPI.
///
/// # Example
///
/// ```no_run
/// use ferrompi::Mpi;
///
/// let mpi = Mpi::init().expect("Failed to initialize MPI");
/// let world = mpi.world();
/// println!("Running on {} processes", world.size());
/// // MPI is finalized when `mpi` goes out of scope
/// ```
pub struct Mpi {
    /// The thread level that was provided
    thread_level: ThreadLevel,
    /// Marker to make Mpi !Send and !Sync
    _marker: PhantomData<*const ()>,
}

impl Mpi {
    /// Initialize MPI with single-threaded support.
    ///
    /// # Errors
    ///
    /// Returns an error if MPI is already initialized or if initialization fails.
    pub fn init() -> Result<Self> {
        Self::init_thread(ThreadLevel::Single)
    }

    /// Initialize MPI with the specified thread support level.
    ///
    /// # Arguments
    ///
    /// * `required` - The minimum thread support level required
    ///
    /// # Returns
    ///
    /// Returns the MPI handle. The actual thread support level provided can be
    /// queried with [`thread_level()`](Self::thread_level).
    ///
    /// # Errors
    ///
    /// Returns an error if MPI is already initialized or if initialization fails.
    pub fn init_thread(required: ThreadLevel) -> Result<Self> {
        // Check if already initialized
        if MPI_INITIALIZED.swap(true, Ordering::SeqCst) {
            return Err(Error::AlreadyInitialized);
        }

        let mut provided: i32 = 0;
        let ret = unsafe { ffi::ferrompi_init_thread(required as i32, &mut provided) };

        if ret != 0 {
            MPI_INITIALIZED.store(false, Ordering::SeqCst);
            return Err(Error::from_code(ret));
        }

        let thread_level = match provided {
            0 => ThreadLevel::Single,
            1 => ThreadLevel::Funneled,
            2 => ThreadLevel::Serialized,
            _ => ThreadLevel::Multiple,
        };

        Ok(Mpi {
            thread_level,
            _marker: PhantomData,
        })
    }

    /// Get the thread support level that was provided.
    pub fn thread_level(&self) -> ThreadLevel {
        self.thread_level
    }

    /// Get a handle to `MPI_COMM_WORLD`.
    pub fn world(&self) -> Communicator {
        Communicator::world()
    }

    /// Get the current wall-clock time.
    ///
    /// This is a high-resolution timer suitable for benchmarking.
    pub fn wtime() -> f64 {
        unsafe { ffi::ferrompi_wtime() }
    }

    /// Get the MPI library version string.
    pub fn version() -> Result<String> {
        let mut buf = [0u8; 256];
        let mut len: i32 = 0;
        let ret = unsafe { ffi::ferrompi_get_version(buf.as_mut_ptr().cast::<i8>(), &mut len) };

        if ret != 0 {
            return Err(Error::from_code(ret));
        }

        let len = len.max(0) as usize;
        let s = std::str::from_utf8(&buf[..len])
            .map_err(|_| Error::Internal("Invalid UTF-8 in version string".into()))?;
        Ok(s.to_string())
    }

    /// Check if MPI has been initialized.
    pub fn is_initialized() -> bool {
        let mut flag: i32 = 0;
        unsafe { ffi::ferrompi_initialized(&mut flag) };
        flag != 0
    }

    /// Check if MPI has been finalized.
    pub fn is_finalized() -> bool {
        let mut flag: i32 = 0;
        unsafe { ffi::ferrompi_finalized(&mut flag) };
        flag != 0
    }
}

impl Drop for Mpi {
    fn drop(&mut self) {
        // Only finalize if we successfully initialized
        if MPI_INITIALIZED.load(Ordering::SeqCst) {
            unsafe {
                ffi::ferrompi_finalize();
            }
            MPI_INITIALIZED.store(false, Ordering::SeqCst);
        }
    }
}

// Mpi is not Send or Sync - MPI must be used from the thread that initialized it
// (unless thread level is Multiple)
// This is enforced by PhantomData<*const ()> in the struct

#[cfg(test)]
mod tests {
    // Note: MPI tests must be run with mpiexec
    // cargo build --examples && mpiexec -n 4 ./target/debug/examples/hello_world
}
