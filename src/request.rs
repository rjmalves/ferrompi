//! Request handles for nonblocking MPI operations.

use crate::error::{Error, Result};
use crate::ffi;

/// A handle to a nonblocking MPI operation.
///
/// This type represents an in-flight MPI operation. You must call `wait()` or
/// `test()` to complete the operation before the associated buffers can be
/// safely accessed.
///
/// # Example
///
/// ```no_run
/// use ferrompi::{Mpi, ReduceOp};
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
///
/// let send = vec![world.rank() as f64; 10];
/// let mut recv = vec![0.0; 10];
///
/// // Start nonblocking all-reduce
/// let request = world.iallreduce_f64(&send, &mut recv, ReduceOp::Sum).unwrap();
///
/// // Do other work while communication proceeds...
///
/// // Wait for completion
/// request.wait().unwrap();
///
/// // Now recv contains the result
/// println!("Sum: {:?}", recv);
/// ```
pub struct Request {
    handle: i64,
    completed: bool,
}

impl Request {
    /// Create a new request from a raw handle.
    pub(crate) fn new(handle: i64) -> Self {
        Request {
            handle,
            completed: false,
        }
    }

    /// Get the raw request handle (for advanced use).
    pub fn raw_handle(&self) -> i64 {
        self.handle
    }

    /// Check if this request has been completed.
    pub fn is_completed(&self) -> bool {
        self.completed
    }

    /// Wait for this operation to complete.
    ///
    /// Blocks until the operation is finished. After this returns successfully,
    /// the associated buffers can be safely accessed.
    pub fn wait(mut self) -> Result<()> {
        if self.completed {
            return Ok(());
        }
        let ret = unsafe { ffi::ferrompi_wait(self.handle) };
        self.completed = true;
        Error::check(ret)
    }

    /// Test if this operation has completed without blocking.
    ///
    /// Returns `true` if the operation is complete, `false` otherwise.
    ///
    /// # Note
    ///
    /// If this returns `true`, the request is consumed and you should not call
    /// `wait()` or `test()` again.
    pub fn test(&mut self) -> Result<bool> {
        if self.completed {
            return Ok(true);
        }
        let mut flag: i32 = 0;
        let ret = unsafe { ffi::ferrompi_test(self.handle, &mut flag) };
        Error::check(ret)?;
        if flag != 0 {
            self.completed = true;
        }
        Ok(flag != 0)
    }

    /// Wait for all requests in a collection to complete.
    ///
    /// This is more efficient than waiting for each request individually.
    pub fn wait_all(requests: Vec<Request>) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }

        let mut handles: Vec<i64> = requests.iter().map(|r| r.handle).collect();
        let ret = unsafe { ffi::ferrompi_waitall(handles.len() as i64, handles.as_mut_ptr()) };

        // Mark all as completed (they're consumed anyway)
        std::mem::forget(requests);

        Error::check(ret)
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        if !self.completed {
            // If the request wasn't waited on, we need to wait now to avoid
            // leaving the operation in an undefined state
            unsafe { ffi::ferrompi_wait(self.handle) };
        }
    }
}
