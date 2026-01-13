//! Safe wrappers for MPI communicator operations.

use crate::error::{Error, Result};
use crate::ffi;
use crate::persistent::PersistentRequest;
use crate::request::Request;
use crate::ReduceOp;
use std::marker::PhantomData;

/// An MPI communicator.
///
/// This type wraps an MPI communicator handle and provides safe methods for
/// collective and point-to-point communication operations.
///
/// # Example
///
/// ```no_run
/// use ferrompi::Mpi;
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
///
/// println!("I am rank {} of {}", world.rank(), world.size());
/// ```
#[derive(Clone)]
pub struct Communicator {
    handle: i32,
    /// Marker to prevent Send/Sync (MPI communicators are not thread-safe)
    _marker: PhantomData<*mut ()>,
}

impl Communicator {
    /// Get a handle to MPI_COMM_WORLD.
    pub(crate) fn world() -> Self {
        Communicator {
            handle: unsafe { ffi::ferrompi_comm_world() },
            _marker: PhantomData,
        }
    }

    /// Get the raw communicator handle (for advanced use).
    pub fn raw_handle(&self) -> i32 {
        self.handle
    }

    /// Get the rank of the calling process in this communicator.
    pub fn rank(&self) -> i32 {
        let mut rank: i32 = 0;
        unsafe { ffi::ferrompi_comm_rank(self.handle, &mut rank) };
        rank
    }

    /// Get the number of processes in this communicator.
    pub fn size(&self) -> i32 {
        let mut size: i32 = 0;
        unsafe { ffi::ferrompi_comm_size(self.handle, &mut size) };
        size
    }

    /// Get the processor name for this process.
    pub fn processor_name(&self) -> Result<String> {
        let mut buf = [0u8; 256];
        let mut len: i32 = 0;
        let ret =
            unsafe { ffi::ferrompi_get_processor_name(buf.as_mut_ptr() as *mut i8, &mut len) };
        Error::check(ret)?;
        let s = std::str::from_utf8(&buf[..len as usize])
            .map_err(|_| Error::Internal("Invalid UTF-8 in processor name".into()))?;
        Ok(s.to_string())
    }

    /// Duplicate this communicator.
    pub fn duplicate(&self) -> Result<Self> {
        let mut new_handle: i32 = 0;
        let ret = unsafe { ffi::ferrompi_comm_dup(self.handle, &mut new_handle) };
        Error::check(ret)?;
        Ok(Communicator {
            handle: new_handle,
            _marker: PhantomData,
        })
    }

    // ========================================================================
    // Synchronization
    // ========================================================================

    /// Barrier synchronization.
    ///
    /// All processes in the communicator must call this function. No process
    /// will return until all processes have entered the barrier.
    pub fn barrier(&self) -> Result<()> {
        let ret = unsafe { ffi::ferrompi_barrier(self.handle) };
        Error::check(ret)
    }

    // ========================================================================
    // Point-to-Point Communication
    // ========================================================================

    /// Send a slice of f64 values to another process.
    pub fn send_f64(&self, data: &[f64], dest: i32, tag: i32) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_send_f64(data.as_ptr(), data.len() as i64, dest, tag, self.handle)
        };
        Error::check(ret)
    }

    /// Receive a slice of f64 values from another process.
    ///
    /// Use `source = -1` for MPI_ANY_SOURCE and `tag = -1` for MPI_ANY_TAG.
    ///
    /// Returns `(actual_source, actual_tag, actual_count)`.
    pub fn recv_f64(&self, data: &mut [f64], source: i32, tag: i32) -> Result<(i32, i32, i64)> {
        let mut actual_source: i32 = 0;
        let mut actual_tag: i32 = 0;
        let mut actual_count: i64 = 0;

        let ret = unsafe {
            ffi::ferrompi_recv_f64(
                data.as_mut_ptr(),
                data.len() as i64,
                source,
                tag,
                self.handle,
                &mut actual_source,
                &mut actual_tag,
                &mut actual_count,
            )
        };
        Error::check(ret)?;
        Ok((actual_source, actual_tag, actual_count))
    }

    // ========================================================================
    // Blocking Collectives
    // ========================================================================

    /// Broadcast a slice of f64 values from root to all processes.
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer to broadcast (input at root, output at others)
    /// * `root` - Rank of the root process
    pub fn broadcast_f64(&self, data: &mut [f64], root: i32) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_bcast_f64(data.as_mut_ptr(), data.len() as i64, root, self.handle)
        };
        Error::check(ret)
    }

    /// Broadcast a slice of i32 values from root to all processes.
    pub fn broadcast_i32(&self, data: &mut [i32], root: i32) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_bcast_i32(data.as_mut_ptr(), data.len() as i64, root, self.handle)
        };
        Error::check(ret)
    }

    /// Broadcast a slice of i64 values from root to all processes.
    pub fn broadcast_i64(&self, data: &mut [i64], root: i32) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_bcast_i64(data.as_mut_ptr(), data.len() as i64, root, self.handle)
        };
        Error::check(ret)
    }

    /// Broadcast raw bytes from root to all processes.
    pub fn broadcast_bytes(&self, data: &mut [u8], root: i32) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_bcast_bytes(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                data.len() as i64,
                root,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// Reduce f64 values to the root process.
    ///
    /// # Arguments
    ///
    /// * `send` - Data to send from this process
    /// * `recv` - Buffer for result (only significant at root)
    /// * `op` - Reduction operation
    /// * `root` - Rank of the root process
    pub fn reduce_f64(
        &self,
        send: &[f64],
        recv: &mut [f64],
        op: ReduceOp,
        root: i32,
    ) -> Result<()> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let ret = unsafe {
            ffi::ferrompi_reduce_f64(
                send.as_ptr(),
                recv.as_mut_ptr(),
                send.len() as i64,
                op as i32,
                root,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// All-reduce f64 values (reduce and broadcast result to all).
    ///
    /// # Arguments
    ///
    /// * `send` - Data to send from this process
    /// * `recv` - Buffer for result
    /// * `op` - Reduction operation
    pub fn allreduce_f64(&self, send: &[f64], recv: &mut [f64], op: ReduceOp) -> Result<()> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let ret = unsafe {
            ffi::ferrompi_allreduce_f64(
                send.as_ptr(),
                recv.as_mut_ptr(),
                send.len() as i64,
                op as i32,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// All-reduce a single f64 value.
    ///
    /// Convenience method for reducing a single scalar.
    pub fn allreduce_scalar(&self, value: f64, op: ReduceOp) -> Result<f64> {
        let send = [value];
        let mut recv = [0.0];
        self.allreduce_f64(&send, &mut recv, op)?;
        Ok(recv[0])
    }

    /// All-reduce f64 values in place.
    pub fn allreduce_inplace_f64(&self, data: &mut [f64], op: ReduceOp) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_allreduce_inplace_f64(
                data.as_mut_ptr(),
                data.len() as i64,
                op as i32,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// Gather f64 values to the root process.
    ///
    /// Each process sends `sendcount` elements. Root receives `sendcount * size`
    /// elements total.
    ///
    /// # Arguments
    ///
    /// * `send` - Data to send from this process
    /// * `recv` - Buffer for received data (only significant at root, must be
    ///   `send.len() * size` elements)
    /// * `root` - Rank of the root process
    pub fn gather_f64(&self, send: &[f64], recv: &mut [f64], root: i32) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_gather_f64(
                send.as_ptr(),
                send.len() as i64,
                recv.as_mut_ptr(),
                send.len() as i64,
                root,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// All-gather f64 values (gather and broadcast to all).
    pub fn allgather_f64(&self, send: &[f64], recv: &mut [f64]) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_allgather_f64(
                send.as_ptr(),
                send.len() as i64,
                recv.as_mut_ptr(),
                send.len() as i64,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// Scatter f64 values from root to all processes.
    ///
    /// Root sends `recvcount * size` elements total, each process receives
    /// `recvcount` elements.
    pub fn scatter_f64(&self, send: &[f64], recv: &mut [f64], root: i32) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_scatter_f64(
                send.as_ptr(),
                recv.len() as i64,
                recv.as_mut_ptr(),
                recv.len() as i64,
                root,
                self.handle,
            )
        };
        Error::check(ret)
    }

    // ========================================================================
    // Nonblocking Collectives
    // ========================================================================

    /// Nonblocking broadcast of f64 values.
    ///
    /// Returns a request handle that must be waited on before accessing the buffer.
    ///
    /// # Safety
    ///
    /// The buffer must remain valid until the request is completed.
    pub fn ibroadcast_f64(&self, data: &mut [f64], root: i32) -> Result<Request> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_ibcast_f64(
                data.as_mut_ptr(),
                data.len() as i64,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check(ret)?;
        Ok(Request::new(request_handle))
    }

    /// Nonblocking all-reduce of f64 values.
    ///
    /// Returns a request handle that must be waited on before accessing the buffer.
    pub fn iallreduce_f64(&self, send: &[f64], recv: &mut [f64], op: ReduceOp) -> Result<Request> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_iallreduce_f64(
                send.as_ptr(),
                recv.as_mut_ptr(),
                send.len() as i64,
                op as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check(ret)?;
        Ok(Request::new(request_handle))
    }

    // ========================================================================
    // Persistent Collectives (MPI 4.0+)
    // ========================================================================

    /// Initialize a persistent broadcast operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer to use for broadcasts (must remain valid for lifetime of handle)
    /// * `root` - Rank of the root process
    ///
    /// # Note
    ///
    /// This requires MPI 4.0+. Returns an error if not supported.
    pub fn bcast_init_f64(&self, data: &mut [f64], root: i32) -> Result<PersistentRequest> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_bcast_init_f64(
                data.as_mut_ptr(),
                data.len() as i64,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        if ret != 0 {
            // Check if it's an "unsupported" error
            return Err(Error::NotSupported(
                "Persistent collectives require MPI 4.0+".into(),
            ));
        }
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent all-reduce operation.
    ///
    /// # Note
    ///
    /// This requires MPI 4.0+.
    pub fn allreduce_init_f64(
        &self,
        send: &[f64],
        recv: &mut [f64],
        op: ReduceOp,
    ) -> Result<PersistentRequest> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_allreduce_init_f64(
                send.as_ptr(),
                recv.as_mut_ptr(),
                send.len() as i64,
                op as i32,
                self.handle,
                &mut request_handle,
            )
        };
        if ret != 0 {
            return Err(Error::NotSupported(
                "Persistent collectives require MPI 4.0+".into(),
            ));
        }
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent in-place all-reduce operation.
    pub fn allreduce_init_inplace_f64(
        &self,
        data: &mut [f64],
        op: ReduceOp,
    ) -> Result<PersistentRequest> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_allreduce_init_inplace_f64(
                data.as_mut_ptr(),
                data.len() as i64,
                op as i32,
                self.handle,
                &mut request_handle,
            )
        };
        if ret != 0 {
            return Err(Error::NotSupported(
                "Persistent collectives require MPI 4.0+".into(),
            ));
        }
        Ok(PersistentRequest::new(request_handle))
    }
}

impl Drop for Communicator {
    fn drop(&mut self) {
        // Don't free COMM_WORLD (handle 0)
        if self.handle != 0 {
            unsafe { ffi::ferrompi_comm_free(self.handle) };
        }
    }
}

// Communicators are not Send or Sync by default
// (MPI communicators have thread-safety requirements)
