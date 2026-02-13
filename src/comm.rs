//! Safe wrappers for MPI communicator operations.
//!
//! All communication methods are generic over [`MpiDatatype`], supporting
//! `f32`, `f64`, `i32`, `i64`, `u8`, `u32`, and `u64`.

use crate::datatype::MpiDatatype;
use crate::error::{Error, Result};
use crate::ffi;
use crate::persistent::PersistentRequest;
use crate::request::Request;
use crate::ReduceOp;
use std::marker::PhantomData;

/// Split types for [`Communicator::split_type`].
///
/// These constants map to MPI communicator split type values. Currently only
/// shared-memory splits are supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SplitType {
    /// Split by shared memory domain (same physical node).
    /// Maps to `MPI_COMM_TYPE_SHARED`.
    Shared = 0,
}

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
    /// Constant for opting out of a communicator split.
    ///
    /// Processes passing this as the `color` to [`split()`](Self::split) will not be
    /// included in any resulting communicator.
    pub const UNDEFINED: i32 = -1;

    /// Get a handle to `MPI_COMM_WORLD`.
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
            unsafe { ffi::ferrompi_get_processor_name(buf.as_mut_ptr().cast::<i8>(), &mut len) };
        Error::check(ret)?;
        let len = len.max(0) as usize;
        let s = std::str::from_utf8(&buf[..len])
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

    /// Split this communicator into sub-communicators based on color and key.
    ///
    /// Processes with the same `color` are placed in the same new communicator.
    /// The `key` controls the rank ordering within the new communicator.
    ///
    /// Returns `None` if this process used [`Communicator::UNDEFINED`] as color.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Mpi;
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let color = world.rank() % 2; // Even/odd split
    /// if let Some(sub) = world.split(color, world.rank()).unwrap() {
    ///     println!("Rank {} in sub-communicator of size {}", sub.rank(), sub.size());
    /// }
    /// ```
    pub fn split(&self, color: i32, key: i32) -> Result<Option<Communicator>> {
        let mut new_handle: i32 = 0;
        let ret = unsafe { ffi::ferrompi_comm_split(self.handle, color, key, &mut new_handle) };
        Error::check(ret)?;
        if new_handle < 0 {
            Ok(None)
        } else {
            Ok(Some(Communicator {
                handle: new_handle,
                _marker: PhantomData,
            }))
        }
    }

    /// Split this communicator by type.
    ///
    /// Processes that share the same resource (determined by `split_type`) are
    /// placed in the same new communicator. The `key` controls the rank ordering
    /// within the new communicator.
    ///
    /// Returns `None` if MPI returns `MPI_COMM_NULL` for this process.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, SplitType};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// if let Some(node) = world.split_type(SplitType::Shared, world.rank()).unwrap() {
    ///     println!("Node has {} processes", node.size());
    /// }
    /// ```
    pub fn split_type(&self, split_type: SplitType, key: i32) -> Result<Option<Communicator>> {
        let mut new_handle: i32 = 0;
        let ret = unsafe {
            ffi::ferrompi_comm_split_type(self.handle, split_type as i32, key, &mut new_handle)
        };
        Error::check(ret)?;
        if new_handle < 0 {
            Ok(None)
        } else {
            Ok(Some(Communicator {
                handle: new_handle,
                _marker: PhantomData,
            }))
        }
    }

    /// Create a communicator containing only processes that share memory.
    ///
    /// This is equivalent to `split_type(SplitType::Shared, self.rank())`.
    /// All processes on the same physical node will be in the same communicator.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`] if MPI unexpectedly returns a null communicator,
    /// which should not happen for `MPI_COMM_TYPE_SHARED` under normal conditions.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Mpi;
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let node = world.split_shared().unwrap();
    /// println!("Node has {} processes, I am local rank {}", node.size(), node.rank());
    /// ```
    pub fn split_shared(&self) -> Result<Communicator> {
        self.split_type(SplitType::Shared, self.rank())?
            .ok_or_else(|| Error::Internal("split_shared returned null communicator".into()))
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
    // Generic Point-to-Point Communication
    // ========================================================================

    /// Send a slice of values to another process.
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer to send
    /// * `dest` - Destination rank
    /// * `tag` - Message tag
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let data = vec![1.0f64, 2.0, 3.0];
    /// world.send(&data, 1, 0).unwrap();
    /// ```
    pub fn send<T: MpiDatatype>(&self, data: &[T], dest: i32, tag: i32) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_send(
                data.as_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                dest,
                tag,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// Receive a slice of values from another process.
    ///
    /// Use `source = -1` for `MPI_ANY_SOURCE` and `tag = -1` for `MPI_ANY_TAG`.
    ///
    /// Returns `(actual_source, actual_tag, actual_count)`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut buf = vec![0.0f64; 10];
    /// let (source, tag, count) = world.recv(&mut buf, 0, 0).unwrap();
    /// ```
    pub fn recv<T: MpiDatatype>(
        &self,
        data: &mut [T],
        source: i32,
        tag: i32,
    ) -> Result<(i32, i32, i64)> {
        let mut actual_source: i32 = 0;
        let mut actual_tag: i32 = 0;
        let mut actual_count: i64 = 0;

        let ret = unsafe {
            ffi::ferrompi_recv(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
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

    /// Nonblocking send.
    ///
    /// Initiates a send operation and returns immediately with a [`Request`]
    /// handle. The send buffer **must not be modified** until the request is
    /// completed via [`Request::wait()`] or [`Request::test()`].
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer to send (must remain valid until the request completes)
    /// * `dest` - Destination rank
    /// * `tag` - Message tag
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let data = vec![1.0f64, 2.0, 3.0];
    /// let req = world.isend(&data, 1, 0).unwrap();
    /// // ... do other work ...
    /// req.wait().unwrap();
    /// ```
    pub fn isend<T: MpiDatatype>(&self, data: &[T], dest: i32, tag: i32) -> Result<Request> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_isend(
                data.as_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                dest,
                tag,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check(ret)?;
        Ok(Request::new(request_handle))
    }

    /// Nonblocking receive.
    ///
    /// Initiates a receive operation and returns immediately with a [`Request`]
    /// handle. The receive buffer **must not be read** until the request is
    /// completed via [`Request::wait()`] or [`Request::test()`].
    ///
    /// Use `source = -1` for `MPI_ANY_SOURCE` and `tag = -1` for `MPI_ANY_TAG`.
    ///
    /// # Arguments
    ///
    /// * `data` - Receive buffer (must remain valid until the request completes)
    /// * `source` - Source rank (or -1 for any source)
    /// * `tag` - Message tag (or -1 for any tag)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut buf = vec![0.0f64; 10];
    /// let req = world.irecv(&mut buf, 0, 0).unwrap();
    /// // ... do other work ...
    /// req.wait().unwrap();
    /// ```
    pub fn irecv<T: MpiDatatype>(&self, data: &mut [T], source: i32, tag: i32) -> Result<Request> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_irecv(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                source,
                tag,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check(ret)?;
        Ok(Request::new(request_handle))
    }

    /// Blocking send-receive.
    ///
    /// Sends data to one process and receives from another (or the same) in a
    /// single operation. This is useful for avoiding deadlocks in ring-style
    /// communication patterns where each process both sends and receives.
    ///
    /// Use `source = -1` for `MPI_ANY_SOURCE` and `recvtag = -1` for `MPI_ANY_TAG`.
    ///
    /// Returns `(actual_source, actual_tag, actual_count)`.
    ///
    /// # Arguments
    ///
    /// * `send` - Buffer to send
    /// * `dest` - Destination rank
    /// * `sendtag` - Send message tag
    /// * `recv` - Receive buffer
    /// * `source` - Source rank (or -1 for any source)
    /// * `recvtag` - Receive message tag (or -1 for any tag)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![world.rank() as f64; 5];
    /// let mut recv = vec![0.0f64; 5];
    /// let next = (world.rank() + 1) % world.size();
    /// let prev = (world.rank() - 1 + world.size()) % world.size();
    /// let (src, tag, count) = world.sendrecv(&send, next, 0, &mut recv, prev, 0).unwrap();
    /// ```
    pub fn sendrecv<T: MpiDatatype>(
        &self,
        send: &[T],
        dest: i32,
        sendtag: i32,
        recv: &mut [T],
        source: i32,
        recvtag: i32,
    ) -> Result<(i32, i32, i64)> {
        let mut actual_source: i32 = 0;
        let mut actual_tag: i32 = 0;
        let mut actual_count: i64 = 0;

        let ret = unsafe {
            ffi::ferrompi_sendrecv(
                send.as_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                dest,
                sendtag,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recv.len() as i64,
                T::TAG as i32,
                source,
                recvtag,
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
    // Generic Blocking Collectives
    // ========================================================================

    /// Broadcast a slice from root to all processes.
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer to broadcast (input at root, output at others)
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut data = vec![0.0f64; 100];
    /// if world.rank() == 0 {
    ///     data.fill(42.0);
    /// }
    /// world.broadcast(&mut data, 0).unwrap();
    /// ```
    pub fn broadcast<T: MpiDatatype>(&self, data: &mut [T], root: i32) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_bcast(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                root,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// Reduce values to the root process.
    ///
    /// # Arguments
    ///
    /// * `send` - Data to send from this process
    /// * `recv` - Buffer for result (only significant at root)
    /// * `op` - Reduction operation
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 10];
    /// let mut recv = vec![0.0f64; 10];
    /// world.reduce(&send, &mut recv, ReduceOp::Sum, 0).unwrap();
    /// ```
    pub fn reduce<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
        root: i32,
    ) -> Result<()> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let ret = unsafe {
            ffi::ferrompi_reduce(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                op as i32,
                root,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// All-reduce values (reduce and broadcast result to all).
    ///
    /// # Arguments
    ///
    /// * `send` - Data to send from this process
    /// * `recv` - Buffer for result
    /// * `op` - Reduction operation
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![world.rank() as f64; 10];
    /// let mut recv = vec![0.0f64; 10];
    /// world.allreduce(&send, &mut recv, ReduceOp::Sum).unwrap();
    /// ```
    pub fn allreduce<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<()> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let ret = unsafe {
            ffi::ferrompi_allreduce(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                op as i32,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// All-reduce values in place.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut data = vec![world.rank() as f64; 10];
    /// world.allreduce_inplace(&mut data, ReduceOp::Sum).unwrap();
    /// ```
    pub fn allreduce_inplace<T: MpiDatatype>(&self, data: &mut [T], op: ReduceOp) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_allreduce_inplace(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                op as i32,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// All-reduce a single scalar value.
    ///
    /// Convenience method for reducing a single element.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let sum = world.allreduce_scalar(world.rank() as f64, ReduceOp::Sum).unwrap();
    /// println!("Sum of all ranks: {sum}");
    /// ```
    pub fn allreduce_scalar<T: MpiDatatype>(&self, value: T, op: ReduceOp) -> Result<T> {
        let send = [value];
        // SAFETY: T is Copy, so zero-init is safe for numeric types
        let mut recv = [value]; // placeholder, will be overwritten
        self.allreduce(&send, &mut recv, op)?;
        Ok(recv[0])
    }

    /// Gather values to the root process.
    ///
    /// Each process sends `send.len()` elements. Root receives
    /// `send.len() * size` elements total.
    ///
    /// # Arguments
    ///
    /// * `send` - Data to send from this process
    /// * `recv` - Buffer for received data (only significant at root, must be
    ///   `send.len() * size` elements)
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![world.rank() as f64; 5];
    /// let mut recv = vec![0.0f64; 5 * world.size() as usize];
    /// world.gather(&send, &mut recv, 0).unwrap();
    /// ```
    pub fn gather<T: MpiDatatype>(&self, send: &[T], recv: &mut [T], root: i32) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_gather(
                send.as_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                root,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// All-gather values (gather and broadcast to all).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![world.rank() as i32; 3];
    /// let mut recv = vec![0i32; 3 * world.size() as usize];
    /// world.allgather(&send, &mut recv).unwrap();
    /// ```
    pub fn allgather<T: MpiDatatype>(&self, send: &[T], recv: &mut [T]) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_allgather(
                send.as_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// Scatter values from root to all processes.
    ///
    /// Root sends `recv.len() * size` elements total, each process receives
    /// `recv.len()` elements.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![0.0f64; 5 * world.size() as usize];
    /// let mut recv = vec![0.0f64; 5];
    /// world.scatter(&send, &mut recv, 0).unwrap();
    /// ```
    pub fn scatter<T: MpiDatatype>(&self, send: &[T], recv: &mut [T], root: i32) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_scatter(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.len() as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recv.len() as i64,
                T::TAG as i32,
                root,
                self.handle,
            )
        };
        Error::check(ret)
    }

    // ========================================================================
    // Generic Nonblocking Collectives
    // ========================================================================

    /// Nonblocking broadcast.
    ///
    /// Returns a request handle that must be waited on before accessing the buffer.
    ///
    /// # Safety Note
    ///
    /// The buffer must remain valid until the request is completed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut data = vec![0.0f64; 100];
    /// let req = world.ibroadcast(&mut data, 0).unwrap();
    /// // ... do other work ...
    /// req.wait().unwrap();
    /// ```
    pub fn ibroadcast<T: MpiDatatype>(&self, data: &mut [T], root: i32) -> Result<Request> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_ibcast(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check(ret)?;
        Ok(Request::new(request_handle))
    }

    /// Nonblocking all-reduce.
    ///
    /// Returns a request handle that must be waited on before accessing the buffer.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 10];
    /// let mut recv = vec![0.0f64; 10];
    /// let req = world.iallreduce(&send, &mut recv, ReduceOp::Sum).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn iallreduce<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<Request> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_iallreduce(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                op as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check(ret)?;
        Ok(Request::new(request_handle))
    }

    // ========================================================================
    // Generic Persistent Collectives (MPI 4.0+)
    // ========================================================================

    /// Initialize a persistent broadcast operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0+.
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer to use for broadcasts (must remain valid for lifetime of handle)
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut data = vec![0.0f64; 100];
    /// let mut persistent = world.bcast_init(&mut data, 0).unwrap();
    /// for _ in 0..100 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn bcast_init<T: MpiDatatype>(
        &self,
        data: &mut [T],
        root: i32,
    ) -> Result<PersistentRequest> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_bcast_init(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                root,
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

    /// Initialize a persistent all-reduce operation.
    ///
    /// Requires MPI 4.0+.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 10];
    /// let mut recv = vec![0.0f64; 10];
    /// let mut persistent = world.allreduce_init(&send, &mut recv, ReduceOp::Sum).unwrap();
    /// for _ in 0..100 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn allreduce_init<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<PersistentRequest> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_allreduce_init(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
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
    ///
    /// Requires MPI 4.0+.
    pub fn allreduce_init_inplace<T: MpiDatatype>(
        &self,
        data: &mut [T],
        op: ReduceOp,
    ) -> Result<PersistentRequest> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_allreduce_init_inplace(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
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

    /// Initialize a persistent gather operation.
    ///
    /// Requires MPI 4.0+.
    pub fn gather_init<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        root: i32,
    ) -> Result<PersistentRequest> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_gather_init(
                send.as_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                root,
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
