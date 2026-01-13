//! Raw FFI bindings to the C wrapper layer.
//!
//! These are low-level unsafe functions. Use the safe wrappers in the parent module.

#![allow(dead_code)]
#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_double, c_int, c_void};

// Type aliases matching the C header
pub type int32_t = i32;
pub type int64_t = i64;

extern "C" {
    // ============================================================
    // Initialization and Finalization
    // ============================================================

    pub fn ferrompi_init_thread(required: c_int, provided: *mut c_int) -> c_int;
    pub fn ferrompi_init() -> c_int;
    pub fn ferrompi_finalize() -> c_int;
    pub fn ferrompi_initialized(flag: *mut c_int) -> c_int;
    pub fn ferrompi_finalized(flag: *mut c_int) -> c_int;

    // ============================================================
    // Communicator Operations
    // ============================================================

    pub fn ferrompi_comm_world() -> int32_t;
    pub fn ferrompi_comm_rank(comm: int32_t, rank: *mut int32_t) -> c_int;
    pub fn ferrompi_comm_size(comm: int32_t, size: *mut int32_t) -> c_int;
    pub fn ferrompi_comm_dup(comm: int32_t, newcomm: *mut int32_t) -> c_int;
    pub fn ferrompi_comm_free(comm: int32_t) -> c_int;

    // ============================================================
    // Synchronization
    // ============================================================

    pub fn ferrompi_barrier(comm: int32_t) -> c_int;

    // ============================================================
    // Point-to-Point Communication
    // ============================================================

    pub fn ferrompi_send_f64(
        buf: *const c_double,
        count: int64_t,
        dest: int32_t,
        tag: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_recv_f64(
        buf: *mut c_double,
        count: int64_t,
        source: int32_t,
        tag: int32_t,
        comm: int32_t,
        actual_source: *mut int32_t,
        actual_tag: *mut int32_t,
        actual_count: *mut int64_t,
    ) -> c_int;

    // ============================================================
    // Collective Operations - Blocking
    // ============================================================

    pub fn ferrompi_bcast_f64(
        buf: *mut c_double,
        count: int64_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_bcast_i32(
        buf: *mut int32_t,
        count: int64_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_bcast_i64(
        buf: *mut int64_t,
        count: int64_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_bcast_bytes(
        buf: *mut c_void,
        count: int64_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_reduce_f64(
        sendbuf: *const c_double,
        recvbuf: *mut c_double,
        count: int64_t,
        op: int32_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_allreduce_f64(
        sendbuf: *const c_double,
        recvbuf: *mut c_double,
        count: int64_t,
        op: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_allreduce_inplace_f64(
        buf: *mut c_double,
        count: int64_t,
        op: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_gather_f64(
        sendbuf: *const c_double,
        sendcount: int64_t,
        recvbuf: *mut c_double,
        recvcount: int64_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_allgather_f64(
        sendbuf: *const c_double,
        sendcount: int64_t,
        recvbuf: *mut c_double,
        recvcount: int64_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_scatter_f64(
        sendbuf: *const c_double,
        sendcount: int64_t,
        recvbuf: *mut c_double,
        recvcount: int64_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    // ============================================================
    // Collective Operations - Nonblocking
    // ============================================================

    pub fn ferrompi_ibcast_f64(
        buf: *mut c_double,
        count: int64_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_iallreduce_f64(
        sendbuf: *const c_double,
        recvbuf: *mut c_double,
        count: int64_t,
        op: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    // ============================================================
    // Request Management
    // ============================================================

    pub fn ferrompi_wait(request: int64_t) -> c_int;
    pub fn ferrompi_test(request: int64_t, flag: *mut int32_t) -> c_int;
    pub fn ferrompi_waitall(count: int64_t, requests: *mut int64_t) -> c_int;
    pub fn ferrompi_request_free(request: int64_t) -> c_int;

    // ============================================================
    // Persistent Collectives (MPI 4.0+)
    // ============================================================

    pub fn ferrompi_bcast_init_f64(
        buf: *mut c_double,
        count: int64_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_allreduce_init_f64(
        sendbuf: *const c_double,
        recvbuf: *mut c_double,
        count: int64_t,
        op: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_allreduce_init_inplace_f64(
        buf: *mut c_double,
        count: int64_t,
        op: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_gather_init_f64(
        sendbuf: *const c_double,
        sendcount: int64_t,
        recvbuf: *mut c_double,
        recvcount: int64_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_start(request: int64_t) -> c_int;
    pub fn ferrompi_startall(count: int64_t, requests: *mut int64_t) -> c_int;

    // ============================================================
    // Utility Functions
    // ============================================================

    pub fn ferrompi_get_version(version: *mut c_char, len: *mut int32_t) -> c_int;
    pub fn ferrompi_get_processor_name(name: *mut c_char, len: *mut int32_t) -> c_int;
    pub fn ferrompi_wtime() -> c_double;
    pub fn ferrompi_abort(comm: int32_t, errorcode: int32_t) -> c_int;
}
