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
    pub fn ferrompi_comm_split(
        comm: int32_t,
        color: int32_t,
        key: int32_t,
        newcomm: *mut int32_t,
    ) -> c_int;
    pub fn ferrompi_comm_split_type(
        comm: int32_t,
        split_type: int32_t,
        key: int32_t,
        newcomm: *mut int32_t,
    ) -> c_int;

    // ============================================================
    // Synchronization
    // ============================================================

    pub fn ferrompi_barrier(comm: int32_t) -> c_int;

    // ============================================================
    // Generic Point-to-Point Communication
    // ============================================================

    pub fn ferrompi_send(
        buf: *const c_void,
        count: int64_t,
        datatype_tag: int32_t,
        dest: int32_t,
        tag: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_recv(
        buf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        source: int32_t,
        tag: int32_t,
        comm: int32_t,
        actual_source: *mut int32_t,
        actual_tag: *mut int32_t,
        actual_count: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_isend(
        buf: *const c_void,
        count: int64_t,
        datatype_tag: int32_t,
        dest: int32_t,
        tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_irecv(
        buf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        source: int32_t,
        tag: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_sendrecv(
        sendbuf: *const c_void,
        sendcount: int64_t,
        send_datatype_tag: int32_t,
        dest: int32_t,
        sendtag: int32_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        recv_datatype_tag: int32_t,
        source: int32_t,
        recvtag: int32_t,
        comm: int32_t,
        actual_source: *mut int32_t,
        actual_tag: *mut int32_t,
        actual_count: *mut int64_t,
    ) -> c_int;

    // ============================================================
    // Generic Collective Operations - Blocking
    // ============================================================

    pub fn ferrompi_bcast(
        buf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_reduce(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_allreduce(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_allreduce_inplace(
        buf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_scan(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_exscan(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_gather(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_allgather(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_scatter(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    // ============================================================
    // Generic V-Collectives (variable-count)
    // ============================================================

    pub fn ferrompi_gatherv(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcounts: *const int32_t,
        displs: *const int32_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_scatterv(
        sendbuf: *const c_void,
        sendcounts: *const int32_t,
        displs: *const int32_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_allgatherv(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcounts: *const int32_t,
        displs: *const int32_t,
        datatype_tag: int32_t,
        comm: int32_t,
    ) -> c_int;

    pub fn ferrompi_alltoallv(
        sendbuf: *const c_void,
        sendcounts: *const int32_t,
        sdispls: *const int32_t,
        recvbuf: *mut c_void,
        recvcounts: *const int32_t,
        rdispls: *const int32_t,
        datatype_tag: int32_t,
        comm: int32_t,
    ) -> c_int;

    // ============================================================
    // Generic Collective Operations - Nonblocking
    // ============================================================

    pub fn ferrompi_ibcast(
        buf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_iallreduce(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    // ============================================================
    // Generic Persistent Collectives (MPI 4.0+)
    // ============================================================

    pub fn ferrompi_bcast_init(
        buf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_allreduce_init(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_allreduce_init_inplace(
        buf: *mut c_void,
        count: int64_t,
        datatype_tag: int32_t,
        op: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    pub fn ferrompi_gather_init(
        sendbuf: *const c_void,
        sendcount: int64_t,
        recvbuf: *mut c_void,
        recvcount: int64_t,
        datatype_tag: int32_t,
        root: int32_t,
        comm: int32_t,
        request: *mut int64_t,
    ) -> c_int;

    // ============================================================
    // Info Object Operations
    // ============================================================

    pub fn ferrompi_info_create(info_handle: *mut int32_t) -> c_int;
    pub fn ferrompi_info_free(info_handle: int32_t) -> c_int;
    pub fn ferrompi_info_set(
        info_handle: int32_t,
        key: *const c_char,
        value: *const c_char,
    ) -> c_int;
    pub fn ferrompi_info_get(
        info_handle: int32_t,
        key: *const c_char,
        value: *mut c_char,
        valuelen: *mut int32_t,
        flag: *mut int32_t,
    ) -> c_int;

    // ============================================================
    // Error Information
    // ============================================================

    pub fn ferrompi_error_info(
        code: c_int,
        error_class: *mut int32_t,
        message: *mut c_char,
        msg_len: *mut int32_t,
    ) -> c_int;

    // ============================================================
    // Request Management
    // ============================================================

    pub fn ferrompi_wait(request: int64_t) -> c_int;
    pub fn ferrompi_test(request: int64_t, flag: *mut int32_t) -> c_int;
    pub fn ferrompi_waitall(count: int64_t, requests: *mut int64_t) -> c_int;
    pub fn ferrompi_request_free(request: int64_t) -> c_int;

    // ============================================================
    // Persistent Request Management
    // ============================================================

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
