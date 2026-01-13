/**
 * ferrompi.h - Thin C wrapper for MPI 4.x features
 * 
 * This header provides a stable FFI interface for Rust to access
 * MPI functionality, particularly MPI 4.0+ features not available
 * in rsmpi.
 * 
 * Design principles:
 * - Minimal logic in C, just call forwarding
 * - Use fixed-width types for FFI safety
 * - Handle table for opaque MPI objects
 * - Error codes returned directly (MPI_SUCCESS = 0)
 */

#ifndef ferrompi_H
#define ferrompi_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Initialization and Finalization
 * ============================================================ */

/**
 * Initialize MPI with thread support
 * @param required Thread level required (0=SINGLE, 1=FUNNELED, 2=SERIALIZED, 3=MULTIPLE)
 * @param provided Output: actual thread level provided
 * @return MPI error code
 */
int ferrompi_init_thread(int required, int* provided);

/**
 * Initialize MPI (single-threaded)
 * @return MPI error code
 */
int ferrompi_init(void);

/**
 * Finalize MPI
 * @return MPI error code
 */
int ferrompi_finalize(void);

/**
 * Check if MPI is initialized
 * @param flag Output: 1 if initialized, 0 otherwise
 * @return MPI error code
 */
int ferrompi_initialized(int* flag);

/**
 * Check if MPI is finalized
 * @param flag Output: 1 if finalized, 0 otherwise
 * @return MPI error code
 */
int ferrompi_finalized(int* flag);

/* ============================================================
 * Communicator Operations
 * ============================================================ */

/**
 * Get handle to MPI_COMM_WORLD
 * @return Communicator handle (always 0 for COMM_WORLD)
 */
int32_t ferrompi_comm_world(void);

/**
 * Get rank in communicator
 * @param comm Communicator handle
 * @param rank Output: rank of calling process
 * @return MPI error code
 */
int ferrompi_comm_rank(int32_t comm, int32_t* rank);

/**
 * Get size of communicator
 * @param comm Communicator handle
 * @param size Output: number of processes
 * @return MPI error code
 */
int ferrompi_comm_size(int32_t comm, int32_t* size);

/**
 * Duplicate a communicator
 * @param comm Source communicator handle
 * @param newcomm Output: new communicator handle
 * @return MPI error code
 */
int ferrompi_comm_dup(int32_t comm, int32_t* newcomm);

/**
 * Free a communicator
 * @param comm Communicator handle to free
 * @return MPI error code
 */
int ferrompi_comm_free(int32_t comm);

/* ============================================================
 * Synchronization
 * ============================================================ */

/**
 * Barrier synchronization
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_barrier(int32_t comm);

/* ============================================================
 * Point-to-Point Communication
 * ============================================================ */

/**
 * Blocking send (f64 array)
 * @param buf Data buffer
 * @param count Number of elements
 * @param dest Destination rank
 * @param tag Message tag
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_send_f64(
    const double* buf,
    int64_t count,
    int32_t dest,
    int32_t tag,
    int32_t comm
);

/**
 * Blocking receive (f64 array)
 * @param buf Receive buffer
 * @param count Maximum number of elements
 * @param source Source rank (or -1 for MPI_ANY_SOURCE)
 * @param tag Message tag (or -1 for MPI_ANY_TAG)
 * @param comm Communicator handle
 * @param actual_source Output: actual source rank
 * @param actual_tag Output: actual tag
 * @param actual_count Output: actual count received
 * @return MPI error code
 */
int ferrompi_recv_f64(
    double* buf,
    int64_t count,
    int32_t source,
    int32_t tag,
    int32_t comm,
    int32_t* actual_source,
    int32_t* actual_tag,
    int64_t* actual_count
);

/* ============================================================
 * Collective Operations - Blocking
 * ============================================================ */

/**
 * Broadcast (f64 array)
 * @param buf Data buffer (significant at root)
 * @param count Number of elements
 * @param root Root rank
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_bcast_f64(
    double* buf,
    int64_t count,
    int32_t root,
    int32_t comm
);

/**
 * Broadcast (i32 array)
 */
int ferrompi_bcast_i32(
    int32_t* buf,
    int64_t count,
    int32_t root,
    int32_t comm
);

/**
 * Broadcast (i64 array)
 */
int ferrompi_bcast_i64(
    int64_t* buf,
    int64_t count,
    int32_t root,
    int32_t comm
);

/**
 * Broadcast (raw bytes)
 */
int ferrompi_bcast_bytes(
    void* buf,
    int64_t count,
    int32_t root,
    int32_t comm
);

/**
 * Reduce operation (f64)
 * @param sendbuf Send buffer
 * @param recvbuf Receive buffer (significant at root)
 * @param count Number of elements
 * @param op Operation (0=SUM, 1=MAX, 2=MIN, 3=PROD)
 * @param root Root rank
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_reduce_f64(
    const double* sendbuf,
    double* recvbuf,
    int64_t count,
    int32_t op,
    int32_t root,
    int32_t comm
);

/**
 * All-reduce operation (f64)
 * @param sendbuf Send buffer
 * @param recvbuf Receive buffer
 * @param count Number of elements
 * @param op Operation (0=SUM, 1=MAX, 2=MIN, 3=PROD)
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_allreduce_f64(
    const double* sendbuf,
    double* recvbuf,
    int64_t count,
    int32_t op,
    int32_t comm
);

/**
 * In-place all-reduce operation (f64)
 */
int ferrompi_allreduce_inplace_f64(
    double* buf,
    int64_t count,
    int32_t op,
    int32_t comm
);

/**
 * Gather (f64)
 * @param sendbuf Send buffer
 * @param sendcount Elements to send
 * @param recvbuf Receive buffer (significant at root)
 * @param recvcount Elements to receive from each process
 * @param root Root rank
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_gather_f64(
    const double* sendbuf,
    int64_t sendcount,
    double* recvbuf,
    int64_t recvcount,
    int32_t root,
    int32_t comm
);

/**
 * All-gather (f64)
 */
int ferrompi_allgather_f64(
    const double* sendbuf,
    int64_t sendcount,
    double* recvbuf,
    int64_t recvcount,
    int32_t comm
);

/**
 * Scatter (f64)
 */
int ferrompi_scatter_f64(
    const double* sendbuf,
    int64_t sendcount,
    double* recvbuf,
    int64_t recvcount,
    int32_t root,
    int32_t comm
);

/* ============================================================
 * Collective Operations - Nonblocking
 * ============================================================ */

/**
 * Nonblocking broadcast (f64)
 * @param buf Data buffer
 * @param count Number of elements
 * @param root Root rank
 * @param comm Communicator handle
 * @param request Output: request handle
 * @return MPI error code
 */
int ferrompi_ibcast_f64(
    double* buf,
    int64_t count,
    int32_t root,
    int32_t comm,
    int64_t* request
);

/**
 * Nonblocking all-reduce (f64)
 */
int ferrompi_iallreduce_f64(
    const double* sendbuf,
    double* recvbuf,
    int64_t count,
    int32_t op,
    int32_t comm,
    int64_t* request
);

/* ============================================================
 * Request Management
 * ============================================================ */

/**
 * Wait for request completion
 * @param request Request handle
 * @return MPI error code
 */
int ferrompi_wait(int64_t request);

/**
 * Test if request is complete
 * @param request Request handle
 * @param flag Output: 1 if complete, 0 otherwise
 * @return MPI error code
 */
int ferrompi_test(int64_t request, int32_t* flag);

/**
 * Wait for all requests
 * @param count Number of requests
 * @param requests Array of request handles
 * @return MPI error code
 */
int ferrompi_waitall(int64_t count, int64_t* requests);

/**
 * Free a request handle
 * @param request Request handle to free
 * @return MPI error code
 */
int ferrompi_request_free(int64_t request);

/* ============================================================
 * Persistent Collectives (MPI 4.0+)
 * ============================================================ */

/**
 * Initialize persistent broadcast (f64)
 * @param buf Data buffer
 * @param count Number of elements
 * @param root Root rank
 * @param comm Communicator handle
 * @param request Output: persistent request handle
 * @return MPI error code
 */
int ferrompi_bcast_init_f64(
    double* buf,
    int64_t count,
    int32_t root,
    int32_t comm,
    int64_t* request
);

/**
 * Initialize persistent all-reduce (f64)
 */
int ferrompi_allreduce_init_f64(
    const double* sendbuf,
    double* recvbuf,
    int64_t count,
    int32_t op,
    int32_t comm,
    int64_t* request
);

/**
 * Initialize persistent all-reduce in-place (f64)
 */
int ferrompi_allreduce_init_inplace_f64(
    double* buf,
    int64_t count,
    int32_t op,
    int32_t comm,
    int64_t* request
);

/**
 * Initialize persistent gather (f64)
 */
int ferrompi_gather_init_f64(
    const double* sendbuf,
    int64_t sendcount,
    double* recvbuf,
    int64_t recvcount,
    int32_t root,
    int32_t comm,
    int64_t* request
);

/**
 * Start a persistent request
 * @param request Persistent request handle
 * @return MPI error code
 */
int ferrompi_start(int64_t request);

/**
 * Start multiple persistent requests
 * @param count Number of requests
 * @param requests Array of request handles
 * @return MPI error code
 */
int ferrompi_startall(int64_t count, int64_t* requests);

/* ============================================================
 * Utility Functions
 * ============================================================ */

/**
 * Get MPI library version string
 * @param version Output buffer (at least 256 bytes)
 * @param len Output: actual length
 * @return MPI error code
 */
int ferrompi_get_version(char* version, int32_t* len);

/**
 * Get processor name
 * @param name Output buffer (at least MPI_MAX_PROCESSOR_NAME bytes)
 * @param len Output: actual length
 * @return MPI error code
 */
int ferrompi_get_processor_name(char* name, int32_t* len);

/**
 * Get wall clock time
 * @return Wall clock time in seconds
 */
double ferrompi_wtime(void);

/**
 * Abort MPI execution
 * @param comm Communicator handle
 * @param errorcode Error code to return
 * @return Does not return
 */
int ferrompi_abort(int32_t comm, int32_t errorcode);

#ifdef __cplusplus
}
#endif

#endif /* ferrompi_H */
