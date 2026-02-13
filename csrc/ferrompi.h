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
 * Datatype Tags
 * ============================================================
 * These tags map to MPI_Datatype values in the C layer.
 * They must match the Rust DatatypeTag enum discriminants.
 */

#define FERROMPI_F32  0
#define FERROMPI_F64  1
#define FERROMPI_I32  2
#define FERROMPI_I64  3
#define FERROMPI_U8   4
#define FERROMPI_U32  5
#define FERROMPI_U64  6

/* ============================================================
 * Communicator Split Type Constants
 * ============================================================
 * These constants map to MPI split type values in the C layer.
 * They must match the Rust SplitType enum discriminants.
 */

#define FERROMPI_COMM_TYPE_SHARED 0

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

/**
 * Split a communicator into sub-communicators based on color and key.
 * Processes with the same color are placed in the same new communicator.
 * The key controls rank ordering within the new communicator.
 * Pass color=-1 to opt out (maps to MPI_UNDEFINED); newcomm will be set to -1.
 * @param comm Source communicator handle
 * @param color Color value (sub-communicator identifier, or -1 for MPI_UNDEFINED)
 * @param key Key value (rank ordering control)
 * @param newcomm Output: new communicator handle (-1 if process opted out)
 * @return MPI error code
 */
int ferrompi_comm_split(int32_t comm, int32_t color, int32_t key, int32_t* newcomm);

/**
 * Split a communicator by type (e.g., shared memory).
 * Processes that share the same resource (e.g., same physical node for
 * FERROMPI_COMM_TYPE_SHARED) are placed in the same new communicator.
 * The key controls rank ordering within the new communicator.
 * If the split produces MPI_COMM_NULL for this process, newcomm is set to -1.
 * @param comm Source communicator handle
 * @param split_type Split type constant (FERROMPI_COMM_TYPE_SHARED)
 * @param key Key value (rank ordering control)
 * @param newcomm Output: new communicator handle (-1 if MPI_COMM_NULL)
 * @return MPI error code
 */
int ferrompi_comm_split_type(int32_t comm, int32_t split_type, int32_t key, int32_t* newcomm);

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
 * Generic Point-to-Point Communication
 * ============================================================ */

/**
 * Blocking send (generic)
 * @param buf Data buffer
 * @param count Number of elements
 * @param datatype_tag Datatype tag (FERROMPI_F32, FERROMPI_F64, etc.)
 * @param dest Destination rank
 * @param tag Message tag
 * @param comm Communicator handle
 * @return MPI error code
 */
int ferrompi_send(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm
);

/**
 * Blocking receive (generic)
 * @param buf Receive buffer
 * @param count Maximum number of elements
 * @param datatype_tag Datatype tag (FERROMPI_F32, FERROMPI_F64, etc.)
 * @param source Source rank (or -1 for MPI_ANY_SOURCE)
 * @param tag Message tag (or -1 for MPI_ANY_TAG)
 * @param comm Communicator handle
 * @param actual_source Output: actual source rank
 * @param actual_tag Output: actual tag
 * @param actual_count Output: actual count received
 * @return MPI error code
 */
int ferrompi_recv(
    void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t source,
    int32_t tag,
    int32_t comm,
    int32_t* actual_source,
    int32_t* actual_tag,
    int64_t* actual_count
);

/* ============================================================
 * Generic Collective Operations - Blocking
 * ============================================================ */

/** Broadcast (generic) */
int ferrompi_bcast(void* buf, int64_t count, int32_t datatype_tag, int32_t root, int32_t comm);

/** Reduce (generic) */
int ferrompi_reduce(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t root, int32_t comm);

/** All-reduce (generic) */
int ferrompi_allreduce(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm);

/** In-place all-reduce (generic) */
int ferrompi_allreduce_inplace(void* buf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm);

/** Gather (generic) */
int ferrompi_gather(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm);

/** All-gather (generic) */
int ferrompi_allgather(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t comm);

/** Scatter (generic) */
int ferrompi_scatter(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm);

/* ============================================================
 * Generic Collective Operations - Nonblocking
 * ============================================================ */

/** Nonblocking broadcast (generic) */
int ferrompi_ibcast(void* buf, int64_t count, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

/** Nonblocking all-reduce (generic) */
int ferrompi_iallreduce(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/* ============================================================
 * Generic Persistent Collectives (MPI 4.0+)
 * ============================================================ */

/** Initialize persistent broadcast (generic) */
int ferrompi_bcast_init(void* buf, int64_t count, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

/** Initialize persistent all-reduce (generic) */
int ferrompi_allreduce_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/** Initialize persistent all-reduce in-place (generic) */
int ferrompi_allreduce_init_inplace(void* buf, int64_t count, int32_t datatype_tag, int32_t op, int32_t comm, int64_t* request);

/** Initialize persistent gather (generic) */
int ferrompi_gather_init(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount, int32_t datatype_tag, int32_t root, int32_t comm, int64_t* request);

/* ============================================================
 * Error Information
 * ============================================================ */

/**
 * Get error class and message string for an MPI error code
 * @param code MPI error code
 * @param error_class Output: error class
 * @param message Output: error message (at least MPI_MAX_ERROR_STRING bytes)
 * @param msg_len Output: actual message length
 * @return MPI error code
 */
int ferrompi_error_info(int code, int32_t* error_class, char* message, int32_t* msg_len);

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
 * RMA Window Operations (to be implemented in tickets 015-018)
 * ============================================================
 *
 * Window handle table supports up to 256 concurrent MPI_Win objects.
 * Functions will be added for:
 * - MPI_Win_allocate_shared
 * - MPI_Win_shared_query
 * - MPI_Win_fence / MPI_Win_lock / MPI_Win_lock_all
 * - MPI_Win_unlock / MPI_Win_unlock_all
 * - MPI_Win_free
 */

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
