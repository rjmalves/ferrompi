/**
 * ferrompi.c - Thin C wrapper for MPI 4.x features
 * 
 * Implementation of the FFI layer for Rust interop.
 */

#include "ferrompi.h"
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <stdio.h>

/* ============================================================
 * Internal State and Handle Management
 * ============================================================ */

// Maximum number of concurrent communicators (excluding COMM_WORLD)
#define MAX_COMMS 64

// Maximum number of concurrent requests
#define MAX_REQUESTS 4096

// Communicator table (index 0 is always COMM_WORLD)
static MPI_Comm comm_table[MAX_COMMS];
static int comm_count = 1;  // Start at 1, 0 is COMM_WORLD

// Request table
static MPI_Request request_table[MAX_REQUESTS];
static int request_used[MAX_REQUESTS];  // 1 if slot is in use
static int next_request_hint = 0;

// Initialize tables
static int tables_initialized = 0;

static void init_tables(void) {
    if (tables_initialized) return;
    
    // Note: MPI_COMM_WORLD must be set AFTER MPI_Init is called
    // comm_table[0] will be set in ferrompi_init* functions
    for (int i = 0; i < MAX_COMMS; i++) {
        comm_table[i] = MPI_COMM_NULL;
    }
    for (int i = 0; i < MAX_REQUESTS; i++) {
        request_table[i] = MPI_REQUEST_NULL;
        request_used[i] = 0;
    }
    tables_initialized = 1;
}

// Get MPI_Comm from handle
static MPI_Comm get_comm(int32_t handle) {
    if (handle < 0 || handle >= MAX_COMMS) {
        return MPI_COMM_NULL;
    }
    return comm_table[handle];
}

// Allocate a communicator handle
static int32_t alloc_comm(MPI_Comm comm) {
    for (int i = 1; i < MAX_COMMS; i++) {
        if (comm_table[i] == MPI_COMM_NULL) {
            comm_table[i] = comm;
            if (i >= comm_count) comm_count = i + 1;
            return i;
        }
    }
    return -1;  // No space
}

// Allocate a request handle
static int64_t alloc_request(MPI_Request req) {
    // Start searching from hint for O(1) common case
    for (int i = 0; i < MAX_REQUESTS; i++) {
        int idx = (next_request_hint + i) % MAX_REQUESTS;
        if (!request_used[idx]) {
            request_table[idx] = req;
            request_used[idx] = 1;
            next_request_hint = (idx + 1) % MAX_REQUESTS;
            return (int64_t)idx;
        }
    }
    return -1;  // No space
}

// Get MPI_Request pointer from handle
static MPI_Request* get_request_ptr(int64_t handle) {
    if (handle < 0 || handle >= MAX_REQUESTS || !request_used[handle]) {
        return NULL;
    }
    return &request_table[handle];
}

// Free a request handle
static void free_request(int64_t handle) {
    if (handle >= 0 && handle < MAX_REQUESTS) {
        request_table[handle] = MPI_REQUEST_NULL;
        request_used[handle] = 0;
    }
}

// Map operation code to MPI_Op
static MPI_Op get_op(int32_t op) {
    switch (op) {
        case 0: return MPI_SUM;
        case 1: return MPI_MAX;
        case 2: return MPI_MIN;
        case 3: return MPI_PROD;
        default: return MPI_SUM;
    }
}

/* ============================================================
 * Initialization and Finalization
 * ============================================================ */

int ferrompi_init_thread(int required, int* provided) {
    init_tables();
    int mpi_required;
    switch (required) {
        case 0: mpi_required = MPI_THREAD_SINGLE; break;
        case 1: mpi_required = MPI_THREAD_FUNNELED; break;
        case 2: mpi_required = MPI_THREAD_SERIALIZED; break;
        case 3: mpi_required = MPI_THREAD_MULTIPLE; break;
        default: mpi_required = MPI_THREAD_SINGLE; break;
    }
    
    int mpi_provided;
    int ret = MPI_Init_thread(NULL, NULL, mpi_required, &mpi_provided);
    
    // Initialize COMM_WORLD after MPI_Init
    if (ret == MPI_SUCCESS) {
        comm_table[0] = MPI_COMM_WORLD;
    }
    
    if (provided) {
        switch (mpi_provided) {
            case MPI_THREAD_SINGLE: *provided = 0; break;
            case MPI_THREAD_FUNNELED: *provided = 1; break;
            case MPI_THREAD_SERIALIZED: *provided = 2; break;
            case MPI_THREAD_MULTIPLE: *provided = 3; break;
            default: *provided = 0; break;
        }
    }
    
    return ret;
}

int ferrompi_init(void) {
    init_tables();
    int ret = MPI_Init(NULL, NULL);
    
    // Initialize COMM_WORLD after MPI_Init
    if (ret == MPI_SUCCESS) {
        comm_table[0] = MPI_COMM_WORLD;
    }
    
    return ret;
}

int ferrompi_finalize(void) {
    // Clean up any remaining communicators
    for (int i = 1; i < comm_count; i++) {
        if (comm_table[i] != MPI_COMM_NULL) {
            MPI_Comm_free(&comm_table[i]);
            comm_table[i] = MPI_COMM_NULL;
        }
    }
    
    // Clean up any remaining requests
    for (int i = 0; i < MAX_REQUESTS; i++) {
        if (request_used[i] && request_table[i] != MPI_REQUEST_NULL) {
            MPI_Request_free(&request_table[i]);
        }
        request_used[i] = 0;
    }
    
    tables_initialized = 0;
    return MPI_Finalize();
}

int ferrompi_initialized(int* flag) {
    return MPI_Initialized(flag);
}

int ferrompi_finalized(int* flag) {
    return MPI_Finalized(flag);
}

/* ============================================================
 * Communicator Operations
 * ============================================================ */

int32_t ferrompi_comm_world(void) {
    return 0;  // COMM_WORLD is always handle 0
}

int ferrompi_comm_rank(int32_t comm_handle, int32_t* rank) {
    MPI_Comm comm = get_comm(comm_handle);
    int r;
    int ret = MPI_Comm_rank(comm, &r);
    *rank = (int32_t)r;
    return ret;
}

int ferrompi_comm_size(int32_t comm_handle, int32_t* size) {
    MPI_Comm comm = get_comm(comm_handle);
    int s;
    int ret = MPI_Comm_size(comm, &s);
    *size = (int32_t)s;
    return ret;
}

int ferrompi_comm_dup(int32_t comm_handle, int32_t* newcomm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Comm newcomm;
    int ret = MPI_Comm_dup(comm, &newcomm);
    if (ret == MPI_SUCCESS) {
        *newcomm_handle = alloc_comm(newcomm);
        if (*newcomm_handle < 0) {
            MPI_Comm_free(&newcomm);
            return MPI_ERR_OTHER;
        }
    }
    return ret;
}

int ferrompi_comm_free(int32_t comm_handle) {
    if (comm_handle == 0) {
        return MPI_ERR_COMM;  // Cannot free COMM_WORLD
    }
    if (comm_handle < 0 || comm_handle >= MAX_COMMS) {
        return MPI_ERR_COMM;
    }
    if (comm_table[comm_handle] == MPI_COMM_NULL) {
        return MPI_SUCCESS;  // Already freed
    }
    int ret = MPI_Comm_free(&comm_table[comm_handle]);
    comm_table[comm_handle] = MPI_COMM_NULL;
    return ret;
}

/* ============================================================
 * Synchronization
 * ============================================================ */

int ferrompi_barrier(int32_t comm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    return MPI_Barrier(comm);
}

/* ============================================================
 * Point-to-Point Communication
 * ============================================================ */

int ferrompi_send_f64(
    const double* buf,
    int64_t count,
    int32_t dest,
    int32_t tag,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    // Use large-count version if needed
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        return MPI_Send_c(buf, (MPI_Count)count, MPI_DOUBLE, dest, tag, comm);
    }
#endif
    return MPI_Send(buf, (int)count, MPI_DOUBLE, dest, tag, comm);
}

int ferrompi_recv_f64(
    double* buf,
    int64_t count,
    int32_t source,
    int32_t tag,
    int32_t comm_handle,
    int32_t* actual_source,
    int32_t* actual_tag,
    int64_t* actual_count
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Status status;
    
    int mpi_source = (source == -1) ? MPI_ANY_SOURCE : source;
    int mpi_tag = (tag == -1) ? MPI_ANY_TAG : tag;
    
    int ret;
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        ret = MPI_Recv_c(buf, (MPI_Count)count, MPI_DOUBLE, mpi_source, mpi_tag, comm, &status);
    } else
#endif
    {
        ret = MPI_Recv(buf, (int)count, MPI_DOUBLE, mpi_source, mpi_tag, comm, &status);
    }
    
    if (ret == MPI_SUCCESS) {
        *actual_source = status.MPI_SOURCE;
        *actual_tag = status.MPI_TAG;
        int cnt;
        MPI_Get_count(&status, MPI_DOUBLE, &cnt);
        *actual_count = (int64_t)cnt;
    }
    
    return ret;
}

/* ============================================================
 * Collective Operations - Blocking
 * ============================================================ */

int ferrompi_bcast_f64(double* buf, int64_t count, int32_t root, int32_t comm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        return MPI_Bcast_c(buf, (MPI_Count)count, MPI_DOUBLE, root, comm);
    }
#endif
    return MPI_Bcast(buf, (int)count, MPI_DOUBLE, root, comm);
}

int ferrompi_bcast_i32(int32_t* buf, int64_t count, int32_t root, int32_t comm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        return MPI_Bcast_c(buf, (MPI_Count)count, MPI_INT32_T, root, comm);
    }
#endif
    return MPI_Bcast(buf, (int)count, MPI_INT32_T, root, comm);
}

int ferrompi_bcast_i64(int64_t* buf, int64_t count, int32_t root, int32_t comm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        return MPI_Bcast_c(buf, (MPI_Count)count, MPI_INT64_T, root, comm);
    }
#endif
    return MPI_Bcast(buf, (int)count, MPI_INT64_T, root, comm);
}

int ferrompi_bcast_bytes(void* buf, int64_t count, int32_t root, int32_t comm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        return MPI_Bcast_c(buf, (MPI_Count)count, MPI_BYTE, root, comm);
    }
#endif
    return MPI_Bcast(buf, (int)count, MPI_BYTE, root, comm);
}

int ferrompi_reduce_f64(
    const double* sendbuf,
    double* recvbuf,
    int64_t count,
    int32_t op,
    int32_t root,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Op mpi_op = get_op(op);
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        return MPI_Reduce_c(sendbuf, recvbuf, (MPI_Count)count, MPI_DOUBLE, mpi_op, root, comm);
    }
#endif
    return MPI_Reduce(sendbuf, recvbuf, (int)count, MPI_DOUBLE, mpi_op, root, comm);
}

int ferrompi_allreduce_f64(
    const double* sendbuf,
    double* recvbuf,
    int64_t count,
    int32_t op,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Op mpi_op = get_op(op);
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        return MPI_Allreduce_c(sendbuf, recvbuf, (MPI_Count)count, MPI_DOUBLE, mpi_op, comm);
    }
#endif
    return MPI_Allreduce(sendbuf, recvbuf, (int)count, MPI_DOUBLE, mpi_op, comm);
}

int ferrompi_allreduce_inplace_f64(double* buf, int64_t count, int32_t op, int32_t comm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Op mpi_op = get_op(op);
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        return MPI_Allreduce_c(MPI_IN_PLACE, buf, (MPI_Count)count, MPI_DOUBLE, mpi_op, comm);
    }
#endif
    return MPI_Allreduce(MPI_IN_PLACE, buf, (int)count, MPI_DOUBLE, mpi_op, comm);
}

int ferrompi_gather_f64(
    const double* sendbuf,
    int64_t sendcount,
    double* recvbuf,
    int64_t recvcount,
    int32_t root,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
#if MPI_VERSION >= 4
    if (sendcount > INT_MAX || recvcount > INT_MAX) {
        return MPI_Gather_c(sendbuf, (MPI_Count)sendcount, MPI_DOUBLE,
                           recvbuf, (MPI_Count)recvcount, MPI_DOUBLE,
                           root, comm);
    }
#endif
    return MPI_Gather(sendbuf, (int)sendcount, MPI_DOUBLE,
                      recvbuf, (int)recvcount, MPI_DOUBLE,
                      root, comm);
}

int ferrompi_allgather_f64(
    const double* sendbuf,
    int64_t sendcount,
    double* recvbuf,
    int64_t recvcount,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
#if MPI_VERSION >= 4
    if (sendcount > INT_MAX || recvcount > INT_MAX) {
        return MPI_Allgather_c(sendbuf, (MPI_Count)sendcount, MPI_DOUBLE,
                               recvbuf, (MPI_Count)recvcount, MPI_DOUBLE,
                               comm);
    }
#endif
    return MPI_Allgather(sendbuf, (int)sendcount, MPI_DOUBLE,
                         recvbuf, (int)recvcount, MPI_DOUBLE,
                         comm);
}

int ferrompi_scatter_f64(
    const double* sendbuf,
    int64_t sendcount,
    double* recvbuf,
    int64_t recvcount,
    int32_t root,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
#if MPI_VERSION >= 4
    if (sendcount > INT_MAX || recvcount > INT_MAX) {
        return MPI_Scatter_c(sendbuf, (MPI_Count)sendcount, MPI_DOUBLE,
                            recvbuf, (MPI_Count)recvcount, MPI_DOUBLE,
                            root, comm);
    }
#endif
    return MPI_Scatter(sendbuf, (int)sendcount, MPI_DOUBLE,
                       recvbuf, (int)recvcount, MPI_DOUBLE,
                       root, comm);
}

/* ============================================================
 * Collective Operations - Nonblocking
 * ============================================================ */

int ferrompi_ibcast_f64(
    double* buf,
    int64_t count,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Request req;
    int ret;
    
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        ret = MPI_Ibcast_c(buf, (MPI_Count)count, MPI_DOUBLE, root, comm, &req);
    } else
#endif
    {
        ret = MPI_Ibcast(buf, (int)count, MPI_DOUBLE, root, comm, &req);
    }
    
    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }
    
    return ret;
}

int ferrompi_iallreduce_f64(
    const double* sendbuf,
    double* recvbuf,
    int64_t count,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    int ret;
    
#if MPI_VERSION >= 4
    if (count > INT_MAX) {
        ret = MPI_Iallreduce_c(sendbuf, recvbuf, (MPI_Count)count, MPI_DOUBLE, mpi_op, comm, &req);
    } else
#endif
    {
        ret = MPI_Iallreduce(sendbuf, recvbuf, (int)count, MPI_DOUBLE, mpi_op, comm, &req);
    }
    
    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }
    
    return ret;
}

/* ============================================================
 * Request Management
 * ============================================================ */

int ferrompi_wait(int64_t request_handle) {
    MPI_Request* req = get_request_ptr(request_handle);
    if (!req) {
        return MPI_ERR_REQUEST;
    }
    int ret = MPI_Wait(req, MPI_STATUS_IGNORE);
    // Don't free persistent requests automatically
    if (*req == MPI_REQUEST_NULL) {
        free_request(request_handle);
    }
    return ret;
}

int ferrompi_test(int64_t request_handle, int32_t* flag) {
    MPI_Request* req = get_request_ptr(request_handle);
    if (!req) {
        return MPI_ERR_REQUEST;
    }
    int f;
    int ret = MPI_Test(req, &f, MPI_STATUS_IGNORE);
    *flag = f;
    if (f && *req == MPI_REQUEST_NULL) {
        free_request(request_handle);
    }
    return ret;
}

int ferrompi_waitall(int64_t count, int64_t* request_handles) {
    if (count <= 0) return MPI_SUCCESS;
    
    // Allocate temporary array of MPI_Request pointers
    MPI_Request* reqs = (MPI_Request*)malloc(count * sizeof(MPI_Request));
    if (!reqs) return MPI_ERR_OTHER;
    
    for (int64_t i = 0; i < count; i++) {
        MPI_Request* req = get_request_ptr(request_handles[i]);
        if (!req) {
            free(reqs);
            return MPI_ERR_REQUEST;
        }
        reqs[i] = *req;
    }
    
    int ret = MPI_Waitall((int)count, reqs, MPI_STATUSES_IGNORE);
    
    // Update handles and free completed non-persistent requests
    for (int64_t i = 0; i < count; i++) {
        MPI_Request* req = get_request_ptr(request_handles[i]);
        if (req) {
            *req = reqs[i];
            if (*req == MPI_REQUEST_NULL) {
                free_request(request_handles[i]);
            }
        }
    }
    
    free(reqs);
    return ret;
}

int ferrompi_request_free(int64_t request_handle) {
    MPI_Request* req = get_request_ptr(request_handle);
    if (!req) {
        return MPI_SUCCESS;  // Already freed
    }
    if (*req != MPI_REQUEST_NULL) {
        int ret = MPI_Request_free(req);
        if (ret != MPI_SUCCESS) return ret;
    }
    free_request(request_handle);
    return MPI_SUCCESS;
}

/* ============================================================
 * Persistent Collectives (MPI 4.0+)
 * ============================================================ */

#if MPI_VERSION >= 4

int ferrompi_bcast_init_f64(
    double* buf,
    int64_t count,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Request req;
    int ret;
    
    if (count > INT_MAX) {
        // Large count version - need MPI_Bcast_init_c if available
        // For now, fall back to regular count (may fail for very large)
        ret = MPI_Bcast_init(buf, (int)count, MPI_DOUBLE, root, comm, MPI_INFO_NULL, &req);
    } else {
        ret = MPI_Bcast_init(buf, (int)count, MPI_DOUBLE, root, comm, MPI_INFO_NULL, &req);
    }
    
    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }
    
    return ret;
}

int ferrompi_allreduce_init_f64(
    const double* sendbuf,
    double* recvbuf,
    int64_t count,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    
    int ret = MPI_Allreduce_init(sendbuf, recvbuf, (int)count, MPI_DOUBLE, 
                                  mpi_op, comm, MPI_INFO_NULL, &req);
    
    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }
    
    return ret;
}

int ferrompi_allreduce_init_inplace_f64(
    double* buf,
    int64_t count,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    
    int ret = MPI_Allreduce_init(MPI_IN_PLACE, buf, (int)count, MPI_DOUBLE,
                                  mpi_op, comm, MPI_INFO_NULL, &req);
    
    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }
    
    return ret;
}

int ferrompi_gather_init_f64(
    const double* sendbuf,
    int64_t sendcount,
    double* recvbuf,
    int64_t recvcount,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Request req;
    
    int ret = MPI_Gather_init(sendbuf, (int)sendcount, MPI_DOUBLE,
                              recvbuf, (int)recvcount, MPI_DOUBLE,
                              root, comm, MPI_INFO_NULL, &req);
    
    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return MPI_ERR_OTHER;
        }
    }
    
    return ret;
}

int ferrompi_start(int64_t request_handle) {
    MPI_Request* req = get_request_ptr(request_handle);
    if (!req) {
        return MPI_ERR_REQUEST;
    }
    return MPI_Start(req);
}

int ferrompi_startall(int64_t count, int64_t* request_handles) {
    if (count <= 0) return MPI_SUCCESS;
    
    MPI_Request* reqs = (MPI_Request*)malloc(count * sizeof(MPI_Request));
    if (!reqs) return MPI_ERR_OTHER;
    
    for (int64_t i = 0; i < count; i++) {
        MPI_Request* req = get_request_ptr(request_handles[i]);
        if (!req) {
            free(reqs);
            return MPI_ERR_REQUEST;
        }
        reqs[i] = *req;
    }
    
    int ret = MPI_Startall((int)count, reqs);
    
    // Copy back the modified requests
    for (int64_t i = 0; i < count; i++) {
        MPI_Request* req = get_request_ptr(request_handles[i]);
        if (req) {
            *req = reqs[i];
        }
    }
    
    free(reqs);
    return ret;
}

#else /* MPI_VERSION < 4 */

// Stub implementations for MPI < 4.0
int ferrompi_bcast_init_f64(double* buf, int64_t count, int32_t root, 
                               int32_t comm_handle, int64_t* request_handle) {
    (void)buf; (void)count; (void)root; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;  // Not supported
}

int ferrompi_allreduce_init_f64(const double* sendbuf, double* recvbuf, int64_t count,
                                   int32_t op, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)recvbuf; (void)count; (void)op; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_allreduce_init_inplace_f64(double* buf, int64_t count, int32_t op,
                                           int32_t comm_handle, int64_t* request_handle) {
    (void)buf; (void)count; (void)op; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_gather_init_f64(const double* sendbuf, int64_t sendcount, double* recvbuf,
                                int64_t recvcount, int32_t root, int32_t comm_handle,
                                int64_t* request_handle) {
    (void)sendbuf; (void)sendcount; (void)recvbuf; (void)recvcount;
    (void)root; (void)comm_handle; (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_start(int64_t request_handle) {
    (void)request_handle;
    return MPI_ERR_OTHER;
}

int ferrompi_startall(int64_t count, int64_t* request_handles) {
    (void)count; (void)request_handles;
    return MPI_ERR_OTHER;
}

#endif /* MPI_VERSION >= 4 */

/* ============================================================
 * Utility Functions
 * ============================================================ */

int ferrompi_get_version(char* version, int32_t* len) {
    int version_num, subversion_num;
    int ret = MPI_Get_version(&version_num, &subversion_num);
    if (ret == MPI_SUCCESS) {
        *len = snprintf(version, 256, "MPI %d.%d", version_num, subversion_num);
    }
    return ret;
}

int ferrompi_get_processor_name(char* name, int32_t* len) {
    int l;
    int ret = MPI_Get_processor_name(name, &l);
    *len = l;
    return ret;
}

double ferrompi_wtime(void) {
    return MPI_Wtime();
}

int ferrompi_abort(int32_t comm_handle, int32_t errorcode) {
    MPI_Comm comm = get_comm(comm_handle);
    return MPI_Abort(comm, errorcode);
}
