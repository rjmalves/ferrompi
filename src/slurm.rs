//! SLURM scheduler environment helpers.
//!
//! These functions read SLURM environment variables to determine job topology.
//! They return `None` if the variable is not set (e.g., when not running under SLURM).
//!
//! # Environment Variables
//!
//! | Function | Variable | Description |
//! |----------|----------|-------------|
//! | `job_id()` | `SLURM_JOB_ID` | Unique job identifier |
//! | `local_rank()` | `SLURM_LOCALID` | Task ID relative to this node |
//! | `local_size()` | `SLURM_NTASKS_PER_NODE` | Number of tasks on this node |
//! | `num_nodes()` | `SLURM_NNODES` | Total number of nodes |
//! | `cpus_per_task()` | `SLURM_CPUS_PER_TASK` | CPUs allocated per task |
//! | `node_name()` | `SLURM_NODENAME` | Name of this compute node |
//! | `node_list()` | `SLURM_NODELIST` | Compact list of allocated nodes |

use std::env;

/// Check if running under SLURM job scheduler.
pub fn is_slurm_job() -> bool {
    env::var("SLURM_JOB_ID").is_ok()
}

/// Get the SLURM job ID.
pub fn job_id() -> Option<String> {
    env::var("SLURM_JOB_ID").ok()
}

/// Get the local (intra-node) rank of this process.
pub fn local_rank() -> Option<i32> {
    env::var("SLURM_LOCALID").ok().and_then(|s| s.parse().ok())
}

/// Get the number of tasks per node.
pub fn local_size() -> Option<i32> {
    env::var("SLURM_NTASKS_PER_NODE")
        .ok()
        .and_then(|s| s.parse().ok())
        .or_else(|| {
            // Fallback: parse first entry of SLURM_TASKS_PER_NODE (format: "4(x2)")
            env::var("SLURM_TASKS_PER_NODE")
                .ok()
                .and_then(|s| s.split('(').next().and_then(|n| n.parse().ok()))
        })
}

/// Get the total number of nodes allocated.
pub fn num_nodes() -> Option<i32> {
    env::var("SLURM_NNODES").ok().and_then(|s| s.parse().ok())
}

/// Get the number of CPUs per task.
pub fn cpus_per_task() -> Option<i32> {
    env::var("SLURM_CPUS_PER_TASK")
        .ok()
        .and_then(|s| s.parse().ok())
}

/// Get the name of the compute node this process is running on.
pub fn node_name() -> Option<String> {
    env::var("SLURM_NODENAME")
        .or_else(|_| env::var("SLURMD_NODENAME"))
        .ok()
}

/// Get the compact node list string (e.g., "node[001-004]").
pub fn node_list() -> Option<String> {
    env::var("SLURM_NODELIST").ok()
}
