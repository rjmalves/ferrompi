//! MPI message status information.
//!
//! This module provides the [`Status`] struct returned by probe operations,
//! containing metadata about a message without actually receiving it.

/// Information about a probed or received MPI message.
///
/// Returned by [`Communicator::probe`](crate::Communicator::probe) and
/// [`Communicator::iprobe`](crate::Communicator::iprobe) to describe an
/// incoming message without consuming it.
///
/// # Example
///
/// ```no_run
/// # use ferrompi::Mpi;
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
///
/// // Blocking probe for any f64 message
/// let status = world.probe::<f64>(-1, -1).unwrap();
/// println!("Message from rank {} with tag {}, {} elements",
///          status.source, status.tag, status.count);
/// ```
#[derive(Debug, Clone)]
pub struct Status {
    /// Source rank of the message.
    pub source: i32,
    /// Tag of the message.
    pub tag: i32,
    /// Number of elements in the message (determined via `MPI_Get_count`).
    pub count: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_clone() {
        let status = Status {
            source: 3,
            tag: 42,
            count: 100,
        };
        let cloned = status.clone();
        assert_eq!(cloned.source, 3);
        assert_eq!(cloned.tag, 42);
        assert_eq!(cloned.count, 100);
    }

    #[test]
    fn status_debug() {
        let status = Status {
            source: 0,
            tag: 1,
            count: 50,
        };
        let debug = format!("{:?}", status);
        assert!(debug.contains("source"));
        assert!(debug.contains("tag"));
        assert!(debug.contains("count"));
    }
}
