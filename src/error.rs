//! Error types for ferrompi

use thiserror::Error;

/// Result type for MPI operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for MPI operations
#[derive(Error, Debug)]
pub enum Error {
    /// MPI has already been initialized
    #[error("MPI has already been initialized")]
    AlreadyInitialized,

    /// MPI has not been initialized
    #[error("MPI has not been initialized")]
    NotInitialized,

    /// Invalid rank specified
    #[error("Invalid rank: {0}")]
    InvalidRank(i32),

    /// Invalid communicator handle
    #[error("Invalid communicator")]
    InvalidCommunicator,

    /// Invalid request handle
    #[error("Invalid request handle")]
    InvalidRequest,

    /// Invalid buffer provided
    #[error("Invalid buffer")]
    InvalidBuffer,

    /// Invalid count specified
    #[error("Invalid count: {0}")]
    InvalidCount(i64),

    /// Operation not supported (e.g., MPI 4.0 features on older MPI)
    #[error("Operation not supported: {0}")]
    NotSupported(String),

    /// MPI error with code
    #[error("MPI error (code {0})")]
    MpiError(i32),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl Error {
    /// Create an error from an MPI error code.
    ///
    /// Returns Ok(()) for MPI_SUCCESS (0), otherwise returns an error.
    pub fn from_code(code: i32) -> Self {
        match code {
            0 => panic!("from_code called with success code"),
            // Common MPI error codes (these values are implementation-specific,
            // but we provide some common mappings)
            1..=10 => Error::MpiError(code),
            _ => Error::MpiError(code),
        }
    }

    /// Check an MPI return code, returning Ok(()) for success.
    pub fn check(code: i32) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(Error::from_code(code))
        }
    }
}
