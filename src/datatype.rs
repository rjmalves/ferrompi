//! MPI datatype trait and type tag mapping.
//!
//! This module provides the [`MpiDatatype`] trait, a sealed trait that maps Rust
//! primitive types to MPI datatype tags for use in generic communication operations.
//!
//! # Supported Types
//!
//! | Rust Type | MPI Equivalent     | Tag Value |
//! |-----------|-------------------|-----------|
//! | `f32`     | `MPI_FLOAT`       | 0         |
//! | `f64`     | `MPI_DOUBLE`      | 1         |
//! | `i32`     | `MPI_INT32_T`     | 2         |
//! | `i64`     | `MPI_INT64_T`     | 3         |
//! | `u8`      | `MPI_UINT8_T`     | 4         |
//! | `u32`     | `MPI_UINT32_T`    | 5         |
//! | `u64`     | `MPI_UINT64_T`    | 6         |

/// Internal module to seal the trait — prevents external implementations.
mod sealed {
    pub trait Sealed {}
}

/// Tag values matching C-side `FERROMPI_*` defines.
///
/// These discriminants must stay in sync with the `#define FERROMPI_*` values
/// in `csrc/ferrompi.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum DatatypeTag {
    /// 32-bit floating point (`MPI_FLOAT`)
    F32 = 0,
    /// 64-bit floating point (`MPI_DOUBLE`)
    F64 = 1,
    /// 32-bit signed integer (`MPI_INT32_T`)
    I32 = 2,
    /// 64-bit signed integer (`MPI_INT64_T`)
    I64 = 3,
    /// 8-bit unsigned integer (`MPI_UINT8_T`)
    U8 = 4,
    /// 32-bit unsigned integer (`MPI_UINT32_T`)
    U32 = 5,
    /// 64-bit unsigned integer (`MPI_UINT64_T`)
    U64 = 6,
}

/// Trait for types that can be used in MPI communication operations.
///
/// This is a **sealed trait** — it cannot be implemented outside this crate.
/// Supported types: [`f32`], [`f64`], [`i32`], [`i64`], [`u8`], [`u32`], [`u64`].
///
/// # Example
///
/// ```no_run
/// use ferrompi::{Mpi, MpiDatatype};
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
///
/// // Works with f64
/// let mut data_f64 = vec![1.0f64; 10];
/// world.broadcast(&mut data_f64, 0).unwrap();
///
/// // Works with i32
/// let mut data_i32 = vec![42i32; 10];
/// world.broadcast(&mut data_i32, 0).unwrap();
/// ```
pub trait MpiDatatype: sealed::Sealed + Copy + Send + 'static {
    /// The datatype tag used for FFI dispatch to the C layer.
    const TAG: DatatypeTag;
}

macro_rules! impl_mpi_datatype {
    ($ty:ty, $tag:expr) => {
        impl sealed::Sealed for $ty {}
        impl MpiDatatype for $ty {
            const TAG: DatatypeTag = $tag;
        }
    };
}

impl_mpi_datatype!(f32, DatatypeTag::F32);
impl_mpi_datatype!(f64, DatatypeTag::F64);
impl_mpi_datatype!(i32, DatatypeTag::I32);
impl_mpi_datatype!(i64, DatatypeTag::I64);
impl_mpi_datatype!(u8, DatatypeTag::U8);
impl_mpi_datatype!(u32, DatatypeTag::U32);
impl_mpi_datatype!(u64, DatatypeTag::U64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_values_match_c_defines() {
        assert_eq!(DatatypeTag::F32 as i32, 0);
        assert_eq!(DatatypeTag::F64 as i32, 1);
        assert_eq!(DatatypeTag::I32 as i32, 2);
        assert_eq!(DatatypeTag::I64 as i32, 3);
        assert_eq!(DatatypeTag::U8 as i32, 4);
        assert_eq!(DatatypeTag::U32 as i32, 5);
        assert_eq!(DatatypeTag::U64 as i32, 6);
    }

    #[test]
    fn datatype_tags_match_c_defines() {
        // Verify each Rust type's TAG constant maps to the correct C-side define
        assert_eq!(f32::TAG as i32, 0); // FERROMPI_F32
        assert_eq!(f64::TAG as i32, 1); // FERROMPI_F64
        assert_eq!(i32::TAG as i32, 2); // FERROMPI_I32
        assert_eq!(i64::TAG as i32, 3); // FERROMPI_I64
        assert_eq!(u8::TAG as i32, 4); // FERROMPI_U8
        assert_eq!(u32::TAG as i32, 5); // FERROMPI_U32
        assert_eq!(u64::TAG as i32, 6); // FERROMPI_U64
    }

    #[test]
    fn datatype_tag_values_are_sequential() {
        let tags = [
            DatatypeTag::F32,
            DatatypeTag::F64,
            DatatypeTag::I32,
            DatatypeTag::I64,
            DatatypeTag::U8,
            DatatypeTag::U32,
            DatatypeTag::U64,
        ];
        for (i, tag) in tags.iter().enumerate() {
            assert_eq!(*tag as i32, i as i32, "Tag {tag:?} should have value {i}");
        }
    }

    #[test]
    fn trait_is_implemented() {
        // Compile-time check that all types implement MpiDatatype
        fn assert_mpi_datatype<T: MpiDatatype>() {}
        assert_mpi_datatype::<f32>();
        assert_mpi_datatype::<f64>();
        assert_mpi_datatype::<i32>();
        assert_mpi_datatype::<i64>();
        assert_mpi_datatype::<u8>();
        assert_mpi_datatype::<u32>();
        assert_mpi_datatype::<u64>();
    }

    #[test]
    fn datatype_tag_debug_format() {
        assert_eq!(format!("{:?}", DatatypeTag::F32), "F32");
        assert_eq!(format!("{:?}", DatatypeTag::F64), "F64");
        assert_eq!(format!("{:?}", DatatypeTag::I32), "I32");
        assert_eq!(format!("{:?}", DatatypeTag::I64), "I64");
        assert_eq!(format!("{:?}", DatatypeTag::U8), "U8");
        assert_eq!(format!("{:?}", DatatypeTag::U32), "U32");
        assert_eq!(format!("{:?}", DatatypeTag::U64), "U64");
    }

    #[test]
    fn datatype_tag_clone_hash() {
        use std::collections::HashSet;
        let tag = DatatypeTag::F64;
        let cloned = tag;
        assert_eq!(cloned, DatatypeTag::F64);

        let mut set = HashSet::new();
        set.insert(DatatypeTag::F32);
        set.insert(DatatypeTag::F64);
        set.insert(DatatypeTag::F32); // duplicate
        assert_eq!(set.len(), 2);
    }
}
