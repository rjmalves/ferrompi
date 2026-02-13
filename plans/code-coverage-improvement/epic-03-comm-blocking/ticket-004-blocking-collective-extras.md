# Ticket 004: Blocking Collective Extras — Missing Operations

## Summary

Review the existing `test_collectives.rs` against all blocking collective methods in `comm.rs` and add tests for any methods not already covered. The existing test covers 25 operations but some may be missing.

## Dependencies

- ticket-001 (coverage runner script)
- ticket-002 (baseline verification — to know what's already covered)

## Acceptance Criteria

- [ ] All blocking collective methods in `comm.rs` are exercised by at least one test example
- [ ] Methods checked: broadcast, reduce, reduce_scalar, reduce_inplace, allreduce, allreduce_inplace, allreduce_scalar, gather, scatter, allgather, alltoall, scan, exscan, scan_scalar, exscan_scalar, reduce_scatter_block, gatherv, scatterv, allgatherv, alltoallv, barrier
- [ ] Any missing methods are tested in either the existing `test_collectives.rs` or a new `test_blocking_extra.rs`
- [ ] Tests verify correct results with assertions (not just "doesn't crash")
- [ ] Tests use multiple data types (at least f64 and i32) to cover the generic instantiation

## Files to Create/Modify

- **Create** (if needed): `examples/test_blocking_extra.rs`
- **Modify** (if needed): `Cargo.toml` (add example entry)
- **Modify**: `tests/run_mpi_coverage.sh` (add to test list)
- **Modify**: `tests/run_mpi_tests.sh` (add to test list)

## Technical Details

### Step 1: Audit existing coverage

Cross-reference the 25 tests in `test_collectives.rs` against the full method list in `comm.rs`:

**Already covered by test_collectives.rs:**

- broadcast (f64, i32)
- allreduce (Sum)
- gather
- scatter
- barrier
- reduce (Sum, Max, non-zero root)
- reduce_scalar
- reduce_inplace
- allgather
- alltoall
- scan
- exscan
- gatherv
- scatterv
- allgatherv
- alltoallv
- reduce_scatter_block
- allreduce_scalar (Sum, Min, Prod)
- allreduce_inplace
- scan_scalar
- exscan_scalar

**Potentially missing or needing additional type coverage:**

- broadcast with u8, u32, u64, i64 types
- reduce with Min, Prod ops
- All operations with i32 type (only broadcast has i32 test)

### Step 2: Create `test_blocking_extra.rs` if needed

Focus on:

1. Multi-type coverage: exercise at least one operation with each supported type (u8, u32, u64, i64, f32)
2. Edge cases: empty buffers where applicable, single-process tests
3. Any methods found uncovered in the baseline audit

### Step 3: Verify coverage improvement

```bash
./tests/run_mpi_coverage.sh
# Check comm.rs coverage for blocking collective methods
```

## Definition of Done

- Every blocking collective method has at least one integration test exercising its FFI path
- Coverage of the blocking collective section of `comm.rs` reaches >90%
