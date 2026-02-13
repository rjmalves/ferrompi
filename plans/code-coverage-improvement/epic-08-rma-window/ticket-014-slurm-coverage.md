# Ticket 014: SLURM Module Coverage

## Summary

Verify and improve coverage of `slurm.rs` module which is behind the `numa` feature gate. Since SLURM functions read environment variables, they can potentially be tested in unit tests by setting env vars, or in integration tests.

## Dependencies

- ticket-001 (coverage runner script)
- ticket-013 (RMA window coverage — since `numa` implies `rma`)

## Acceptance Criteria

- [ ] `slurm.rs` appears in coverage report when running with `--features numa`
- [ ] SLURM helper functions are exercised (either via env var mocking in unit tests or integration tests)
- [ ] Coverage of `slurm.rs` reaches >80%

## Files to Create/Modify

- **Modify** (if needed): `src/slurm.rs` (add unit tests if not present)
- **Modify**: `tests/run_mpi_coverage.sh` (ensure `--features numa` run is included)

## Technical Details

### Step 1: Read slurm.rs to understand what needs coverage

The SLURM module likely reads environment variables like `SLURM_CPUS_PER_TASK`, `SLURM_NTASKS_PER_NODE`, etc. These functions are pure Rust (no FFI) and can be tested with unit tests by temporarily setting env vars.

### Step 2: Add unit tests with env var mocking

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_cpus_per_task() {
        env::set_var("SLURM_CPUS_PER_TASK", "8");
        let result = cpus_per_task();
        assert_eq!(result, Some(8));
        env::remove_var("SLURM_CPUS_PER_TASK");
    }

    #[test]
    fn test_cpus_per_task_missing() {
        env::remove_var("SLURM_CPUS_PER_TASK");
        let result = cpus_per_task();
        assert_eq!(result, None);
    }
}
```

### Step 3: Run coverage with numa feature

```bash
./tests/run_mpi_coverage.sh numa
```

## Definition of Done

- `slurm.rs` shows in coverage report
- All pure-Rust functions in `slurm.rs` are covered by unit tests
- Coverage >80%
