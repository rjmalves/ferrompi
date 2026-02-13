# Ticket 002: Verify Existing Coverage Baseline

## Summary

Run the coverage runner script with just the existing test examples and unit tests to establish a baseline. Document which methods are now covered and which still need new tests.

## Dependencies

- ticket-001 (coverage runner script)

## Acceptance Criteria

- [ ] Baseline coverage report generated with existing tests
- [ ] Per-file coverage numbers documented
- [ ] List of uncovered methods/functions identified per file
- [ ] Baseline numbers used to validate that new test tickets improve coverage
- [ ] Coverage numbers include both unit tests and existing MPI integration examples

## Files to Create/Modify

- **Modify**: `plans/code-coverage-improvement/00-master-plan.md` (update baseline numbers if they changed)

## Technical Details

### Step 1: Run the coverage script

```bash
./tests/run_mpi_coverage.sh
```

### Step 2: Examine the per-file report

```bash
cargo llvm-cov report --summary-only
```

### Step 3: Compare against the investigation baseline

The investigation found ~42.57% line coverage from unit tests alone. With the existing MPI examples (test_collectives, test_nonblocking, test_comm_split) now included, coverage should be higher.

### Step 4: Document gaps

For each source file, record which public functions are still not covered. This validates which subsequent tickets are still needed.

## Definition of Done

- Baseline report exists and can be reproduced
- Gaps are documented to guide remaining tickets
