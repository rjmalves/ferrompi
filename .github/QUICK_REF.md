# CI/CD Quick Reference

## Common Commands

### Before Pushing
```bash
# Run all checks locally
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings -W clippy::pedantic
cargo test --all-features
cargo build --examples --release

# Test examples with MPI
for ex in hello_world ring allreduce nonblocking persistent_bcast; do
  mpiexec -n 4 ./target/release/examples/$ex
done
```

### Publishing a Release
```bash
# 1. Update version in Cargo.toml
# 2. Update CHANGELOG.md
# 3. Commit
git commit -am "chore: release v0.2.0"

# 4. Tag and push
git tag v0.2.0
git push origin v0.2.0

# CI will automatically:
# - Run security audit
# - Test on Linux + macOS
# - Publish to crates.io
# - Create GitHub release
```

### Running Benchmarks
```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench my_benchmark

# With criterion output
cargo bench -- --verbose
```

## Workflow Triggers

| Workflow   | Push to main | PR | Tag | Schedule | Manual |
|------------|--------------|----|----|----------|--------|
| Tests      | ✅           | ✅ | ❌ | ❌       | ✅     |
| Benchmarks | ✅           | ✅ | ❌ | ❌       | ✅     |
| Publish    | ❌           | ❌ | ✅ | ❌       | ❌     |
| Docs       | ✅           | ✅ | ❌ | ❌       | ✅     |
| Security   | ✅           | ✅ | ❌ | Weekly   | ✅     |

## Required Secrets

### CRATES_IO_TOKEN
1. Go to https://crates.io/settings/tokens
2. Create new token with "publish-update" scope
3. Add to GitHub: Settings → Secrets → Actions
4. Name: `CRATES_IO_TOKEN`

### CODECOV_TOKEN (optional)
1. Go to https://codecov.io/
2. Sign up and add repository
3. Copy token
4. Add to GitHub secrets
5. Name: `CODECOV_TOKEN`

## Status Checks

### Required for Merge (Recommended)
- ✅ Test Suite
- ✅ Documentation Build
- ✅ Security Audit

### Optional
- Documentation Quality Check
- Code Coverage
- Benchmarks (informational only)

## Troubleshooting

### Clippy Pedantic Warnings
```bash
# Allow specific pedantic lints
#[allow(clippy::must_use_candidate)]
#[allow(clippy::missing_errors_doc)]
#[allow(clippy::module_name_repetitions)]
```

### Coverage Issues
```bash
# Run coverage locally
cargo install cargo-llvm-cov
cargo llvm-cov --all-features --lcov --output-path lcov.info
```

### Benchmark Regressions
If benchmarks alert on PR:
1. Download Criterion HTML reports from artifacts
2. Compare baseline vs current
3. Profile locally: `cargo bench`
4. Investigate hot paths

## URLs

- **Actions**: https://github.com/rjmalves/ferrompi/actions
- **Releases**: https://github.com/rjmalves/ferrompi/releases
- **Docs**: https://rjmalves.github.io/ferrompi/
- **Crate**: https://crates.io/crates/ferrompi
- **Security**: https://github.com/rjmalves/ferrompi/security

## Badge Markdown

```markdown
[![CI](https://github.com/rjmalves/ferrompi/workflows/Tests/badge.svg)](https://github.com/rjmalves/ferrompi/actions)
[![codecov](https://codecov.io/gh/rjmalves/ferrompi/branch/main/graph/badge.svg)](https://codecov.io/gh/rjmalves/ferrompi)
[![Security](https://github.com/rjmalves/ferrompi/workflows/Security/badge.svg)](https://github.com/rjmalves/ferrompi/actions)
```
