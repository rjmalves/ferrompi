# CI/CD Documentation

This repository uses GitHub Actions for continuous integration and deployment.

## Workflows Overview

### 🧪 Tests (`test.yml`)

**Triggers:** Push to main/master/develop, Pull Requests

**Jobs:**
- **Multi-platform testing**: Ubuntu + macOS with stable + nightly Rust
- **MSRV check**: Ensures compatibility with Rust 1.70+
- **Minimal versions**: Tests with minimal dependency versions
- **Documentation build**: Validates all docs build without warnings
- **Code coverage**: Generates coverage reports for Codecov
- **MPI integration tests**: Runs all examples with `mpiexec -n 4`

**Features:**
- ✅ Parallel testing across OS and Rust versions
- ✅ Clippy with pedantic lints
- ✅ Format checking with rustfmt
- ✅ Real MPI environment testing
- ✅ Caching for faster builds

### 📊 Benchmarks (`benchmark.yml`)

**Triggers:** Push to main, Pull Requests, Manual dispatch

**Jobs:**
- **Performance tracking**: Runs Criterion benchmarks
- **Regression detection**: Alerts on >10% performance degradation
- **Historical tracking**: Stores benchmark history
- **PR comparisons**: Compares PR performance vs main branch

**Artifacts:**
- Criterion HTML reports (30 days retention)
- Benchmark results JSON (90 days retention)

### 📦 Publish (`publish.yml`)

**Triggers:** Git tags matching `v[0-9]+.[0-9]+.[0-9]+*`

**Jobs:**
1. **Security Audit**: cargo-audit check
2. **Version Verification**: Ensures tag matches Cargo.toml
3. **Multi-platform Tests**: Linux + macOS
4. **Dry-run Publish**: Validates package before publishing
5. **Publish to crates.io**: Releases the crate
6. **GitHub Release**: Creates GitHub release with changelog

**Required Secrets:**
- `CRATES_IO_TOKEN`: API token from crates.io

**Publishing Flow:**
```bash
# 1. Update version in Cargo.toml
# 2. Update CHANGELOG.md
# 3. Commit changes
git commit -am "chore: release v0.2.0"

# 4. Create and push tag
git tag v0.2.0
git push origin v0.2.0

# 5. CI automatically publishes to crates.io
```

### 📚 Documentation (`docs.yml`)

**Triggers:** Push to main, Pull Requests

**Jobs:**
- **Build docs**: Generates documentation with rustdoc
- **Deploy to GitHub Pages**: Auto-deploys on main branch
- **Quality checks**: Ensures no missing docs or broken links
- **Doc tests**: Validates all code examples in docs

**Live Docs:** https://rjmalves.github.io/ferrompi/

### 🔒 Security (`security.yml`)

**Triggers:** Push, Pull Requests, Weekly schedule (Mondays), Manual

**Jobs:**
1. **Security Audit**: Checks for known vulnerabilities
2. **Dependency Review**: Reviews new dependencies in PRs
3. **Outdated Dependencies**: Identifies outdated crates
4. **Unused Dependencies**: Detects unused dependencies
5. **Cargo Deny**: License and ban checks
6. **Security Clippy**: Security-focused lints

**Schedule:** Runs weekly on Mondays at 00:00 UTC

### 🏷️ PR Labels (`pr-labels.yml`)

**Triggers:** Pull Request events

**Jobs:**
- **Auto-labeling**: Labels PRs based on changed files
- **Size labeling**: Adds size labels (XS, S, M, L, XL, XXL)

**Labels:**
- `src`: Changes to source code
- `documentation`: Markdown or docs changes
- `examples`: Changes to examples
- `tests`: Test modifications
- `benchmarks`: Benchmark changes
- `ci-cd`: CI/CD workflow changes
- `dependencies`: Dependency updates
- `build`: Build system changes
- `c-code`: C source changes

### 🤖 Dependabot (`dependabot.yml`)

**Schedule:** Weekly on Mondays at 09:00

**Updates:**
- **Cargo dependencies**: Groups patch and minor updates
- **GitHub Actions**: Keeps workflows up to date

**Configuration:**
- Maximum 10 open PRs for Cargo
- Maximum 5 open PRs for Actions
- Auto-assigns to @rjmalves
- Prefixes commits with "deps:" or "ci:"

## Setup Instructions

### 1. Enable GitHub Actions

Actions are enabled by default. No additional setup needed.

### 2. Configure Secrets

Required for publishing:

1. Go to repository **Settings** → **Secrets and variables** → **Actions**
2. Add the following secrets:

| Secret | Description | How to Get |
|--------|-------------|------------|
| `CRATES_IO_TOKEN` | crates.io API token | https://crates.io/settings/tokens |
| `CODECOV_TOKEN` (optional) | Codecov token | https://codecov.io/ |

### 3. Enable GitHub Pages

For documentation deployment:

1. Go to **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: **gh-pages** / **root**
4. Save

### 4. Configure Branch Protection (Recommended)

Protect the main branch:

1. Go to **Settings** → **Branches**
2. Add rule for `main` or `master`:
   - ✅ Require status checks before merging
   - ✅ Require branches to be up to date
   - Select: `Test Suite`, `Documentation Build`, `Security Audit`
   - ✅ Require conversation resolution before merging
   - ✅ Include administrators

## CI/CD Best Practices

### Running Tests Locally

Before pushing, ensure all checks pass:

```bash
# Format code
cargo fmt --all

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings -W clippy::pedantic

# Run tests
cargo test --all-features

# Build examples
cargo build --examples --release

# Test with MPI
mpiexec -n 4 ./target/release/examples/hello_world
```

### Adding Benchmarks

Create benchmarks in `benches/`:

```rust
// benches/my_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn my_benchmark(c: &mut Criterion) {
    c.bench_function("my_function", |b| {
        b.iter(|| {
            // Code to benchmark
            black_box(expensive_operation());
        });
    });
}

criterion_group!(benches, my_benchmark);
criterion_main!(benches);
```

Add to `Cargo.toml`:

```toml
[[bench]]
name = "my_benchmark"
harness = false
```

### Updating Changelog

Follow [Keep a Changelog](https://keepachangelog.com/):

```markdown
## [Unreleased]

### Added
- New feature X

### Fixed
- Bug Y

## [0.2.0] - 2024-01-15

### Added
- Feature A
- Feature B
```

## Troubleshooting

### Workflow Failures

**MPI installation fails:**
- Check if package names match in workflow
- Ubuntu uses `mpich` and `libmpich-dev`
- macOS uses `mpich` from Homebrew

**Tests fail only in CI:**
- Ensure examples work with `mpiexec -n 4`
- Check for race conditions in parallel tests
- Verify environment variables

**Coverage upload fails:**
- Codecov token may be required for private repos
- Public repos work without token

### Benchmark Alerts

If benchmarks regress:
1. Check the Criterion HTML reports in artifacts
2. Profile locally: `cargo bench`
3. Investigate hot paths
4. Update alert threshold in `benchmark.yml` if intentional

### Publishing Issues

**Version mismatch:**
```bash
# Ensure tag matches Cargo.toml
grep version Cargo.toml
# Should match the tag (without 'v' prefix)
```

**Publish fails:**
- Ensure `CRATES_IO_TOKEN` is set correctly
- Check crate name availability
- Verify all required metadata in `Cargo.toml`

## Monitoring

### Status Badges

Add to your README:

```markdown
[![CI](https://github.com/rjmalves/ferrompi/workflows/Tests/badge.svg)](https://github.com/rjmalves/ferrompi/actions)
[![codecov](https://codecov.io/gh/rjmalves/ferrompi/branch/main/graph/badge.svg)](https://codecov.io/gh/rjmalves/ferrompi)
[![Security](https://github.com/rjmalves/ferrompi/workflows/Security/badge.svg)](https://github.com/rjmalves/ferrompi/actions)
```

### Useful Links

- **Actions:** https://github.com/rjmalves/ferrompi/actions
- **Releases:** https://github.com/rjmalves/ferrompi/releases
- **Security:** https://github.com/rjmalves/ferrompi/security
- **Insights:** https://github.com/rjmalves/ferrompi/pulse

## Future Improvements

Potential enhancements:

- [ ] Add fuzzing with `cargo-fuzz`
- [ ] Integration with rust-analyzer for IDE checks
- [ ] Performance regression dashboard
- [ ] Automated changelog generation
- [ ] Cross-compilation for more platforms
- [ ] Docker-based testing for reproducibility
- [ ] Nightly performance tracking
- [ ] Integration tests with different MPI implementations (OpenMPI, Intel MPI)

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Rust CI Best Practices](https://doc.rust-lang.org/cargo/guide/continuous-integration.html)
- [Criterion.rs](https://github.com/bheisler/criterion.rs)
- [cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov)
