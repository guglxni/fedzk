# Changelog

All notable changes to FedZK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-05-10

### Added
- PyPI package available for production use
- Full documentation with installation and usage guides

### Changed
- Upgraded from release candidate to general availability
- Improved testing reliability

## [1.0.0-rc1] - 2025-05-06

### Added
- Circom compile CI job
- SLSA provenance + SBOM on release

### Fixed
- Duplicate tool blocks in pyproject.toml
- Poetry lockfile, dependency conflicts
- Circuit duplication by centralizing circuit files

## [0.1.0] - 2025-04-24

### Added
- Initial release of FedZK framework
- Client training module with gradient update and proof generation
- Zero-knowledge circuit implementation for gradient verification
- MPC server for remote proof generation and verification
- API key authentication for secure MPC server access
- Coordinator API for model aggregation
- Batch processing support for efficient proof generation
- End-to-end benchmarking suite
- Command-line interface for all operations
- Comprehensive documentation including user guide, contributions guide, and API reference
- Example scripts and service deployment configurations

### Changed
- N/A (Initial release)

### Fixed
- N/A (Initial release)

[1.0.0-rc1]: https://github.com/guglxni/fedzk/compare/v0.1.0...v1.0.0-rc1
[0.1.0]: https://github.com/guglxni/fedzk/releases/tag/v0.1.0
[1.0.0]: https://github.com/guglxni/fedzk/compare/v1.0.0-rc1...v1.0.0 