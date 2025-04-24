#!/bin/bash

# Create directory structure
mkdir -p fedzk/docs/legal \
         fedzk/scripts/ci \
         fedzk/scripts/deployment \
         fedzk/scripts/hooks \
         fedzk/build/docker \
         fedzk/build/packaging \
         fedzk/artifacts/benchmarks \
         fedzk/artifacts/tests \
         fedzk/artifacts/proofs \
         fedzk/zk/circuits

# Move legal and project documentation
mv CONTRIBUTING.md fedzk/docs/ 2>/dev/null
mv ROADMAP.md fedzk/docs/ 2>/dev/null
mv SECURITY.md fedzk/docs/legal/ 2>/dev/null
mv license_header.txt fedzk/docs/legal/ 2>/dev/null

# Move Docker and deployment files
mv Dockerfile.* fedzk/build/docker/ 2>/dev/null
mv docker-compose.yml fedzk/build/docker/ 2>/dev/null
mv requirements-docs.txt fedzk/build/packaging/ 2>/dev/null
mv pyproject.toml fedzk/build/packaging/ 2>/dev/null

# Move CI/CD configuration
mv mkdocs.yml fedzk/scripts/ci/ 2>/dev/null

# Move artifacts
mv benchmark_results fedzk/artifacts/benchmarks/ 2>/dev/null
mv test_results fedzk/artifacts/tests/ 2>/dev/null
mv *-benchmark_test.json fedzk/artifacts/benchmarks/ 2>/dev/null
mv *-proof.json fedzk/artifacts/proofs/ 2>/dev/null
mv *-witness.wtns fedzk/artifacts/tests/ 2>/dev/null
mv secure_benchmark.json fedzk/artifacts/benchmarks/ 2>/dev/null
mv secure_report.csv fedzk/artifacts/benchmarks/ 2>/dev/null

# Move circuit and ZK-related files
mv circuits/* fedzk/zk/circuits/ 2>/dev/null
mv zk/* fedzk/zk/ 2>/dev/null

# Move test-related files
mv test-*.json fedzk/artifacts/tests/ 2>/dev/null
mv test-*.wtns fedzk/artifacts/tests/ 2>/dev/null
mv model_update.sym fedzk/artifacts/tests/ 2>/dev/null

# Clean up empty directories
rmdir circuits zk 2>/dev/null

echo "Project files organized successfully!" 