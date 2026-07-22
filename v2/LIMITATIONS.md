# FEDzk v2 — Limitations (from measured work, 2026-07-22)

Honest constraints for paper Discussion / OSS docs. Do not claim beyond these.

## Cryptography & ceremony
- N=4 legacy artifacts are **frozen** from an earlier pipeline; Circom source `model_update.circom` **drifts** from the shipped wasm witness shape (`gradients[n]` only). Artifacts are authoritative until regenerate.
- N=64 and N=256 use local snarkjs Powers-of-Tau — labeled `ceremony=dev_unsafe`. **Not** a multiparty production ceremony.
- Prove path remains **snarkjs**; Rust (`fedzk-zk`) currently accelerates **verify** only (E5).

## Protocol
- Chunk commitment check is structural (meta ↔ outer commitment). Binding full quantized grads → commitment is by construction on the honest client path; coordinator does not yet recompute commitment from submitted float gradients.
- AdvancedProofValidator heuristics can false-score honest Groth16 proofs; coordinator softens that for crypto-verified paths but the scorer remains in-tree.

## Scale & ML evaluation
- Adult / CIFAR configs in `spec/experiments/` are **synthetic or stubs** for systems metrics. Full UCI Adult / CIFAR-10 accuracy tables are not claimed yet.
- Circuit N pads/truncates with logging; silent loss of high-dim CNN grads is a research risk until chunking is the default for large models.

## Ops
- GitHub Actions workflow under `v2/.github/` is a **template** only (Actions discovers root `.github/`). Local gate: `scripts/ci_local.sh`.
- Docker compose under `v2/deploy/` is demo-grade; Node+snarkjs in the coordinator image is the documented production weak link Phase 3 removes.

## Measured bright spots (not overclaims)
- Attack suite rejection rate **1.0** on tamper/empty/commitment-mismatch cases (`artifacts/transcripts/attack-rejection.json`).
- Rust verify ~2.4× faster than snarkjs on N=4/N=64 verify microbench (`e5-backend-ablation.json`).
