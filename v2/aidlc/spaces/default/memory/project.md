# Project specialization — FEDzk

## Intent

Build FEDzk into a contribution-grade open-source ZK-FL framework and research platform
(client-side Groth16 proofs of update integrity + reproducible experiments).

## Scope selection (AIDLC)

- **Primary scope:** `mvp` for walking skeleton + Phase 0–2, then promote to `feature`/`enterprise` for ops.
- **Depth:** Comprehensive for crypto/architecture stages; Standard elsewhere.
- **Test strategy:** Comprehensive for prover/coordinator; Standard for demos.

## Human-proxy / loop autonomy (Cursor)

User has requested loop engineering with agents taking the human seat where needed:

- Conductor MAY auto-approve stage gates for documentation/planning artifacts when:
  1. Sensors (required sections, consistency with CONCEPTS) pass, AND
  2. No irreversible prod deploy / secret rotation / release tag is involved.
- Construction code changes still prefer explicit user review unless user says "autonomous construction".
- All auto-approvals MUST be logged under the intent `audit/` as `human-proxy-approve`.

## Upstream

- AIDLC methodology: https://github.com/awslabs/aidlc-workflows/tree/v2
- Pin file: `docs/agent-kit/.aidlc-v2.pin`

## Stack locks

- Python + PyTorch (FL), Circom (circuits), Rust prove path (Strategy C→B).
- Proof wire: `fedzk.proof.v1` snarkjs-shaped Groth16 JSON.

## Known brownfield debt (from reverse engineering)

- CLI package shadows cli.py
- torch optional but hard-imported
- n=4 circuits; quantizer unwired
- MPC artifact path defaults wrong
- Dual coordinators; fictional e2e APIs
- 223 secrets tracked


## Learnings from 260722 loop
- Empty `src/fedzk/cli/` package silently breaks console entrypoints — always verify `from fedzk.cli import app` in doctor.
- Quantizer existence ≠ prove-path wiring; treat encode as part of the cryptographic statement.
- Dual FastAPI coordinators are a protocol bug, not an optional deploy flavor.
- Human-proxy may approve inception docs; Construction Bolt 1 code still defaults to user-visible PR.
- Updated: 2026-07-22T12:53:22.779597+00:00
