# Prove-path spike notes (Phase 3 U41 — not started)

Status as of 2026-07-22 on branch `v2`:

| Stage | Engine | Notes |
| --- | --- | --- |
| Quantize | Python `GradientQuantizer` | Required before witness |
| Witness + prove | **snarkjs** (Node) | Default; still the only prove path |
| Verify | snarkjs **or** `fedzk-zk` (arkworks) | `FEDZK_ZK_BACKEND=snarkjs\|rust\|auto` |
| HTTP verify | `fedzk-zk serve` + `fedzk.prover.sidecar` | Optional |

## Why verify-first
E5 shows ~2.4× faster verify on arkworks vs snarkjs for N=4/64. Prove still needs Circom wasm witness + zkey; options for U41:

1. **ark-circom / circom-compat** zkey prove (time-box; CircomReduction required)
2. **rapidsnark FFI** prove, keep arkworks verify
3. Stay snarkjs prove + Rust verify (acceptable interim for paper systems section)

## Non-goals this spike
- Do not claim Rust prove until golden cross-prove matches snarkjs
- Do not remove Node from prove images until U41 exits

See `LIMITATIONS.md` and PLAN Phase 3.
