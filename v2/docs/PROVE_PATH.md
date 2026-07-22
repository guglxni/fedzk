# Prove-path spike notes (Phase 3 U41)

Status as of 2026-07-22 on branch `v2`:

| Stage | Engine | Notes |
| --- | --- | --- |
| Quantize | Python `GradientQuantizer` | Required before witness |
| Witness + prove | **snarkjs** (Node) | Default; still the only prove path |
| Verify | snarkjs **or** `fedzk-zk` (arkworks) | `FEDZK_ZK_BACKEND=snarkjs\|rust\|auto` |
| HTTP verify | `fedzk-zk serve` + `fedzk.prover.sidecar` | Optional |

## Why verify-first
E5 shows ~2.4× faster verify on arkworks vs snarkjs for N=4/64. With
`FEDZK_ZK_BACKEND=rust`, Adult measure **submit** p50 dropped ~240ms → ~137ms
(coordinator uses `ZKVerifier` → rust), while prove remains ~270–280ms snarkjs.

## U41 time-box decision (recorded)
**Preference order for prove-path spike (1 week max):**

1. **Stay snarkjs prove + Rust verify** as paper default (already measured; lowest risk).
2. **rapidsnark FFI** only if a 2-day spike shows zkey load + prove parity on N=4 golden.
3. **ark-circom prove** deferred — highest integration risk (CircomReduction / zkey format); revisit after (2) fails.

Exit criteria for claiming “Rust prove”: bit-identical public signals + snarkjs-verify OK on goldens for N∈{4,64}.

## Non-goals this spike
- Do not claim Rust prove until golden cross-prove matches snarkjs
- Do not remove Node from prove images until U41 exits

See `LIMITATIONS.md` and PLAN Phase 3.
