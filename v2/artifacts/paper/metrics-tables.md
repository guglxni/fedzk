# FEDzk Phase 4 — metrics tables (auto-generated)

Source: `adult-lr-measure.json` (n_circuit=64, ceremony=dev_unsafe).

## T1 — Adult LR wall-time (ms / client update)

| Arm | mean | p50 | max |
| --- | ---: | ---: | ---: |
| baseline train (no ZK) | 12.5 | 1.3 | 35.2 |
| FEDzk prove (snarkjs) | 283.0 | 273.7 | 308.3 |
| FEDzk rust verify | 70.2 | 69.7 | 71.3 |
| FEDzk coordinator submit | 138.8 | 137.2 | 142.6 |

Train loss (FEDzk arm mean): `0.7172777454058329`

## T3 — Attack rejection (coordinator fail-closed)

- Honest accept: `True`
- Attacks rejected: `5/5`
- Rejection rate: `1.0`

| Case | expect | http | ok |
| --- | --- | ---: | --- |
| honest | accept | 200 | True |
| tamper_pi_a | reject | 400 | True |
| tamper_public | reject | 400 | True |
| tamper_commitment_top | reject | 400 | True |
| tamper_commitment_meta | reject | 400 | True |
| empty_proof | reject | 400 | True |

---
Regenerate: `python scripts/adult_lr_measure.py && python scripts/attack_rejection_suite.py && python scripts/render_paper_metrics.py`

## Adult LR — verify backend matrix (submit path)

| Backend | prove p50 | rust_verify p50 | submit p50 |
| --- | ---: | ---: | ---: |
| snarkjs | 257.6 | 68.0 | 239.1 |
| rust | 252.9 | 67.7 | 128.0 |
