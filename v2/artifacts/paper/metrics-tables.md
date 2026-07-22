# FEDzk Phase 4 — metrics tables (auto-generated)

Source: `adult-lr-measure.json` (n_circuit=64, ceremony=dev_unsafe).

## T1 — Adult LR wall-time (ms / client update)

| Arm | mean | p50 | max |
| --- | ---: | ---: | ---: |
| baseline train (no ZK) | 5.9 | 1.3 | 15.3 |
| FEDzk prove (snarkjs) | 272.1 | 260.2 | 296.0 |
| FEDzk rust verify | 74.3 | 71.4 | 80.5 |
| FEDzk coordinator submit | 242.5 | 240.5 | 247.5 |

Train loss (FEDzk arm mean): `0.6719358811775843`

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

## E5 — Verify backend ablation (ms)

| Case | snarkjs mean | snarkjs p50 | rust mean | rust p50 |
| --- | ---: | ---: | ---: | ---: |
| n4_golden | 180.3 | 179.3 | 75.1 | 73.7 |
| n64_live | 189.4 | 182.0 | 71.3 | 70.7 |
