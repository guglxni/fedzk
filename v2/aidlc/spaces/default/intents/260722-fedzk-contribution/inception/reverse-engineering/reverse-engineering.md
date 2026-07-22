# Reverse Engineering — Integrated (conductor)

Lead synthesis after developer contribution.

## Brownfield verdict
Research-demo scaffolding with real snarkjs hooks but broken packaging/API honesty.
Not production-ready; README overclaims.

## Critical defects (ordered)
1. CLI package shadows cli.py — entrypoint dead
2. Quantizer unwired from prove path
3. MPC defaults → missing circuits/build
4. Dual coordinator stacks
5. E2E fictional types
6. 223+ secrets tracked

## Implications for units
Bolt 1 must include CLI unshadow + doctor + N_dev prove/verify with quantization.
See units-generation/units.md.
