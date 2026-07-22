# Units Generation — Integrated DAG (conductor + architect)

## Bolt 1 — Walking skeleton (sequential, gated)
- U00 Fix CLI shadow (cli/__init__.py exports app OR remove package)
- U01 fedzk doctor (toolchain + artifact hashes)
- U02 Wire GradientQuantizer into ZKProver/MPC/CLI
- U03 Prove/verify/FedAvg smoke on N_dev=4 integers with golden

## Phase 0 parallel after Bolt 1
- U10 MPC artifact default paths → src/fedzk/zk/
- U11 Collapse dual coordinators (archive aggregator.py)
- U12 Rewrite e2e to real symbols
- U13 Secrets untrack + gitleaks
- U14 torch/deps honesty in pyproject
- U15 Docs/API alignment (getting_started)

## Next waves
- U20–U24 Parameterized N + chunk protocol + goldens
- U30–U33 Stable SDK exports
- U40–U43 Rust fedzk-zk (Strategy C→B)
- U50–U52 Paper experiments / transcripts

Dependency: Bolt1 → (U10…U15 parallel) → U20… → U30… → U40… → U50…
