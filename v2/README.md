# FEDzk v2 (subfolder rebuild)

This directory is the **v2 contribution rebuild**. It lives beside the unchanged GitHub tree at the repo root.

| Path | Role |
|------|------|
| Repo root (`/`) | Same as [`origin/main`](https://github.com/guglxni/fedzk) — do not modify for v2 work |
| **`v2/`** (this folder) | HTML-first agent kit + Bolt1+ codebase |

## Agent kit (HTML-first)

```bash
cd v2/docs/agent-kit && python3 -m http.server 8765
# open http://127.0.0.1:8765/MANIFEST.html
# conductor diary: scratchpad.html
```

## Install & sensors (from this folder)

```bash
cd v2
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
# Needs snarkjs + node on PATH
fedzk doctor
fedzk --help

cat > /tmp/grads.json <<'EOF'
{"w": [0.001, -0.002, 0.003, 0.0]}
EOF
fedzk generate -i /tmp/grads.json -o /tmp/proof.json
fedzk verify -i /tmp/proof.json

pytest tests/unit/test_phase0_bolt1.py -q
```

## Verify backend

```bash
# default: snarkjs subprocess
fedzk doctor

# arkworks verify via fedzk-zk (build first)
(cd rust && cargo build -p fedzk-zk)
export FEDZK_ZK_BACKEND=rust   # or auto|snarkjs
# optional: export FEDZK_ZK_BIN=/path/to/fedzk-zk
fedzk doctor                   # checks zk_backend health when rust
fedzk verify -i /tmp/proof.json
```

See [`docs/PROVE_PATH.md`](docs/PROVE_PATH.md). Prove remains snarkjs until U41.

## Sensors / paper scripts

```bash
./scripts/ci_local.sh
python scripts/adult_lr_measure.py
python scripts/adult_lr_backend_matrix.py   # snarkjs vs rust submit matrix
python scripts/attack_rejection_suite.py
python scripts/e5_backend_ablation.py
python scripts/render_paper_metrics.py
```

## What v2 fixes (Bolt 1)

- CLI package shadow removed (`cli.py` is the module)
- `fedzk doctor` with artifact hashes
- `GradientQuantizer` on prove path
- MPC defaults → `src/fedzk/zk/`
- Coordinator `aggregator` → deprecated shim to `api:app`
- Honest Alpha README (this file); root README left as upstream

## Status

See [`docs/agent-kit/scratchpad.html`](docs/agent-kit/scratchpad.html) and [`docs/agent-kit/PLAN.html`](docs/agent-kit/PLAN.html).

## License

Same as parent repo (FSL-1.1-Apache-2.0).
