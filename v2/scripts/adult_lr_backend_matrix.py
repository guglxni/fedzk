#!/usr/bin/env python3
# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""Run Adult measure under snarkjs + rust verify backends; write comparison matrix."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_measure():
    spec = importlib.util.spec_from_file_location(
        "adult_lr_measure", ROOT / "scripts" / "adult_lr_measure.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    from fedzk.prover.engine import find_fedzk_zk

    mod = _load_measure()
    rows = []
    for name in ("snarkjs", "rust"):
        if name == "rust" and find_fedzk_zk() is None:
            rows.append(
                {"backend": "rust", "skipped": True, "reason": "fedzk-zk missing"}
            )
            continue
        os.environ["FEDZK_ZK_BACKEND"] = name
        out = mod.run_measure()
        m = out.get("metrics") or {}
        rows.append(
            {
                "backend": name,
                "zk_backend": out.get("zk_backend"),
                "metrics": m,
                "baseline_train": (out.get("baseline") or {})
                .get("metrics", {})
                .get("train"),
                "n_circuit": out.get("n_circuit"),
                "ceremony": out.get("ceremony"),
            }
        )

    matrix = {
        "wire": "fedzk.proof.v1",
        "experiment": "adult-lr-backend-matrix",
        "rows": rows,
    }
    dest = ROOT / "artifacts" / "transcripts" / "adult-lr-backend-matrix.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(matrix, indent=2))

    md_path = ROOT / "artifacts" / "paper" / "metrics-tables.md"
    lines = [
        "",
        "## Adult LR — verify backend matrix (submit path)",
        "",
        "| Backend | prove p50 | rust_verify p50 | submit p50 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        if r.get("skipped"):
            lines.append(f"| {r['backend']} | skipped | — | — |")
            continue
        m = r.get("metrics") or {}
        lines.append(
            f"| {r['backend']} | "
            f"{(m.get('prove') or {}).get('p50_ms', 0):.1f} | "
            f"{(m.get('rust_verify') or {}).get('p50_ms', 0):.1f} | "
            f"{(m.get('submit') or {}).get('p50_ms', 0):.1f} |"
        )
    lines.append("")
    if md_path.is_file():
        text = md_path.read_text()
        marker = "## Adult LR — verify backend matrix"
        if marker in text:
            text = text.split(marker)[0].rstrip() + "\n"
        md_path.write_text(text + "\n".join(lines))
    print(json.dumps({"wrote": str(dest), "summary": [
        {k: r.get(k) for k in ("backend", "zk_backend", "skipped") if k in r or True}
        | {"submit_p50": (r.get("metrics") or {}).get("submit", {}).get("p50_ms")}
        for r in rows
    ]}, indent=2))


if __name__ == "__main__":
    main()
