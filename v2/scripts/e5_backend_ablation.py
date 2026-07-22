#!/usr/bin/env python3
# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""E5 microbench — snarkjs vs arkworks (fedzk-zk) verify latency on N=4 golden + N=64 live proof."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedzk.prover.chunked import prove_update  # noqa: E402
from fedzk.prover.engine import find_fedzk_zk, verify_with_rust  # noqa: E402
from fedzk.prover.zkgenerator import ASSET_DIR  # noqa: E402
from fedzk.zk.circuit_config import PROFILE_N4, PROFILE_N64  # noqa: E402


def _snarkjs_verify(proof: dict, public: list, vkey: Path) -> float:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "proof.json").write_text(json.dumps(proof))
        (tdp / "public.json").write_text(json.dumps(public))
        t0 = time.perf_counter()
        r = subprocess.run(
            [
                "snarkjs",
                "groth16",
                "verify",
                str(vkey),
                str(tdp / "public.json"),
                str(tdp / "proof.json"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        if r.returncode != 0 or "OK" not in (r.stdout or ""):
            raise RuntimeError(f"snarkjs verify failed: {r.stderr}")
        return ms


def _rust_verify(proof: dict, public: list, vkey: Path) -> float:
    t0 = time.perf_counter()
    ok = verify_with_rust(proof, public, str(vkey))
    ms = (time.perf_counter() - t0) * 1000.0
    if not ok:
        raise RuntimeError("rust verify returned False")
    return ms


def _stat(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {
        "n": len(xs),
        "mean_ms": statistics.fmean(xs),
        "p50_ms": xs[len(xs) // 2],
        "p95_ms": xs[max(0, int(len(xs) * 0.95) - 1)],
        "max_ms": max(xs),
    }


def bench(label: str, proof: dict, public: list, vkey: Path, rounds: int = 20) -> dict:
    # warmup
    _snarkjs_verify(proof, public, vkey)
    _rust_verify(proof, public, vkey)
    s, r = [], []
    for _ in range(rounds):
        s.append(_snarkjs_verify(proof, public, vkey))
        r.append(_rust_verify(proof, public, vkey))
    return {"label": label, "snarkjs": _stat(s), "rust": _stat(r), "rounds": rounds}


def main() -> None:
    if find_fedzk_zk() is None:
        raise SystemExit("fedzk-zk binary required — cargo build -p fedzk-zk in v2/rust")

    golden = json.loads(
        (ROOT / "tests" / "goldens" / "model_update_n4" / "proof.json").read_text()
    )
    rows = [
        bench(
            "n4_golden",
            golden["proof"],
            golden["public_inputs"],
            ASSET_DIR / PROFILE_N4.vkey_name,
        )
    ]
    proved = prove_update({"w": torch.randn(64) * 0.001}, n=64)
    rows.append(
        bench(
            "n64_live",
            proved["proof"],
            proved["public_inputs"],
            ASSET_DIR / PROFILE_N64.vkey_name,
        )
    )

    out = {
        "wire": "fedzk.proof.v1",
        "experiment": "e5-backend-ablation",
        "rows": rows,
    }
    dest = ROOT / "artifacts" / "transcripts" / "e5-backend-ablation.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))

    # append markdown table
    md = ROOT / "artifacts" / "paper" / "metrics-tables.md"
    lines = [
        "",
        "## E5 — Verify backend ablation (ms)",
        "",
        "| Case | snarkjs mean | snarkjs p50 | rust mean | rust p50 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['snarkjs']['mean_ms']:.1f} | "
            f"{row['snarkjs']['p50_ms']:.1f} | {row['rust']['mean_ms']:.1f} | "
            f"{row['rust']['p50_ms']:.1f} |"
        )
    lines.append("")
    if md.is_file():
        text = md.read_text()
        if "## E5 —" in text:
            text = text.split("## E5 —")[0].rstrip() + "\n"
        md.write_text(text + "\n".join(lines))
    else:
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text("# FEDzk metrics\n" + "\n".join(lines))

    print(json.dumps({"wrote": str(dest), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
