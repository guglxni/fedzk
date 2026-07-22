# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""Backend matrix transcript shape."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "artifacts" / "transcripts" / "adult-lr-backend-matrix.json"


def test_backend_matrix_transcript():
    assert MATRIX.is_file(), "run scripts/adult_lr_backend_matrix.py first"
    data = json.loads(MATRIX.read_text())
    assert data["experiment"] == "adult-lr-backend-matrix"
    backends = {r["backend"] for r in data["rows"] if not r.get("skipped")}
    assert "snarkjs" in backends
    assert "rust" in backends
    for r in data["rows"]:
        if r.get("skipped"):
            continue
        assert r["metrics"]["submit"]["p50_ms"] > 0
