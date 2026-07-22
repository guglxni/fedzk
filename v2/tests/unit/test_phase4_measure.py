# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""Phase 4 — Adult measurement transcript smoke."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEASURE = ROOT / "artifacts" / "transcripts" / "adult-lr-measure.json"


def test_adult_lr_measure_transcript_shape():
    assert MEASURE.is_file(), "run scripts/adult_lr_measure.py first"
    data = json.loads(MEASURE.read_text())
    assert data["wire"] == "fedzk.proof.v1"
    assert data["experiment"] == "adult-lr-measure"
    assert data["n_circuit"] in (4, 64, 256)
    assert "prove" in data["metrics"]
    assert data["metrics"]["prove"].get("n", 0) >= 1
    assert data["rounds"]
    assert all(c["http"] == 200 for r in data["rounds"] for c in r["clients"])
