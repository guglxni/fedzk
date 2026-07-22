# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""E5 backend ablation transcript shape."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E5 = ROOT / "artifacts" / "transcripts" / "e5-backend-ablation.json"


def test_e5_ablation_transcript():
    assert E5.is_file(), "run scripts/e5_backend_ablation.py first"
    data = json.loads(E5.read_text())
    assert data["experiment"] == "e5-backend-ablation"
    assert len(data["rows"]) >= 2
    for row in data["rows"]:
        assert row["rust"]["mean_ms"] > 0
        assert row["snarkjs"]["mean_ms"] > 0
        # Sanity: rust should not be wildly slower than snarkjs on this machine
        assert row["rust"]["mean_ms"] < row["snarkjs"]["mean_ms"] * 2.5
