# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""Phase 3 scaffold — FEDZK_ZK_BACKEND + fedzk-zk health."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from fedzk.prover import engine as eng

GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "goldens"
    / "model_update_n4"
    / "proof.json"
)
VKEY = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "fedzk"
    / "zk"
    / "verification_key.json"
)


def test_default_backend_snarkjs(monkeypatch):
    monkeypatch.delenv("FEDZK_ZK_BACKEND", raising=False)
    assert eng.selected_backend() == "snarkjs"
    assert eng.resolve_backend() == "snarkjs"


def test_rust_binary_builds_and_health():
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "rust" / "target" / "debug" / "fedzk-zk",
        root / "rust" / "fedzk-zk" / "target" / "debug" / "fedzk-zk",
    ]
    if not any(p.is_file() for p in candidates):
        pytest.skip("run: cargo build -p fedzk-zk in v2/rust")
    ok, payload = eng.rust_health()
    assert ok is True
    assert payload.get("wire") == "fedzk.proof.v1"


def test_rust_verify_fail_closed_until_arkworks(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    bin_path = next(
        (
            p
            for p in (
                root / "rust" / "target" / "debug" / "fedzk-zk",
                root / "rust" / "fedzk-zk" / "target" / "debug" / "fedzk-zk",
            )
            if p.is_file()
        ),
        None,
    )
    if bin_path is None:
        pytest.skip("fedzk-zk not built")
    monkeypatch.setenv("FEDZK_ZK_BACKEND", "rust")
    monkeypatch.setenv("FEDZK_ZK_BIN", str(bin_path))
    data = json.loads(GOLDEN.read_text())
    with pytest.raises(eng.RustEngineUnavailable):
        eng.verify_with_rust(data["proof"], data["public_inputs"], str(VKEY))
