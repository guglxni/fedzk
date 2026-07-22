# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""Phase 4 — attack rejection transcript + ZKVerifier rust backend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fedzk.prover.verifier import ZKVerifier
from fedzk.prover.zkgenerator import ASSET_DIR
from fedzk.zk.circuit_config import PROFILE_N4

ROOT = Path(__file__).resolve().parents[2]
ATTACK = ROOT / "artifacts" / "transcripts" / "attack-rejection.json"
GOLDEN = ROOT / "tests" / "goldens" / "model_update_n4" / "proof.json"


def test_attack_rejection_transcript():
    assert ATTACK.is_file(), "run scripts/attack_rejection_suite.py first"
    data = json.loads(ATTACK.read_text())
    assert data["experiment"] == "attack-rejection"
    assert data["metrics"]["honest_accept"] is True
    assert data["metrics"]["rejection_rate"] == 1.0
    assert data["metrics"]["attacks_rejected"] == data["metrics"]["attacks_total"]


def test_zkverifier_rust_backend(monkeypatch):
    bin_path = ROOT / "rust" / "target" / "debug" / "fedzk-zk"
    if not bin_path.is_file():
        pytest.skip("fedzk-zk not built")
    monkeypatch.setenv("FEDZK_ZK_BACKEND", "rust")
    monkeypatch.setenv("FEDZK_ZK_BIN", str(bin_path))
    data = json.loads(GOLDEN.read_text())
    v = ZKVerifier(verification_key_path=str(ASSET_DIR / PROFILE_N4.vkey_name))
    assert v.verify_real_proof(data["proof"], data["public_inputs"]) is True
