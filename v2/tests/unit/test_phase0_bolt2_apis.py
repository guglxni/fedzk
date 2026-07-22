# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""Bolt 2 sensors — real public APIs only (no fictional CoordinatorAPI/ZKGenerator)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from fedzk.zk.chunk_protocol import (
    commit_quantized,
    split_into_chunks,
    verify_bundle_commitments,
    ChunkProofBundle,
)
from fedzk.zk.circuit_config import N_DEV, PROFILE_N4, require_supported_n


def test_public_exports_are_real():
    import fedzk

    assert hasattr(fedzk, "ZKProver")
    assert hasattr(fedzk, "ZKVerifier")
    assert hasattr(fedzk, "GradientQuantizer")
    # Guard against regressing to fictional names in docs/tests
    assert not hasattr(fedzk, "ZKGenerator")
    assert not hasattr(fedzk, "CoordinatorAPI")
    assert not hasattr(fedzk, "FederatedTrainer")


def test_require_supported_n():
    require_supported_n(N_DEV)
    require_supported_n(64)
    try:
        require_supported_n(256)
        assert False, "expected ValueError for N=256"
    except ValueError as e:
        assert "not shipped" in str(e)


def test_chunk_protocol_commitment_stable():
    vals = [1, 2, 3, 4, 5, 6]
    specs = split_into_chunks(vals, n=4)
    assert len(specs) == 2
    assert specs[0].values == [1, 2, 3, 4]
    assert specs[1].values == [5, 6, 0, 0]
    assert specs[0].commitment == specs[1].commitment == commit_quantized(vals)
    bundle = ChunkProofBundle(
        commitment=specs[0].commitment,
        n=4,
        circuit_id="model_update",
        chunks=[
            {"meta": specs[0].to_public_meta(), "proof": {}},
            {"meta": specs[1].to_public_meta(), "proof": {}},
        ],
    )
    assert verify_bundle_commitments(bundle)


def test_coordinator_health_real_app():
    from fedzk.coordinator.api import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("healthy", "degraded", "unhealthy")
    assert "cryptographic_verification" in body


def test_coordinator_status_real_app():
    from fedzk.coordinator.api import app

    client = TestClient(app)
    r = client.get("/status")
    assert r.status_code == 200
    assert "pending_updates" in r.json()
    assert "model_version" in r.json()


def test_local_trainer_symbol():
    from fedzk.client.trainer import LocalTrainer

    assert LocalTrainer is not None


def test_profile_n4_names():
    assert PROFILE_N4.wasm_name == "model_update.wasm"
    assert PROFILE_N4.n == 4


def test_golden_n4_verify_if_present():
    """If goldens were recorded, verify with ZKVerifier (real symbols)."""
    golden = Path(__file__).resolve().parents[1] / "goldens" / "model_update_n4" / "proof.json"
    if not golden.is_file():
        return
    from fedzk.prover.verifier import ZKVerifier
    from fedzk.prover.zkgenerator import ASSET_DIR

    data = json.loads(golden.read_text())
    vkey = ASSET_DIR / "verification_key.json"
    verifier = ZKVerifier(verification_key_path=str(vkey))
    ok = verifier.verify_real_proof(data["proof"], data["public_inputs"])
    assert ok is True
