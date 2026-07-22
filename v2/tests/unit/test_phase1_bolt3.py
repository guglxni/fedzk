# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""Bolt 3 — N=64 prove, chunked prove, coordinator chunk verify."""

from __future__ import annotations

import torch
from fastapi.testclient import TestClient

from fedzk.prover.chunked import prove_update
from fedzk.prover.verifier import ZKVerifier
from fedzk.zk.circuit_config import PROFILE_N64
from fedzk.prover.zkgenerator import ASSET_DIR as ZK_ASSET


def test_prove_n64_single():
    g = {"w": torch.randn(64) * 0.001}
    out = prove_update(g, n=64)
    assert out["mode"] == "single"
    assert out["n"] == 64
    assert out["circuit_id"] == "model_update_gradients_n64"
    vkey = ZK_ASSET / PROFILE_N64.vkey_name
    ok = ZKVerifier(verification_key_path=str(vkey)).verify_real_proof(
        out["proof"], out["public_inputs"]
    )
    assert ok is True


def test_prove_n256_single():
    g = {"w": torch.randn(256) * 0.001}
    out = prove_update(g, n=256)
    assert out["mode"] == "single"
    assert out["n"] == 256
    from fedzk.zk.circuit_config import PROFILE_N256

    vkey = ZK_ASSET / PROFILE_N256.vkey_name
    ok = ZKVerifier(verification_key_path=str(vkey)).verify_real_proof(
        out["proof"], out["public_inputs"]
    )
    assert ok is True


def test_prove_chunked_n4():
    # 10 values → 3 chunks of n=4
    g = {"w": torch.arange(10, dtype=torch.float32) * 0.001}
    out = prove_update(g, n=4, prefer_single_if_fits=False)
    # force chunked: prefer_single False with len>0 still might single if len<=4
    # use prefer and len>4
    assert out["mode"] == "chunked"
    assert len(out["chunks"]) == 3
    assert out["commitment"]


def test_coordinator_accepts_chunk_bundle():
    from fedzk.coordinator import logic as logic_mod
    from fedzk.coordinator.api import app

    # Reset pending state between tests
    logic_mod.pending_updates.clear()

    g = {"w": torch.arange(9, dtype=torch.float32) * 0.001}
    proved = prove_update(g, n=4)  # len=9 → chunked
    assert proved["mode"] == "chunked"

    client = TestClient(app)
    payload = {
        "gradients": {"w": g["w"].tolist()},
        "proof": {},
        "public_inputs": [],
        "client_id": "chunk-client-1",
        "chunk_bundle": {
            "commitment": proved["commitment"],
            "n": proved["n"],
            "circuit_id": proved["circuit_id"],
            "chunks": proved["chunks"],
        },
    }
    r = client.post("/submit_update", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["status"] in ("accepted", "aggregated")
