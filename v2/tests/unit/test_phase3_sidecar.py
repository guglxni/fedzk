# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""Sidecar HTTP client unit tests (starts fedzk-zk serve briefly)."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from fedzk.prover import sidecar
from fedzk.prover.zkgenerator import ASSET_DIR

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "goldens" / "model_update_n4" / "proof.json"
BIN = ROOT / "rust" / "target" / "debug" / "fedzk-zk"


@pytest.fixture(scope="module")
def running_sidecar():
    if not BIN.is_file():
        pytest.skip("fedzk-zk not built")
    proc = subprocess.Popen(
        [str(BIN), "serve", "--bind", "127.0.0.1:8799"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = "http://127.0.0.1:8799"
    for _ in range(40):
        try:
            sidecar.healthz(base=base, timeout=0.2)
            break
        except sidecar.SidecarUnavailable:
            time.sleep(0.05)
    else:
        proc.kill()
        pytest.skip("sidecar failed to start")
    yield base
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_sidecar_health_and_verify(running_sidecar):
    base = running_sidecar
    h = sidecar.healthz(base=base)
    assert h.get("ok") is True
    data = json.loads(GOLDEN.read_text())
    ok = sidecar.verify_via_sidecar(
        data["proof"],
        data["public_inputs"],
        ASSET_DIR / "verification_key.json",
        base=base,
    )
    assert ok is True


def test_sidecar_rejects_tamper(running_sidecar):
    base = running_sidecar
    data = json.loads(GOLDEN.read_text())
    proof = json.loads(json.dumps(data["proof"]))
    proof["pi_a"][0] = str(int(proof["pi_a"][0]) + 1)
    ok = sidecar.verify_via_sidecar(
        proof,
        data["public_inputs"],
        ASSET_DIR / "verification_key.json",
        base=base,
    )
    assert ok is False
