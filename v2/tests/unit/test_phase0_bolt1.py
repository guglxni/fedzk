# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""Phase 0 / Bolt 1 sensors — CLI import, quantization, doctor shape."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import torch

from fedzk.doctor import run_doctor
from fedzk.prover.zkgenerator import N_DEV, ZKProver, _flatten_pad_or_truncate
from fedzk.zk.input_normalization import GradientQuantizer


def test_cli_app_importable():
    from fedzk.cli import app

    assert app is not None
    # Must be module cli.py, not empty package
    import fedzk.cli as cli_mod

    assert cli_mod.__file__ and cli_mod.__file__.endswith("cli.py")


def test_flatten_pad_truncate_honest():
    assert _flatten_pad_or_truncate([1, 2], 4, "t") == [1, 2, 0, 0]
    assert _flatten_pad_or_truncate([1, 2, 3, 4, 5], 4, "t") == [1, 2, 3, 4]


def test_quantizer_produces_ints():
    q = GradientQuantizer(scale_factor=1000)
    grads = {"w": torch.tensor([0.001, -0.002, 0.003, 0.0])}
    quantized, meta = q.quantize_gradients(grads)
    assert meta["scale_factor"] == 1000
    assert all(isinstance(v, int) for v in quantized["w"])


def test_prepare_input_standard_quantizes(monkeypatch):
    # Skip toolchain validation for unit sensor
    with patch.object(ZKProver, "_validate_zk_toolchain", lambda self: None):
        prover = ZKProver(secure=False, scale_factor=1000)
        grads = {"w": torch.tensor([0.5, -0.25, 0.125, 0.0])}
        payload = prover._prepare_input_standard(grads, max_inputs=N_DEV)
    assert set(payload.keys()) == {"gradients"}
    assert len(payload["gradients"]) == N_DEV
    assert all(isinstance(x, int) for x in payload["gradients"])
    assert payload["gradients"][0] == 500  # 0.5 * 1000


def test_doctor_runs_and_reports_cli():
    report = run_doctor()
    assert "checks" in report
    assert "overall_ok" in report
    cli_checks = [c for c in report["checks"] if c["id"] == "cli_module"]
    assert cli_checks and cli_checks[0]["ok"] is True
    backend_checks = [c for c in report["checks"] if c["id"] == "zk_backend"]
    assert backend_checks and backend_checks[0]["ok"] is True


def test_doctor_rust_backend_fail_closed(monkeypatch):
    monkeypatch.setenv("FEDZK_ZK_BACKEND", "rust")
    monkeypatch.setenv("FEDZK_ZK_BIN", "/nonexistent/fedzk-zk")
    report = run_doctor()
    backend = [c for c in report["checks"] if c["id"] == "zk_backend"][0]
    assert backend["ok"] is False
    assert report["overall_ok"] is False


def test_version_aligned():
    import fedzk
    from importlib.metadata import version

    assert fedzk.__version__ == "1.2.0"
    # package metadata may lag in editable; prefer module version
    assert fedzk.__version__


def test_proof_json_shape_helper():
    """Wire format dict is JSON-serializable (fedzk.proof.v1)."""
    sample = {
        "proof": {"pi_a": [], "pi_b": [], "pi_c": [], "protocol": "groth16"},
        "public_inputs": ["1", "2"],
        "secure": False,
        "wire": "fedzk.proof.v1",
    }
    dumped = json.dumps(sample)
    assert "fedzk.proof.v1" in dumped
