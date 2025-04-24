import argparse
import json
from pathlib import Path

import pytest
import torch

from fedzk.cli import generate_command, load_gradient_data
from fedzk.prover.zkgenerator import ZKProver

class DummyResponse:
    def raise_for_status(self):
        raise Exception("Network error")

@pytest.fixture(autouse=True)
def patch_environment(monkeypatch, tmp_path):
    # Create a dummy input JSON for single proof mode
    data = {"param1": [1.0, 2.0]}
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps(data))
    # Patch load_gradient_data to load from our dummy file
    monkeypatch.setattr('fedzk.cli.load_gradient_data', lambda path: {"param1": torch.tensor([1.0, 2.0])})
    # Patch requests.post to simulate network failure
    import requests
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: DummyResponse())
    # Stub local proof generation to known output
    monkeypatch.setattr(ZKProver, 'generate_real_proof', lambda self, grad: ("local_proof", ["local_sig"]))
    return input_file, tmp_path

def test_mpc_fallback(tmp_path, patch_environment, capsys):
    input_file, tmp_path = patch_environment
    output_file = tmp_path / "out.json"

    # Prepare args with mpc_server and fallback enabled
    args = argparse.Namespace(
        input=str(input_file),
        output=str(output_file),
        secure=False,
        batch=False,
        chunk_size=4,
        max_norm=100.0,
        min_active=1,
        mpc_server="http://bad",
        api_key="testkey",
        fallback_disabled=False,
        fallback_mode="warn"
    )

    # Run generate_command, expect fallback to local proof
    rc = generate_command(args)
    assert rc == 0

    captured = capsys.readouterr()
    # Check that fallback warning was printed
    assert "falling back to local ZK proof" in captured.out

    # Verify output file contains local proof
    assert output_file.exists()
    result = json.loads(output_file.read_text())
    # Only verify essential fields; additional flags may be present
    assert result.get("proof") == "local_proof"
    assert result.get("public_inputs") == ["local_sig"]

def test_mpc_no_fallback(tmp_path, patch_environment, capsys):
    input_file, tmp_path = patch_environment
    output_file = tmp_path / "out2.json"

    # Prepare args with mpc_server and fallback disabled
    args = argparse.Namespace(
        input=str(input_file),
        output=str(output_file),
        secure=False,
        batch=False,
        chunk_size=4,
        max_norm=100.0,
        min_active=1,
        mpc_server="http://bad",
        api_key="testkey",
        fallback_disabled=True,
        fallback_mode="warn"
    )

    # Run generate_command, expect failure (no fallback)
    rc = generate_command(args)
    assert rc == 1

    captured = capsys.readouterr()
    # Should report MPC error without fallback message
    assert "MPC generate_proof failed" in captured.out
    assert not output_file.exists() 