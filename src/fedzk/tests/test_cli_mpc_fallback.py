import json
import os
import subprocess
import sys
import argparse # Keep for dummy_response if needed, but not for calling generate_command

import pytest
import torch
import typer # Import typer
from typer.testing import CliRunner # Import CliRunner

from fedzk.cli import app # Import the Typer app from your CLI module
# For testing, it might be easier to directly import the command if app structure is complex
from fedzk.cli import generate_command # Or wherever generate_command is defined
from fedzk.prover.zkgenerator import ZKProver
from fedzk.mpc.client import MPCClient # Assuming this is the intended client

runner = CliRunner()

# Dummy response for requests.post mock
class DummyResponse:
    status_code = 500  # Simulate server error
    def raise_for_status(self):
        raise requests.exceptions.HTTPError("MPC Server down")
    def json(self):
        return {"error": "MPC Server down"}

@pytest.fixture(autouse=True)
def patch_environment(monkeypatch, tmp_path):
    data = {"param1": [1.0, 2.0]} # Example gradient data
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps(data))
    monkeypatch.setattr("fedzk.cli.load_gradient_data", lambda path: {"param1": torch.tensor([1.0, 2.0])})
    import requests
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: DummyResponse())
    # Patch the ZKProver's proof generation directly, not on the class if instance is used
    # This depends on how MPCClient uses ZKProver for fallback
    # For the MPCClient stub, local generation is also stubbed, so this might not be hit by MPCClient calls
    # However, if generate_command itself calls ZKProver directly in some paths, this is needed.
    def mock_zk_generate_proof(self_zk, gradient_dict):
        return {"proof": "local_proof_from_zkprover"}, ["local_sig_from_zkprover"]
    monkeypatch.setattr(ZKProver, "generate_proof", mock_zk_generate_proof)
    return str(input_file), tmp_path # Return path as string

def test_mpc_fallback(tmp_path, patch_environment, capsys):
    input_file_str, _ = patch_environment
    output_file_str = str(tmp_path / "out.json")

    result = runner.invoke(
        app, 
        [
            "generate",
            "--input", input_file_str,
            "--output", output_file_str,
            "--mpc-server", "http://bad",
            "--api-key", "testkey",
            "--fallback-mode", "warn"
        ],
        catch_exceptions=False 
    )
    
    assert "[MPCClient STUB] MPC server failed. WARN: Falling back to local proof." in result.stdout
    assert "[MPCClient STUB] Generating proof locally (stubbed)..." in result.stdout
    assert "Proof saved" in result.stdout 
    assert result.exit_code == 0 

    with open(output_file_str, "r") as f:
        saved_result = json.load(f) # This is the tuple [proof_dict, public_inputs_list]
    assert isinstance(saved_result, list)
    assert len(saved_result) == 2
    proof_dict = saved_result[0]
    public_inputs_list = saved_result[1]
    assert proof_dict["proof"] == "local_stub_proof_from_mpc_client_fallback"
    assert public_inputs_list == ["local_stub_public_inputs"]

def test_mpc_no_fallback(tmp_path, patch_environment, capsys):
    input_file_str, _ = patch_environment
    output_file_str = str(tmp_path / "out2.json")

    result = runner.invoke(
        app, 
        [
            "generate",
            "--input", input_file_str,
            "--output", output_file_str,
            "--mpc-server", "http://bad",
            "--api-key", "testkey",
            "--fallback-disabled"
        ],
        catch_exceptions=True 
    )

    assert "[MPCClient STUB] MPC server failed and fallback is disabled." in result.stdout
    assert "Error generating proof: MPC server call failed and fallback is disabled (stub behavior)" in result.stdout
    assert result.exit_code == 1 
    assert not os.path.exists(output_file_str) # File should not be created on error
