import os
import json
import argparse
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

import pytest
import torch
import requests

from fedzk.cli import generate_command, load_gradient_data, setup_parser
from fedzk.prover.zkgenerator import ZKProver

# Setup logging capture for testing
@pytest.fixture
def setup_logging_capture():
    logger = logging.getLogger()
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    
    log_records = []
    
    class TestLogHandler(logging.Handler):
        def emit(self, record):
            log_records.append(record)
    
    handler = TestLogHandler()
    logger.addHandler(handler)
    
    yield log_records
    
    logger.removeHandler(handler)
    logger.setLevel(previous_level)

@pytest.fixture
def setup_test_environment(tmp_path, monkeypatch):
    # Create dummy input file
    input_file = tmp_path / "gradients.json"
    input_data = {"param1": [1.0, 2.0, 3.0]}
    with open(input_file, "w") as f:
        json.dump(input_data, f)
    
    # Mock load_gradient_data to return tensor
    monkeypatch.setattr(
        'fedzk.cli.load_gradient_data', 
        lambda path: {"param1": torch.tensor([1.0, 2.0, 3.0])}
    )
    
    # Setup fake ZK directory and files
    zk_dir = tmp_path / "zk"
    zk_dir.mkdir()
    
    # Create fake circuit and key files
    (zk_dir / "model_update.wasm").touch()
    (zk_dir / "proving_key.zkey").touch()
    (zk_dir / "model_update_secure.wasm").touch()
    (zk_dir / "proving_key_secure.zkey").touch()
    
    # Mock Path.exists to always return True for ZK files
    original_exists = Path.exists
    def mock_exists(self):
        if "zk" in str(self):
            return True
        return original_exists(self)
    
    monkeypatch.setattr(Path, "exists", mock_exists)
    
    # Mock ZKProver to return known values
    def mock_generate_proof(self, grads):
        return "local_proof", ["local_public_input"]
    
    def mock_generate_secure_proof(self, grads, max_norm=100.0, min_active=1):
        return "local_secure_proof", {"public_inputs": ["local_secure_input"]}
    
    monkeypatch.setattr(ZKProver, "generate_real_proof", mock_generate_proof)
    monkeypatch.setattr(ZKProver, "generate_real_proof_secure", mock_generate_secure_proof)
    
    # Create a function to generate args
    def create_args(
        mpc_server="http://fake-server:9999",
        api_key="test-key",
        secure=False,
        fallback_disabled=False,
        fallback_mode="silent",
        output=None
    ):
        if output is None:
            output = str(tmp_path / "output.json")
            
        return argparse.Namespace(
            input=str(input_file),
            output=output,
            secure=secure,
            batch=False,
            chunk_size=4,
            max_norm=100.0,
            min_active=1,
            mpc_server=mpc_server,
            api_key=api_key,
            fallback_disabled=fallback_disabled,
            fallback_mode=fallback_mode
        )
    
    return tmp_path, create_args


class MockFailedResponse:
    """Simulates a failed HTTP response from requests"""
    
    def __init__(self, status_code=500, reason="Internal Server Error"):
        self.status_code = status_code
        self.reason = reason
    
    def raise_for_status(self):
        raise requests.exceptions.HTTPError(
            f"{self.status_code} {self.reason}",
            response=self
        )


def test_fallback_disabled(setup_test_environment, capsys):
    """Test that --fallback-disabled prevents fallback to local proof"""
    tmp_path, create_args = setup_test_environment
    
    # Create args with fallback disabled
    args = create_args(fallback_disabled=True)
    output_file = Path(args.output)
    
    # Mock requests.post to simulate failure
    with patch('requests.post', side_effect=requests.exceptions.ConnectionError("Connection refused")):
        # Run generate command, should fail with error
        result = generate_command(args)
        
        # Check results
        assert result == 1, "Command should fail with exit code 1"
        
        # Check that error message was printed
        captured = capsys.readouterr()
        assert "Error: MPC generate_proof failed" in captured.out
        assert "Connection refused" in captured.out
        
        # Output file should not exist
        assert not output_file.exists(), "Output file should not be created on failure"


def test_fallback_mode_strict(setup_test_environment, capsys):
    """Test that fallback_mode=strict prevents fallback to local proof"""
    tmp_path, create_args = setup_test_environment
    
    # Create args with fallback mode set to strict
    args = create_args(fallback_mode="strict")
    output_file = Path(args.output)
    
    # Mock requests.post to simulate HTTP 500
    with patch('requests.post', return_value=MockFailedResponse()):
        # Run generate command, should fail with error
        result = generate_command(args)
        
        # Check results
        assert result == 1, "Command should fail with exit code 1"
        
        # Check that error message was printed
        captured = capsys.readouterr()
        assert "Error: MPC generate_proof failed" in captured.out
        assert "500 Internal Server Error" in captured.out
        
        # Output file should not exist
        assert not output_file.exists(), "Output file should not be created on failure"


def test_fallback_mode_warn(setup_test_environment, capsys, setup_logging_capture):
    """Test that fallback_mode=warn logs a warning but falls back to local proof"""
    tmp_path, create_args = setup_test_environment
    log_records = setup_logging_capture
    
    # Create args with fallback mode set to warn
    args = create_args(fallback_mode="warn")
    output_file = Path(args.output)
    
    # Mock requests.post to simulate network failure
    with patch('requests.post', side_effect=requests.exceptions.ConnectionError("Connection refused")):
        # Run generate command, should succeed with fallback
        result = generate_command(args)
        
        # Check results
        assert result == 0, "Command should succeed with fallback"
        
        # Check warning was printed
        captured = capsys.readouterr()
        assert "MPC server failed" in captured.out
        assert "falling back to local ZK proof" in captured.out
        
        # Output file should exist with local proof
        assert output_file.exists(), "Output file should be created with local proof"
        
        # Check file content
        with open(output_file) as f:
            proof_data = json.load(f)
        assert proof_data.get("proof") == "local_proof"
        assert proof_data.get("public_inputs") == ["local_public_input"]
        
        # Check telemetry was logged
        assert len(log_records) > 0, "Fallback event should be logged"
        
        # Find fallback log record
        fallback_record = None
        for record in log_records:
            if "Fallback event" in record.message:
                fallback_record = record
                break
        
        assert fallback_record is not None, "Fallback event not found in logs"
        
        # Check telemetry content
        log_data = json.loads(fallback_record.message.replace("Fallback event: ", ""))
        assert "timestamp" in log_data
        assert log_data["exception"] == "ConnectionError"
        assert log_data["circuit"] == "standard"
        assert "cli_command" in log_data


def test_fallback_mode_silent(setup_test_environment, capsys, setup_logging_capture):
    """Test that fallback_mode=silent doesn't log a warning but still falls back"""
    tmp_path, create_args = setup_test_environment
    log_records = setup_logging_capture
    
    # Create args with fallback mode set to silent
    args = create_args(fallback_mode="silent")
    output_file = Path(args.output)
    
    # Mock requests.post to simulate timeout
    with patch('requests.post', side_effect=requests.exceptions.Timeout("Request timed out")):
        # Run generate command, should succeed with fallback
        result = generate_command(args)
        
        # Check results
        assert result == 0, "Command should succeed with fallback"
        
        # Check no warning was printed
        captured = capsys.readouterr()
        assert "MPC server failed" not in captured.out
        assert "falling back to local ZK proof" not in captured.out
        
        # Output file should exist with local proof
        assert output_file.exists(), "Output file should be created with local proof"
        with open(output_file) as f:
            proof_data = json.load(f)
        assert proof_data.get("proof") == "local_proof"
        
        # Telemetry should still be logged
        assert len(log_records) > 0, "Fallback event should be logged even in silent mode"
        
        fallback_record = None
        for record in log_records:
            if "Fallback event" in record.message:
                fallback_record = record
                break
        
        assert fallback_record is not None, "Fallback event not found in logs"


def test_fallback_env_variable(setup_test_environment, capsys, monkeypatch):
    """Test that FALLBACK_MODE environment variable controls fallback behavior"""
    tmp_path, create_args = setup_test_environment
    
    # Set environment variable
    monkeypatch.setenv("FALLBACK_MODE", "warn")
    
    # Force re-creation of parser to pick up new env var
    parser = setup_parser()
    
    # Create args without explicitly setting fallback_mode (should use env var)
    # First need to create a base args object
    base_args = create_args()
    # Then extract the args from the parser with defaults from env var
    parser_args, unknown = parser.parse_known_args([
        "generate",
        "--input", base_args.input,
        "--output", base_args.output,
        "--mpc-server", base_args.mpc_server,
        "--api-key", base_args.api_key
    ])
    
    # Verify that the fallback_mode is set to "warn" from env var
    assert parser_args.fallback_mode == "warn", "fallback_mode should be set from FALLBACK_MODE env var"
    
    # Continue with the test using our manually merged args
    args = argparse.Namespace(**vars(parser_args))
    output_file = Path(args.output)
    
    # Mock requests.post to simulate network failure
    with patch('requests.post', side_effect=requests.exceptions.ConnectionError("Connection refused")):
        # Run generate command, should succeed with fallback and warning
        result = generate_command(args)
        
        # Check results
        assert result == 0, "Command should succeed with fallback"
        
        # Check warning was printed (because env var set warning mode)
        captured = capsys.readouterr()
        assert "MPC server failed" in captured.out
        assert "falling back to local ZK proof" in captured.out
        
        # Output file should exist with local proof
        assert output_file.exists(), "Output file should be created with local proof"


def test_secure_circuit_fallback(setup_test_environment, capsys):
    """Test fallback with secure circuit type"""
    tmp_path, create_args = setup_test_environment
    
    # Create args with secure=True
    args = create_args(secure=True, fallback_mode="warn")
    output_file = Path(args.output)
    
    # Mock requests.post to simulate failure
    with patch('requests.post', side_effect=requests.exceptions.RequestException("Request failed")):
        # Run generate command, should succeed with fallback
        result = generate_command(args)
        
        # Check results
        assert result == 0, "Command should succeed with fallback"
        
        # Check that secure circuit proof was generated
        assert output_file.exists()
        with open(output_file) as f:
            proof_data = json.load(f)
        assert proof_data.get("proof") == "local_secure_proof"
        assert "public_inputs" in proof_data 