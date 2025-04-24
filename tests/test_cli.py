"""Tests for the CLI module."""

import pytest
from typer.testing import CliRunner

from fedzk.cli import app


runner = CliRunner()


def test_client_train_command():
    """Test the client train command."""
    result = runner.invoke(app, ["client", "train"])
    assert result.exit_code == 0
    assert "Training model locally" in result.stdout


def test_client_prove_command():
    """Test the client prove command."""
    result = runner.invoke(app, ["client", "prove"])
    assert result.exit_code == 0
    assert "Generating zero-knowledge proof" in result.stdout


def test_benchmark_command():
    """Test the benchmark command."""
    result = runner.invoke(app, ["benchmark", "run", "--help"])
    assert result.exit_code == 0
    assert "Run benchmarks" in result.stdout 