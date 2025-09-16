# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""Tests for the CLI module."""

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
    assert "Run end-to-end benchmark" in result.stdout
