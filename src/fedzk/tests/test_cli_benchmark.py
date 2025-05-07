# Smoke tests for the FedZK CLI benchmark commands

import subprocess
import sys

import pytest


def run_cli_command(args):
    """
    Run the FedZK CLI entrypoint with the given arguments and return the CompletedProcess.
    """
    return subprocess.run(
        [sys.executable, "-m", "fedzk.cli"] + args,
        capture_output=True,
        text=True
    )


def test_benchmark_help():
    """CLI should display help for 'benchmark run' without errors."""
    result = run_cli_command(["benchmark", "run", "--help"])
    assert result.returncode == 0
    # Check for Typer's help output, which usually includes 'Usage:'
    assert "Usage:" in result.stdout or "Usage:" in result.stderr # Typer might use stderr for help on error
    # Check that relevant options are listed
    for opt in ["--clients", "--secure", "--mpc-server", "--output", "--csv"]:
        assert opt in result.stdout or opt in result.stderr


def test_cli_no_command():
    """Running CLI without command should print help and exit with code 1."""
    result = run_cli_command([])
    # expecting non-zero exit code (no command provided)
    assert result.returncode != 0
    # Typer usually exits with code 1 or 2 for no command and prints help to stdout or stderr
    assert "Usage:" in result.stdout or "Usage:" in result.stderr

@pytest.mark.parametrize("cmd", ["benchmark", "benchmark run"])
def test_invalid_benchmark_command(cmd):
    """Invalid benchmark subcommands should show help and exit code 1."""
    args = cmd.split()
    result = run_cli_command(args)
    assert result.returncode != 0
    # Typer usually exits with code 1 or 2 for bad subcommands and prints help to stdout or stderr
    assert "Usage:" in result.stdout or "Usage:" in result.stderr



