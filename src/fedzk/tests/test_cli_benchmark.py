# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

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
    # Check that relevant options are listed - strip ANSI codes for better matching
    combined_output = result.stdout + result.stderr
    # Remove ANSI escape codes for cleaner string matching
    import re
    clean_output = re.sub(r'\x1b\[[0-9;]*m', '', combined_output)
    for opt in ["--clients", "--secure", "--mpc-server", "--output", "--csv"]:
        assert opt in clean_output, f"Option {opt} not found in help output"


def test_cli_no_command():
    """Running CLI without command should print help and exit with code 1."""
    result = run_cli_command([])
    # expecting non-zero exit code (no command provided)
    assert result.returncode != 0
    # Typer usually exits with code 1 or 2 for no command and prints help to stdout or stderr
    assert "Usage:" in result.stdout or "Usage:" in result.stderr

@pytest.mark.parametrize("cmd", ["benchmark"])
def test_invalid_benchmark_command(cmd):
    """Invalid benchmark subcommands (not providing a runnable command) should show help and exit non-zero."""
    args = cmd.split()
    result = run_cli_command(args)
    assert result.returncode != 0 # Top-level 'benchmark' with no subcommand should error
    assert "Usage:" in result.stdout or "Usage:" in result.stderr

# It might be useful to add a new test for 'benchmark run' specifically to check its successful execution with defaults.
# def test_benchmark_run_default_success():
#     result = run_cli_command(["benchmark", "run"])
#     assert result.returncode == 0
#     assert "Benchmark report saved to benchmark_report.json" in result.stdout # Or similar success message



