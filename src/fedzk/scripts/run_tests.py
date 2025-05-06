#!/usr/bin/env python
"""
Comprehensive test runner for FedZK.

This script runs all tests and generates a comprehensive report.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in Python path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class TestReporter:
    """Handles test reporting and visualization."""

    def __init__(self, output_dir: str):
        """
        Initialize the test reporter.
        
        Args:
            output_dir: Directory to save test reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_data = {
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }

    def add_test_result(self, test_type: str, result: Dict[str, Any]) -> None:
        """
        Add test result to the report.
        
        Args:
            test_type: Type of test (unit, integration, benchmark)
            result: Test result data
        """
        if test_type not in self.report_data["results"]:
            self.report_data["results"][test_type] = []

        self.report_data["results"][test_type].append(result)

    def save_report(self) -> str:
        """
        Save the test report to a JSON file.
        
        Returns:
            Path to the saved report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"test_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(self.report_data, f, indent=2)

        return str(report_file)

    def generate_html_report(self) -> str:
        """
        Generate HTML report from test results.
        
        Returns:
            Path to the HTML report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_file = self.output_dir / f"test_report_{timestamp}.html"

        # Get summary statistics
        summary = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "error_tests": 0,
            "duration": 0
        }

        for test_type, results in self.report_data["results"].items():
            for result in results:
                if "summary" in result:
                    summary["total_tests"] += result["summary"].get("total", 0)
                    summary["passed_tests"] += result["summary"].get("passed", 0)
                    summary["failed_tests"] += result["summary"].get("failed", 0)
                    summary["skipped_tests"] += result["summary"].get("skipped", 0)
                    summary["error_tests"] += result["summary"].get("error", 0)
                    summary["duration"] += result["summary"].get("duration", 0)

        # Generate HTML
        with open(html_file, "w") as f:
            f.write("<!DOCTYPE html>\n")
            f.write("<html lang='en'>\n")
            f.write("<head>\n")
            f.write("  <meta charset='UTF-8'>\n")
            f.write("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n")
            f.write("  <title>FedZK Test Report</title>\n")
            f.write("  <style>\n")
            f.write("    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }\n")
            f.write("    h1, h2, h3 { color: #333; }\n")
            f.write("    .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }\n")
            f.write("    .passed { color: green; }\n")
            f.write("    .failed { color: red; }\n")
            f.write("    .skipped { color: orange; }\n")
            f.write("    .error { color: darkred; }\n")
            f.write("    table { border-collapse: collapse; width: 100%; }\n")
            f.write("    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
            f.write("    th { background-color: #f2f2f2; }\n")
            f.write("    tr:nth-child(even) { background-color: #f9f9f9; }\n")
            f.write("  </style>\n")
            f.write("</head>\n")
            f.write("<body>\n")

            f.write("  <h1>FedZK Test Report</h1>\n")
            f.write(f"  <p>Generated on: {self.report_data['timestamp']}</p>\n")

            # Summary section
            f.write("  <div class='summary'>\n")
            f.write("    <h2>Summary</h2>\n")
            f.write(f"    <p>Total tests: {summary['total_tests']}</p>\n")
            f.write(f"    <p class='passed'>Passed: {summary['passed_tests']}</p>\n")
            f.write(f"    <p class='failed'>Failed: {summary['failed_tests']}</p>\n")
            f.write(f"    <p class='skipped'>Skipped: {summary['skipped_tests']}</p>\n")
            f.write(f"    <p class='error'>Error: {summary['error_tests']}</p>\n")
            f.write(f"    <p>Total duration: {summary['duration']:.2f} seconds</p>\n")
            f.write("  </div>\n")

            # Results by test type
            for test_type, results in self.report_data["results"].items():
                f.write(f"  <h2>{test_type.title()} Tests</h2>\n")

                for i, result in enumerate(results):
                    f.write(f"  <h3>Run {i+1}</h3>\n")

                    if "summary" in result:
                        f.write("  <table>\n")
                        f.write("    <tr><th>Metric</th><th>Value</th></tr>\n")
                        for key, value in result["summary"].items():
                            f.write(f"    <tr><td>{key.title()}</td><td>{value}</td></tr>\n")
                        f.write("  </table>\n")

                    if "details" in result and result["details"]:
                        f.write("  <h4>Test Details</h4>\n")
                        f.write("  <table>\n")
                        f.write("    <tr><th>Name</th><th>Outcome</th><th>Duration</th></tr>\n")

                        for detail in result["details"]:
                            outcome_class = detail.get("outcome", "")
                            f.write("    <tr>\n")
                            f.write(f"      <td>{detail.get('name', '')}</td>\n")
                            f.write(f"      <td class='{outcome_class}'>{outcome_class}</td>\n")
                            f.write(f"      <td>{detail.get('duration', 0):.3f}s</td>\n")
                            f.write("    </tr>\n")

                        f.write("  </table>\n")

            f.write("</body>\n")
            f.write("</html>\n")

        return str(html_file)


def parse_pytest_output(output: str) -> Dict[str, Any]:
    """
    Parse pytest output to extract test results.
    
    Args:
        output: pytest output
        
    Returns:
        Dictionary with test result data
    """
    lines = output.splitlines()

    # Extract summary using regex for pytest final summary line
    summary = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "error": 0, "duration": 0.0}
    for line in lines:
        # Match lines like '=== 1 failed, 6 passed in 1.05s ==='
        if "passed" in line or "failed" in line:
            parts = re.findall(r"(\d+)\s+(passed|failed|skipped)", line)
            for num_str, key in parts:
                summary[key] = int(num_str)
            # Total tests
            summary["total"] = summary.get("passed", 0) + summary.get("failed", 0) + summary.get("skipped", 0) + summary.get("error", 0)
            # Extract duration in seconds
            m = re.search(r"in\s+([0-9\.]+)s", line)
            if m:
                summary["duration"] = float(m.group(1))
            break

    # Extract details for each test
    details = []
    current_test = None
    capture_details = False

    for line in lines:
        if line.startswith("test") and "::" in line:
            parts = line.split(" ")
            current_test = {
                "name": parts[0],
                "outcome": "unknown"
            }
            capture_details = True

            if "PASSED" in line:
                current_test["outcome"] = "passed"
                try:
                    current_test["duration"] = float(line.split("[")[1].split("]")[0])
                except (IndexError, ValueError):
                    current_test["duration"] = 0.0
                details.append(current_test)
                current_test = None
                capture_details = False
            elif "FAILED" in line:
                current_test["outcome"] = "failed"
                try:
                    current_test["duration"] = float(line.split("[")[1].split("]")[0])
                except (IndexError, ValueError):
                    current_test["duration"] = 0.0
                details.append(current_test)
                current_test = None
                capture_details = False
            elif "SKIPPED" in line:
                current_test["outcome"] = "skipped"
                try:
                    current_test["duration"] = float(line.split("[")[1].split("]")[0])
                except (IndexError, ValueError):
                    current_test["duration"] = 0.0
                details.append(current_test)
                current_test = None
                capture_details = False

    return {
        "summary": summary,
        "details": details
    }


def run_unit_tests(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run unit tests and return results.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary with test results
    """
    print("Running unit tests...")

    cmd = [sys.executable, "-m", "pytest", "-xvs"]

    if args.test_pattern:
        cmd.append(args.test_pattern)

    if args.coverage:
        cmd.extend(["--cov=fedzk", "--cov-report=term"])

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time

    output = result.stdout + result.stderr

    if args.verbose:
        print(output)

    test_results = parse_pytest_output(output)
    test_results["summary"]["duration"] = duration

    return test_results


def run_integration_tests(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run integration tests and return results.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary with test results
    """
    print("Running integration tests...")

    cmd = [sys.executable, "-m", "pytest", "-xvs", "tests/test_integration.py"]

    if args.coverage:
        cmd.extend(["--cov=fedzk", "--cov-report=term"])

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time

    output = result.stdout + result.stderr

    if args.verbose:
        print(output)

    test_results = parse_pytest_output(output)
    test_results["summary"]["duration"] = duration

    return test_results


def validate_circuits_compile() -> bool:
    # Check for pre-built circuit artifacts instead of compiling at runtime
    print("🛠️ Checking presence of pre-built circuit artifacts...")
    required = [
        "src/fedzk/zk/circuits/build/model_update.r1cs",
        "src/fedzk/zk/circuits/build/model_update_secure.r1cs"
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        print("❌ Missing circuit artifacts:", missing)
        return False
    print("✅ Circuit artifacts found.")
    return True


def run_zk_benchmarks(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run ZK benchmarks and return results.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary with benchmark results
    """
    print("Running ZK benchmarks...")

    from fedzk.benchmark.cli import run_zk_benchmark

    # Prepare benchmark arguments namespace
    benchmark_args = argparse.Namespace(
        iterations=args.benchmark_iterations,
        max_inputs=1000,
        memory_profile=args.memory_profile,
        secure=args.secure,
        output_dir=args.output_dir,
        report=True
    )
    benchmark_args.r1cs_file = "src/fedzk/zk/circuits/build/model_update.r1cs"
    benchmark_args.secure_r1cs_file = "src/fedzk/zk/circuits/build/model_update_secure.r1cs"

    # Attempt to run ZK benchmarks, catch errors to prevent crashing
    try:
        run_zk_benchmark(benchmark_args)
    except Exception as e:
        return {"error": str(e)}

    # Load the most recent benchmark result
    result_files = list(Path(args.output_dir).glob("zk_benchmark_*.json"))
    result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    if not result_files:
        return {"error": "No benchmark results found"}

    with open(result_files[0], "r") as f:
        benchmark_data = json.load(f)

    return {
        "benchmark_data": benchmark_data,
        "file_path": str(result_files[0])
    }


def run_e2e_benchmarks(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run end-to-end benchmarks and return results.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary with benchmark results
    """
    if not args.e2e:
        return {"skipped": True}

    print("Running end-to-end benchmarks...")

    from fedzk.benchmark.cli import run_e2e_benchmark

    # Create a namespace with the benchmark arguments
    benchmark_args = argparse.Namespace(
        clients=3,
        rounds=2,
        model="mlp",
        dataset="mnist",
        zk_enabled=True,
        secure=args.secure,
        memory_profile=args.memory_profile,
        output_dir=args.output_dir,
        report=True
    )

    run_e2e_benchmark(benchmark_args)

    # Load the most recent benchmark result
    result_files = list(Path(args.output_dir).glob("e2e_benchmark_*.json"))
    result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    if not result_files:
        return {"error": "No benchmark results found"}

    with open(result_files[0], "r") as f:
        benchmark_data = json.load(f)

    return {
        "benchmark_data": benchmark_data,
        "file_path": str(result_files[0])
    }


def setup_parser() -> argparse.ArgumentParser:
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive tests and benchmarks for FedZK."
    )

    parser.add_argument("--unit", action="store_true",
                       help="Run unit tests")
    parser.add_argument("--integration", action="store_true",
                       help="Run integration tests")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run ZK benchmarks")
    parser.add_argument("--e2e", action="store_true",
                       help="Run end-to-end benchmarks")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests and benchmarks")
    parser.add_argument("--test-pattern", type=str,
                       help="Pattern for selecting test files")
    parser.add_argument("--coverage", action="store_true",
                       help="Generate coverage report")
    parser.add_argument("--secure", action="store_true",
                       help="Use secure zero-knowledge circuit")
    parser.add_argument("--memory-profile", action="store_true",
                       help="Enable memory profiling")
    parser.add_argument("--benchmark-iterations", type=int, default=3,
                       help="Number of iterations for benchmarks")
    parser.add_argument("--output-dir", type=str, default="test_results",
                       help="Directory to save test results")
    parser.add_argument("--report", action="store_true",
                       help="Generate HTML report")
    parser.add_argument("--verbose", action="store_true",
                       help="Show verbose output")
    parser.add_argument("--check-circuits", action="store_true",
                       help="Check circuit artifacts")

    return parser


def main() -> None:
    """Main function."""
    parser = setup_parser()
    args = parser.parse_args()

    # If no tests specified, show help
    if not any([args.unit, args.integration, args.benchmark, args.e2e, args.all]):
        parser.print_help()
        sys.exit(1)

    # If --all specified, run all tests
    if args.all:
        args.unit = True
        args.integration = True
        args.benchmark = True
        args.e2e = True

    # Create reporter
    reporter = TestReporter(args.output_dir)

    # Run tests and collect results
    if args.unit:
        unit_results = run_unit_tests(args)
        reporter.add_test_result("unit", unit_results)
        print(f"Unit tests complete: {unit_results['summary']['passed']} passed, "
              f"{unit_results['summary'].get('failed', 0)} failed, "
              f"{unit_results['summary'].get('skipped', 0)} skipped\n")

    if args.integration:
        integration_results = run_integration_tests(args)
        reporter.add_test_result("integration", integration_results)
        print(f"Integration tests complete: {integration_results['summary']['passed']} passed, "
              f"{integration_results['summary'].get('failed', 0)} failed, "
              f"{integration_results['summary'].get('skipped', 0)} skipped\n")

    if args.benchmark:
        zk_results = {}
        if validate_circuits_compile():
            print("✅ Running ZK benchmarks...")
            zk_results = run_zk_benchmarks(args)
            print("ZK benchmarks complete\n")
        else:
            zk_results = {
                "status": "skipped",
                "reason": "Circom circuit compilation failed. Check line endings, version (>=2.2.2), or syntax."
            }
            print("⚠️ ZK benchmarks skipped due to failed Circom compilation.\n")
        reporter.add_test_result("zk_benchmark", zk_results)

    if args.e2e:
        e2e_results = run_e2e_benchmarks(args)
        reporter.add_test_result("e2e_benchmark", e2e_results)
        print("End-to-end benchmarks complete\n")

    # Save report
    report_file = reporter.save_report()
    print(f"Test report saved to: {report_file}")

    # Generate HTML report if requested
    if args.report:
        html_report = reporter.generate_html_report()
        print(f"HTML report generated: {html_report}")


if __name__ == "__main__":
    main()
