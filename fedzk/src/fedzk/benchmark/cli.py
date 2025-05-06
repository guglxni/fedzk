# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
Command-line interface for running FedZK benchmarks.

This module provides a simple command-line interface for running various
benchmark suites for FedZK.
"""

import argparse
import sys
from pathlib import Path

from fedzk.benchmark.benchmark_zk import ZKBenchmark
from fedzk.benchmark.end_to_end import FedZKBenchmark as EndToEndBenchmark
from fedzk.benchmark.visualization import BenchmarkVisualizer


def setup_parser() -> argparse.ArgumentParser:
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run FedZK benchmarks and visualize results."
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ZK Benchmark command
    zk_parser = subparsers.add_parser("zk", help="Run ZK proof generation and verification benchmarks")
    zk_parser.add_argument("--iterations", type=int, default=5,
                         help="Number of iterations for each operation")
    zk_parser.add_argument("--max-inputs", type=int, default=1000,
                         help="Maximum number of inputs for circuit")
    zk_parser.add_argument("--memory-profile", action="store_true",
                         help="Enable memory profiling")
    zk_parser.add_argument("--secure", action="store_true",
                         help="Use secure zero-knowledge circuit")
    zk_parser.add_argument("--output-dir", type=str, default="benchmark_results",
                         help="Directory to save benchmark results")
    zk_parser.add_argument("--report", action="store_true",
                         help="Generate HTML report of results")

    # End-to-end benchmarks command
    e2e_parser = subparsers.add_parser("e2e", help="Run end-to-end federated learning benchmarks")
    e2e_parser.add_argument("--clients", type=int, default=3,
                          help="Number of clients")
    e2e_parser.add_argument("--rounds", type=int, default=3,
                          help="Number of federated learning rounds")
    e2e_parser.add_argument("--model", type=str, choices=["mlp", "cnn", "linear"], default="mlp",
                          help="Model architecture to use")
    e2e_parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10", "synthetic"],
                          default="mnist", help="Dataset to use")
    e2e_parser.add_argument("--zk-enabled", action="store_true",
                          help="Enable zero-knowledge proofs")
    e2e_parser.add_argument("--secure", action="store_true",
                          help="Use secure zero-knowledge circuit")
    e2e_parser.add_argument("--memory-profile", action="store_true",
                          help="Enable memory profiling")
    e2e_parser.add_argument("--output-dir", type=str, default="benchmark_results",
                          help="Directory to save benchmark results")
    e2e_parser.add_argument("--report", action="store_true",
                          help="Generate HTML report of results")

    # Visualization command
    vis_parser = subparsers.add_parser("visualize", help="Visualize benchmark results")
    vis_parser.add_argument("--results-dir", type=str, required=True,
                          help="Directory containing benchmark results")
    vis_parser.add_argument("--result-file", type=str,
                          help="Specific result file to visualize (optional)")
    vis_parser.add_argument("--compare", action="store_true",
                          help="Compare multiple benchmark results")
    vis_parser.add_argument("--report", action="store_true",
                          help="Generate HTML report")
    vis_parser.add_argument("--output-dir", type=str,
                          help="Directory to save visualizations (default is results directory)")

    return parser


def run_zk_benchmark(args: argparse.Namespace) -> None:
    """
    Run ZK benchmark with the specified arguments.
    
    Args:
        args: Command-line arguments
    """
    # Create output directory for benchmark results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the ZK benchmarking suite with the correct output directory
    benchmark = ZKBenchmark(output_dir=args.output_dir)

    # Run all benchmarks with the specified number of iterations
    benchmark.run_all_benchmarks(num_iterations=args.iterations)


def run_e2e_benchmark(args: argparse.Namespace) -> None:
    """
    Run end-to-end benchmark with the specified arguments.
    
    Args:
        args: Command-line arguments
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = EndToEndBenchmark(
        num_clients=args.clients,
        num_rounds=args.rounds,
        model_type=args.model,
        dataset=args.dataset,
        zk_enabled=args.zk_enabled,
        secure=args.secure,
        profile_memory=args.memory_profile
    )

    print(f"Running end-to-end benchmarks with {args.clients} clients and {args.rounds} rounds...")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Zero-knowledge: {'enabled' if args.zk_enabled else 'disabled'}")
    if args.zk_enabled:
        print(f"Using {'secure' if args.secure else 'standard'} circuit")
    print(f"Memory profiling: {'enabled' if args.memory_profile else 'disabled'}")

    results = benchmark.run_benchmarks()

    # Save results
    result_file = results.save_to_file(output_dir)
    print(f"\nBenchmark results saved to: {result_file}")

    # Print summary
    results.print_summary()

    # Generate report if requested
    if args.report:
        visualizer = BenchmarkVisualizer(output_dir)
        report_path = visualizer.generate_report(
            result_file=Path(result_file).name
        )
        print(f"\nBenchmark report generated: {report_path}")


def visualize_results(args: argparse.Namespace) -> None:
    """
    Visualize benchmark results with the specified arguments.
    
    Args:
        args: Command-line arguments
    """
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' does not exist.")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = BenchmarkVisualizer(results_dir)

    if args.compare:
        # Compare multiple benchmark results
        files = list(results_dir.glob("*.json"))
        if not files:
            print(f"Error: No JSON files found in '{results_dir}'.")
            sys.exit(1)

        # Load all result files
        visualizer.load_results()

        # Get file names
        file_names = [f.name for f in files]

        # Create comparison plot
        print(f"Comparing {len(file_names)} benchmark results...")

        plot_path = output_dir / "benchmark_comparison.png"
        visualizer.compare_benchmarks(
            file_names=file_names,
            save_path=plot_path,
            show_plot=True
        )

        print(f"Comparison plot saved to: {plot_path}")

    elif args.result_file:
        # Visualize specific result file
        file_path = results_dir / args.result_file

        if not file_path.exists():
            print(f"Error: Result file '{file_path}' does not exist.")
            sys.exit(1)

        # Load the specified result file
        results = visualizer.load_results(args.result_file)

        # Create plot
        print(f"Visualizing benchmark results from '{args.result_file}'...")

        plot_path = output_dir / f"{args.result_file.replace('.json', '.png')}"
        visualizer.plot_operation_durations(
            result_file=args.result_file,
            save_path=plot_path,
            show_plot=True
        )

        print(f"Plot saved to: {plot_path}")

        # Generate report if requested
        if args.report:
            report_path = visualizer.generate_report(
                result_file=args.result_file
            )
            print(f"Benchmark report generated: {report_path}")

    else:
        # Visualize most recent result file
        visualizer.load_results()

        if not visualizer.loaded_results:
            print(f"Error: No JSON files found in '{results_dir}'.")
            sys.exit(1)

        print("Visualizing most recent benchmark results...")

        plot_path = output_dir / "latest_benchmark.png"
        visualizer.plot_operation_durations(
            save_path=plot_path,
            show_plot=True
        )

        print(f"Plot saved to: {plot_path}")

        # Generate report if requested
        if args.report:
            report_path = visualizer.generate_report()
            print(f"Benchmark report generated: {report_path}")


def main():
    """Main entry point for the benchmark CLI."""
    parser = setup_parser()
    args = parser.parse_args()

    if args.command == "zk":
        run_zk_benchmark(args)
    elif args.command == "e2e":
        run_e2e_benchmark(args)
    elif args.command == "visualize":
        visualize_results(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
