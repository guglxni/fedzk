# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
Visualization utilities for FedZK benchmark results.

This module provides tools to visualize and compare benchmark results.
"""

import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np


class BenchmarkVisualizer:
    """Visualizes benchmark results from JSON files."""

    def __init__(self, results_dir: Union[str, Path]):
        """
        Initialize the benchmark visualizer.
        
        Args:
            results_dir: Directory containing benchmark result JSON files
        """
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise ValueError(f"Results directory {results_dir} does not exist")

        self.result_files = list(self.results_dir.glob("*.json"))
        self.loaded_results = {}

    def load_results(self, filename: Optional[str] = None) -> Dict:
        """
        Load benchmark results from a file.
        
        Args:
            filename: Name of the results file to load, or None to load all files
            
        Returns:
            Dictionary with loaded benchmark results
        """
        if filename:
            file_path = self.results_dir / filename
            if not file_path.exists():
                raise ValueError(f"Results file {filename} does not exist")

            with open(file_path, "r") as f:
                results = json.load(f)

            self.loaded_results[filename] = results
            return results

        # Load all result files
        for file_path in self.result_files:
            try:
                with open(file_path, "r") as f:
                    results = json.load(f)
                    self.loaded_results[file_path.name] = results
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")

        return self.loaded_results

    def plot_operation_durations(self,
                               operation_names: Optional[List[str]] = None,
                               result_file: Optional[str] = None,
                               figsize: tuple = (12, 8),
                               save_path: Optional[Union[str, Path]] = None,
                               show_plot: bool = True) -> plt.Figure:
        """
        Plot the durations of benchmark operations.
        
        Args:
            operation_names: Names of operations to plot, or None for all
            result_file: Name of the result file to use, or None to use most recent
            figsize: Figure size (width, height) in inches
            save_path: Path to save the plot, or None to not save
            show_plot: Whether to display the plot
            
        Returns:
            The matplotlib Figure object
        """
        # Load results if not already loaded
        if not self.loaded_results:
            self.load_results()

        # Use the most recent result file if not specified
        if result_file is None:
            file_timestamps = [
                (name, self._extract_timestamp(res))
                for name, res in self.loaded_results.items()
            ]
            if not file_timestamps:
                raise ValueError("No result files loaded")
            result_file = max(file_timestamps, key=lambda x: x[1])[0]

        results = self.loaded_results.get(result_file)
        if not results:
            raise ValueError(f"Results file {result_file} not loaded")

        # Extract summary data
        summary = results.get("summary", {})
        if not summary:
            raise ValueError(f"No summary data found in {result_file}")

        # Filter operations if specified
        if operation_names:
            summary = {k: v for k, v in summary.items() if k in operation_names}

        # Create data for plotting
        ops = []
        means = []
        mins = []
        maxs = []
        stds = []

        for op_name, stats in summary.items():
            if "error" in stats:
                continue

            ops.append(op_name)
            means.append(stats["mean"])
            mins.append(stats["min"])
            maxs.append(stats["max"])
            stds.append(stats["std"])

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(ops))
        bar_width = 0.7

        # Plot bars with error bars
        bars = ax.bar(x, means, bar_width, yerr=stds,
                    capsize=5, alpha=0.7, label="Mean Duration")

        # Add min/max markers
        for i, (min_val, max_val) in enumerate(zip(mins, maxs)):
            ax.plot([i, i], [min_val, max_val], "k-", alpha=0.5)
            ax.plot(i, min_val, "kv", alpha=0.5)
            ax.plot(i, max_val, "k^", alpha=0.5)

        # Add labels and title
        benchmark_name = results.get("benchmark_name", "Benchmark")
        ax.set_xlabel("Operations")
        ax.set_ylabel("Duration (seconds)")
        ax.set_title(f"{benchmark_name} Results")
        ax.set_xticks(x)
        ax.set_xticklabels(ops, rotation=45, ha="right")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f"{height:.4f}s",
                   ha="center", va="bottom", rotation=0)

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        # Show if requested
        if show_plot:
            plt.show()

        return fig

    def compare_benchmarks(self,
                         file_names: List[str],
                         operation_names: Optional[List[str]] = None,
                         figsize: tuple = (14, 10),
                         save_path: Optional[Union[str, Path]] = None,
                         show_plot: bool = True) -> plt.Figure:
        """
        Compare benchmark results from multiple files.
        
        Args:
            file_names: Names of result files to compare
            operation_names: Names of operations to compare, or None for all common ones
            figsize: Figure size (width, height) in inches
            save_path: Path to save the plot, or None to not save
            show_plot: Whether to display the plot
            
        Returns:
            The matplotlib Figure object
        """
        # Load results if needed
        for file_name in file_names:
            if file_name not in self.loaded_results:
                self.load_results(file_name)

        # Get results for each file
        results_list = [self.loaded_results.get(name) for name in file_names]
        if not all(results_list):
            missing = [name for i, name in enumerate(file_names) if not results_list[i]]
            raise ValueError(f"Results not loaded for: {', '.join(missing)}")

        # Get common operations if not specified
        if operation_names is None:
            operation_sets = [set(res.get("summary", {}).keys()) for res in results_list]
            operation_names = list(set.intersection(*operation_sets))

        if not operation_names:
            raise ValueError("No common operations found between the benchmark files")

        # Collect data for comparison
        data = {}
        for op_name in operation_names:
            data[op_name] = {
                "means": [],
                "stds": [],
                "labels": []
            }

            for i, (file_name, results) in enumerate(zip(file_names, results_list)):
                summary = results.get("summary", {})
                if op_name in summary and "error" not in summary[op_name]:
                    data[op_name]["means"].append(summary[op_name]["mean"])
                    data[op_name]["stds"].append(summary[op_name]["std"])

                    # Create label from file name
                    label = file_name.split("_")[0] if "_" in file_name else file_name
                    data[op_name]["labels"].append(label)

        # Create plot
        n_ops = len(operation_names)
        n_cols = min(3, n_ops)
        n_rows = (n_ops + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_ops == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        # Plot each operation
        for i, op_name in enumerate(operation_names):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]

            op_data = data[op_name]
            x = np.arange(len(op_data["labels"]))
            width = 0.6

            # Plot bar chart
            bars = ax.bar(x, op_data["means"], width, yerr=op_data["stds"],
                        capsize=5, alpha=0.7)

            # Add labels
            ax.set_title(op_name)
            ax.set_xlabel("Benchmark")
            ax.set_ylabel("Duration (seconds)")
            ax.set_xticks(x)
            ax.set_xticklabels(op_data["labels"], rotation=45, ha="right")

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f"{height:.4f}s",
                       ha="center", va="bottom", rotation=0)

        # Hide unused subplots
        for i in range(n_ops, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis("off")

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        # Show if requested
        if show_plot:
            plt.show()

        return fig

    def generate_report(self,
                      result_file: Optional[str] = None,
                      output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a comprehensive HTML report from benchmark results.
        
        Args:
            result_file: Name of the result file to use, or None to use most recent
            output_path: Path to save the HTML report, or None for default location
            
        Returns:
            Path to the generated HTML report
        """
        # Load results if not already loaded
        if not self.loaded_results:
            self.load_results()

        # Use the most recent result file if not specified
        if result_file is None:
            file_timestamps = [
                (name, self._extract_timestamp(res))
                for name, res in self.loaded_results.items()
            ]
            if not file_timestamps:
                raise ValueError("No result files loaded")
            result_file = max(file_timestamps, key=lambda x: x[1])[0]

        results = self.loaded_results.get(result_file)
        if not results:
            raise ValueError(f"Results file {result_file} not loaded")

        # Determine output path
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.results_dir / f"benchmark_report_{timestamp}.html"
        else:
            output_path = Path(output_path)

        # Generate summary plots
        plot_path = output_path.with_suffix(".png")
        self.plot_operation_durations(result_file=result_file,
                                    save_path=plot_path,
                                    show_plot=False)

        # Create HTML report
        benchmark_name = results.get("benchmark_name", "Benchmark")
        timestamp = results.get("timestamp", "Unknown")
        summary = results.get("summary", {})

        # Build HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{benchmark_name} Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .summary-plot {{ width: 100%; max-width: 1000px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>{benchmark_name} Benchmark Report</h1>
            <p><strong>Generated:</strong> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Benchmark Timestamp:</strong> {timestamp}</p>
            
            <h2>Summary Results</h2>
            <img src="{plot_path.name}" class="summary-plot" alt="Benchmark Summary Plot">
            
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Operation</th>
                    <th>Mean (s)</th>
                    <th>Median (s)</th>
                    <th>Min (s)</th>
                    <th>Max (s)</th>
                    <th>Std Dev (s)</th>
                    <th>Success Rate</th>
                </tr>
        """

        # Add rows for each operation
        for op_name, stats in summary.items():
            if "error" in stats:
                html += f"""
                <tr>
                    <td>{op_name}</td>
                    <td colspan="6">Error: {stats["error"]}</td>
                </tr>
                """
            else:
                html += f"""
                <tr>
                    <td>{op_name}</td>
                    <td>{stats["mean"]:.6f}</td>
                    <td>{stats["median"]:.6f}</td>
                    <td>{stats["min"]:.6f}</td>
                    <td>{stats["max"]:.6f}</td>
                    <td>{stats["std"]:.6f}</td>
                    <td>{stats["success_rate"] * 100:.2f}% ({int(stats["success_rate"] * stats["num_runs"])}/{stats["num_runs"]})</td>
                </tr>
                """

        html += """
            </table>
            
            <h2>Benchmark Information</h2>
        """

        # Add any additional benchmark information
        for key, value in results.items():
            if key not in ["benchmark_name", "timestamp", "results", "summary"] and not isinstance(value, dict):
                html += f"""
                <p><strong>{key}:</strong> {value}</p>
                """

        # Add benchmark configuration if available
        if "benchmark_info" in results:
            html += """
            <h3>Configuration</h3>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
            """

            for key, value in results["benchmark_info"].items():
                html += f"""
                <tr>
                    <td>{key}</td>
                    <td>{value}</td>
                </tr>
                """

            html += """
            </table>
            """

        html += """
        </body>
        </html>
        """

        # Write the HTML to file
        with open(output_path, "w") as f:
            f.write(html)

        return str(output_path)

    def _extract_timestamp(self, results: Dict) -> datetime.datetime:
        """Extract timestamp from results dictionary."""
        timestamp_str = results.get("timestamp", "")
        try:
            return datetime.datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            # Return epoch time if timestamp can't be parsed
            return datetime.datetime.fromtimestamp(0)


def compare_multiple_benchmarks(result_dirs: List[Union[str, Path]],
                              output_dir: Optional[Union[str, Path]] = None,
                              operation_filter: Optional[List[str]] = None) -> str:
    """
    Compare benchmark results from multiple directories.
    
    Args:
        result_dirs: List of directories containing benchmark results
        output_dir: Directory to save comparison results, or None to use first dir
        operation_filter: List of operations to compare, or None for all common ones
        
    Returns:
        Path to the generated comparison report
    """
    if not result_dirs:
        raise ValueError("At least one result directory is required")

    # Determine output directory
    if output_dir is None:
        output_dir = result_dirs[0]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get latest result from each directory
    latest_results = []
    dir_labels = []

    for dir_path in result_dirs:
        dir_path = Path(dir_path)
        visualizer = BenchmarkVisualizer(dir_path)
        visualizer.load_results()

        # Get latest result file
        if not visualizer.loaded_results:
            print(f"Warning: No result files found in {dir_path}")
            continue

        file_timestamps = [
            (name, visualizer._extract_timestamp(res))
            for name, res in visualizer.loaded_results.items()
        ]
        latest_file, _ = max(file_timestamps, key=lambda x: x[1])

        latest_results.append((dir_path, latest_file, visualizer.loaded_results[latest_file]))
        dir_labels.append(dir_path.name)

    if not latest_results:
        raise ValueError("No benchmark results found in any directory")

    # Create combined data for plotting
    combined_data = {}

    # Find common operations
    if operation_filter is None:
        op_sets = [set(res.get("summary", {}).keys()) for _, _, res in latest_results]
        common_ops = list(set.intersection(*op_sets))
    else:
        common_ops = operation_filter

    if not common_ops:
        raise ValueError("No common operations found between benchmark results")

    # Extract data for each operation
    for op_name in common_ops:
        combined_data[op_name] = {
            "means": [],
            "stds": [],
            "labels": []
        }

        for dir_path, _, results in latest_results:
            summary = results.get("summary", {})
            if op_name in summary and "error" not in summary[op_name]:
                combined_data[op_name]["means"].append(summary[op_name]["mean"])
                combined_data[op_name]["stds"].append(summary[op_name]["std"])
                combined_data[op_name]["labels"].append(dir_path.name)

    # Generate comparison plots
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = output_dir / f"benchmark_comparison_{timestamp}.html"
    plot_file = output_dir / f"benchmark_comparison_{timestamp}.png"

    # Create comparison plot
    n_ops = len(common_ops)
    n_cols = min(3, n_ops)
    n_rows = (n_ops + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10))
    if n_ops == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot each operation
    for i, op_name in enumerate(common_ops):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        op_data = combined_data[op_name]
        x = np.arange(len(op_data["labels"]))
        width = 0.6

        # Plot bar chart
        bars = ax.bar(x, op_data["means"], width, yerr=op_data["stds"],
                    capsize=5, alpha=0.7)

        # Add labels
        ax.set_title(op_name)
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Duration (seconds)")
        ax.set_xticks(x)
        ax.set_xticklabels(op_data["labels"], rotation=45, ha="right")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f"{height:.4f}s",
                   ha="center", va="bottom", rotation=0)

    # Hide unused subplots
    for i in range(n_ops, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")

    # Create HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>FedZK Benchmark Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .comparison-plot {{ width: 100%; max-width: 1000px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>FedZK Benchmark Comparison</h1>
        <p><strong>Generated:</strong> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Configurations Compared:</strong> {', '.join(dir_labels)}</p>
        
        <h2>Comparison Results</h2>
        <img src="{plot_file.name}" class="comparison-plot" alt="Benchmark Comparison Plot">
        
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>Operation</th>
    """

    # Add column headers for each configuration
    for dir_label in dir_labels:
        html += f"""
                <th>{dir_label} Mean (s)</th>
                <th>{dir_label} Std Dev (s)</th>
        """

    html += """
            </tr>
    """

    # Add rows for each operation
    for op_name in common_ops:
        html += f"""
            <tr>
                <td>{op_name}</td>
        """

        for i, dir_label in enumerate(dir_labels):
            op_data = combined_data[op_name]
            if i < len(op_data["means"]):
                html += f"""
                <td>{op_data["means"][i]:.6f}</td>
                <td>{op_data["stds"][i]:.6f}</td>
                """
            else:
                html += """
                <td>N/A</td>
                <td>N/A</td>
                """

        html += """
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """

    # Write the HTML to file
    with open(comparison_file, "w") as f:
        f.write(html)

    return str(comparison_file)
