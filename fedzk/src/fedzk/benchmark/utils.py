# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
Benchmark utilities for FedZK.

This module provides utilities for benchmarking performance of FedZK components.
"""

import datetime
import json
import time
import tracemalloc
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class BenchmarkResults:
    """Container for benchmark results."""

    def __init__(self, benchmark_name: str):
        """
        Initialize a benchmark results container.
        
        Args:
            benchmark_name: Name of the benchmark suite
        """
        self.benchmark_name = benchmark_name
        self.results = {}
        self.timestamp = datetime.datetime.now().isoformat()

    def add_result(self, operation: str, duration: float,
                 model_size: Optional[Dict[str, int]] = None,
                 success: bool = True,
                 metadata: Optional[Dict[str, Any]] = None,
                 memory_usage: Optional[Dict[str, float]] = None):
        """
        Add a benchmark result.
        
        Args:
            operation: Name of the operation benchmarked
            duration: Duration in seconds
            model_size: Size of the model (optional)
            success: Whether the operation was successful
            metadata: Additional metadata about the operation
            memory_usage: Memory usage stats if memory profiling was enabled
        """
        if operation not in self.results:
            self.results[operation] = []

        result = {
            "duration": duration,
            "success": success,
            "timestamp": datetime.datetime.now().isoformat()
        }

        if model_size is not None:
            result["model_size"] = model_size

        if metadata is not None:
            result["metadata"] = metadata

        if memory_usage is not None:
            result["memory_usage"] = memory_usage

        self.results[operation].append(result)

    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of benchmark results.
        
        Returns:
            Dictionary with summary statistics for each operation
        """
        summary = {}

        for operation, results in self.results.items():
            durations = [r["duration"] for r in results if r["success"]]

            if not durations:
                summary[operation] = {"error": "No successful runs"}
                continue

            summary[operation] = {
                "mean": np.mean(durations),
                "median": np.median(durations),
                "min": min(durations),
                "max": max(durations),
                "std": np.std(durations),
                "num_runs": len(results),
                "success_rate": sum(1 for r in results if r["success"]) / len(results)
            }

        return summary

    def print_summary(self):
        """Print a summary of benchmark results to the console."""
        summary = self.get_summary()

        print(f"\n--- {self.benchmark_name} Benchmark Summary ---")

        for operation, stats in summary.items():
            print(f"\n{operation}:")

            if "error" in stats:
                print(f"  Error: {stats['error']}")
                continue

            print(f"  Mean: {stats['mean']:.6f} seconds")
            print(f"  Median: {stats['median']:.6f} seconds")
            print(f"  Min: {stats['min']:.6f} seconds")
            print(f"  Max: {stats['max']:.6f} seconds")
            print(f"  Std Dev: {stats['std']:.6f} seconds")
            print(f"  Success Rate: {stats['success_rate'] * 100:.2f}% ({int(stats['success_rate'] * stats['num_runs'])}/{stats['num_runs']} runs)")

    def save_to_file(self, output_dir: Union[str, Path]) -> str:
        """
        Save benchmark results to a JSON file.
        
        Args:
            output_dir: Directory to save the results file
            
        Returns:
            Path to the saved results file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.benchmark_name}_{timestamp_str}.json"
        output_path = output_dir / filename

        data = {
            "benchmark_name": self.benchmark_name,
            "timestamp": self.timestamp,
            "results": self.results,
            "summary": self.get_summary()
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return str(output_path)


def benchmark(operation_name: str = None, profile_memory: bool = False):
    """
    Decorator for benchmarking functions.
    
    Args:
        operation_name: Name of the operation being benchmarked
                      (defaults to function name if not provided)
        profile_memory: Whether to profile memory usage during the operation
    
    Returns:
        Decorated function that records benchmark results
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, results: BenchmarkResults, *args, **kwargs):
            # Get operation name
            op_name = operation_name or func.__name__

            # Determine model size if gradient_dict is in kwargs
            model_size = None
            if "gradient_dict" in kwargs:
                grad_dict = kwargs["gradient_dict"]
                model_size = {
                    "num_parameters": len(grad_dict),
                    "total_elements": sum(np.prod(g.shape) for g in grad_dict.values())
                }

            # Get metadata from kwargs
            metadata = {}
            for meta_key in ["max_inputs", "max_norm", "min_active"]:
                if meta_key in kwargs:
                    metadata[meta_key] = kwargs[meta_key]

            # Initialize memory profiling if requested
            memory_usage = None
            if profile_memory:
                tracemalloc.start()
                memory_start = tracemalloc.get_traced_memory()

            # Measure time
            start_time = time.time()
            success = True
            result = None

            try:
                result = func(self, *args, **kwargs)
            except Exception as e:
                success = False
                metadata["error"] = str(e)
                raise
            finally:
                duration = time.time() - start_time

                # Collect memory usage if profiling
                if profile_memory:
                    memory_current = tracemalloc.get_traced_memory()
                    memory_usage = {
                        "peak_mb": memory_current[1] / (1024 * 1024),
                        "current_mb": memory_current[0] / (1024 * 1024),
                        "diff_mb": (memory_current[0] - memory_start[0]) / (1024 * 1024)
                    }
                    tracemalloc.stop()

                results.add_result(
                    operation=op_name,
                    duration=duration,
                    model_size=model_size,
                    success=success,
                    metadata=metadata if metadata else None,
                    memory_usage=memory_usage
                )

            return result

        return wrapper

    return decorator


def generate_random_gradients(shape_dict: Dict[str, Tuple[int, ...]],
                            scale: float = 1.0,
                            device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Generate random gradients for benchmarking.
    
    Args:
        shape_dict: Dictionary mapping parameter names to shapes
        scale: Scale factor for the gradients
        device: Device to create tensors on
        
    Returns:
        Dictionary of random gradients with the given shapes
    """
    gradients = {}

    for name, shape in shape_dict.items():
        # Create random tensor
        grad = torch.randn(shape, device=device) * scale
        gradients[name] = grad

    return gradients


class MemoryProfiler:
    """Utility for profiling memory usage during operations."""

    def __init__(self):
        """Initialize the memory profiler."""
        self.snapshots = {}
        self.active = False

    def start(self):
        """Start memory profiling."""
        if not self.active:
            tracemalloc.start()
            self.active = True

    def take_snapshot(self, label: str):
        """
        Take a memory snapshot with a label.
        
        Args:
            label: Label for the snapshot
        """
        if not self.active:
            return

        snapshot = tracemalloc.take_snapshot()
        self.snapshots[label] = snapshot

    def compare_snapshots(self, label1: str, label2: str) -> List[Dict]:
        """
        Compare two snapshots and return difference statistics.
        
        Args:
            label1: Label of first snapshot
            label2: Label of second snapshot
            
        Returns:
            List of dictionaries with memory difference statistics
        """
        if not self.active or label1 not in self.snapshots or label2 not in self.snapshots:
            return []

        snapshot1 = self.snapshots[label1]
        snapshot2 = self.snapshots[label2]

        top_stats = snapshot2.compare_to(snapshot1, "lineno")
        return [
            {
                "file": str(stat.traceback.frame.filename),
                "line": stat.traceback.frame.lineno,
                "size_diff": stat.size_diff,
                "size_diff_mb": stat.size_diff / (1024 * 1024),
                "count_diff": stat.count_diff
            }
            for stat in top_stats[:10]  # Get top 10 differences
        ]

    def get_current_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.
        
        Returns:
            Dictionary with current and peak memory usage in MB
        """
        if not self.active:
            return {"current_mb": 0.0, "peak_mb": 0.0}

        current, peak = tracemalloc.get_traced_memory()
        return {
            "current_mb": current / (1024 * 1024),
            "peak_mb": peak / (1024 * 1024)
        }

    def stop(self):
        """Stop memory profiling and clear snapshots."""
        if self.active:
            tracemalloc.stop()
            self.snapshots.clear()
            self.active = False
