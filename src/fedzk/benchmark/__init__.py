# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
FEDzk Benchmarking package.

This package provides comprehensive benchmarking for FEDzk components.
"""

from fedzk.benchmark.utils import BenchmarkResults, benchmark, generate_random_gradients

__all__ = [
    "BenchmarkResults",
    "benchmark",
    "generate_random_gradients"
]
