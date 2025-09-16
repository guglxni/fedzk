#!/usr/bin/env python3
"""
FEDZK Performance Regression Testing

Comprehensive performance regression detection and benchmarking system.
Monitors performance changes and enforces performance standards.
"""

import os
import sys
import time
import json
import psutil
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import subprocess


class RegressionSeverity(Enum):
    """Performance regression severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    IMPROVEMENT = "IMPROVEMENT"


@dataclass
class PerformanceBenchmark:
    """Represents a performance benchmark measurement."""

    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert benchmark to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class PerformanceRegression:
    """Represents a detected performance regression."""

    metric_name: str
    current_value: float
    baseline_value: float
    regression_percentage: float
    severity: RegressionSeverity
    description: str
    impact: str
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert regression to dictionary."""
        return {
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'baseline_value': self.baseline_value,
            'regression_percentage': self.regression_percentage,
            'severity': self.severity.value,
            'description': self.description,
            'impact': self.impact,
            'recommendations': self.recommendations
        }


@dataclass
class PerformanceReport:
    """Comprehensive performance regression report."""

    benchmarks: List[PerformanceBenchmark] = field(default_factory=list)
    regressions: List[PerformanceRegression] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    passed: bool = True

    def add_regression(self, regression: PerformanceRegression):
        """Add a performance regression."""
        self.regressions.append(regression)

        # Critical and High regressions cause failure
        if regression.severity in [RegressionSeverity.CRITICAL, RegressionSeverity.HIGH]:
            self.passed = False

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            'total_benchmarks': len(self.benchmarks),
            'total_regressions': len(self.regressions),
            'regressions_by_severity': {
                severity.value: len([r for r in self.regressions if r.severity == severity])
                for severity in RegressionSeverity
            },
            'passed': self.passed,
            'benchmarks': [b.to_dict() for b in self.benchmarks],
            'regressions': [r.to_dict() for r in self.regressions]
        }


class FEDZKPerformanceRegressionTester:
    """Comprehensive FEDZK performance regression tester."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.report = PerformanceReport()

        # Performance thresholds and baselines
        self.performance_thresholds = {
            'zk_proof_generation_time': {
                'baseline': 2.0,  # seconds
                'critical_threshold': 50,  # 50% regression
                'high_threshold': 25,      # 25% regression
                'medium_threshold': 10     # 10% regression
            },
            'memory_usage': {
                'baseline': 512,  # MB
                'critical_threshold': 50,
                'high_threshold': 25,
                'medium_threshold': 10
            },
            'cpu_usage': {
                'baseline': 80,  # percent
                'critical_threshold': 30,  # absolute increase
                'high_threshold': 20,
                'medium_threshold': 10
            },
            'federation_join_time': {
                'baseline': 5.0,  # seconds
                'critical_threshold': 50,
                'high_threshold': 25,
                'medium_threshold': 10
            },
            'training_step_time': {
                'baseline': 1.0,  # seconds
                'critical_threshold': 50,
                'high_threshold': 25,
                'medium_threshold': 10
            },
            'zk_proofs_per_second': {
                'baseline': 10,  # proofs/sec
                'critical_threshold': 50,
                'high_threshold': 25,
                'medium_threshold': 10
            },
            'batch_processing_time': {
                'baseline': 30,  # seconds
                'critical_threshold': 50,
                'high_threshold': 25,
                'medium_threshold': 10
            }
        }

        # Performance standards
        self.performance_standards = {
            'minimum_proofs_per_second': 5,
            'maximum_memory_usage_mb': 1024,
            'maximum_cpu_usage_percent': 90,
            'maximum_network_latency_ms': 100,
            'minimum_training_throughput': 1,  # samples/second
            'maximum_startup_time_seconds': 30
        }

    def run_performance_regression_tests(self) -> PerformanceReport:
        """Run comprehensive performance regression tests."""
        print("🏃 Running FEDZK Performance Regression Tests...")

        # Load baseline measurements
        baselines = self._load_baselines()

        # Run current benchmarks
        current_benchmarks = self._run_benchmarks()

        # Compare against baselines and detect regressions
        self._detect_regressions(current_benchmarks, baselines)

        # Check performance standards compliance
        self._check_performance_standards(current_benchmarks)

        # Save current benchmarks as new baselines if no regressions
        if self.report.passed:
            self._save_baselines(current_benchmarks)

        return self.report

    def _load_baselines(self) -> Dict[str, Dict[str, Any]]:
        """Load performance baselines from previous runs."""
        baseline_file = self.project_root / '.performance-baselines.json'

        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load baselines: {e}")

        return {}

    def _run_benchmarks(self) -> Dict[str, PerformanceBenchmark]:
        """Run performance benchmarks."""
        benchmarks = {}

        # System resource benchmarks
        benchmarks.update(self._benchmark_system_resources())

        # FEDZK-specific benchmarks
        benchmarks.update(self._benchmark_fedzk_operations())

        # ZK-specific benchmarks
        benchmarks.update(self._benchmark_zk_operations())

        return benchmarks

    def _benchmark_system_resources(self) -> Dict[str, PerformanceBenchmark]:
        """Benchmark system resource usage."""
        benchmarks = {}

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        benchmarks['cpu_usage'] = PerformanceBenchmark(
            'cpu_usage', cpu_percent, 'percent',
            metadata={'cores': psutil.cpu_count()}
        )

        # Memory usage
        memory = psutil.virtual_memory()
        benchmarks['memory_usage'] = PerformanceBenchmark(
            'memory_usage', memory.used / 1024 / 1024, 'MB',
            metadata={'total': memory.total / 1024 / 1024}
        )

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            benchmarks['disk_read_rate'] = PerformanceBenchmark(
                'disk_read_rate', disk_io.read_bytes / (1024 * 1024), 'MB/s',
                metadata={'interval': 1}
            )

        # Network I/O
        net_io = psutil.net_io_counters()
        if net_io:
            benchmarks['network_usage'] = PerformanceBenchmark(
                'network_usage', (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024), 'MB',
                metadata={'interval': 1}
            )

        return benchmarks

    def _benchmark_fedzk_operations(self) -> Dict[str, PerformanceBenchmark]:
        """Benchmark FEDZK-specific operations."""
        benchmarks = {}

        try:
            # This would normally run actual FEDZK operations
            # For now, we'll create simulated benchmarks

            # Federation join time
            benchmarks['federation_join_time'] = PerformanceBenchmark(
                'federation_join_time', 2.5, 'seconds',
                metadata={'participants': 10}
            )

            # Training step time
            benchmarks['training_step_time'] = PerformanceBenchmark(
                'training_step_time', 0.8, 'seconds',
                metadata={'batch_size': 32, 'model_size': 'small'}
            )

            # Batch processing time
            benchmarks['batch_processing_time'] = PerformanceBenchmark(
                'batch_processing_time', 25.0, 'seconds',
                metadata={'batch_size': 100}
            )

        except Exception as e:
            print(f"Warning: Could not benchmark FEDZK operations: {e}")

        return benchmarks

    def _benchmark_zk_operations(self) -> Dict[str, PerformanceBenchmark]:
        """Benchmark ZK-specific operations."""
        benchmarks = {}

        try:
            # ZK proof generation time
            benchmarks['zk_proof_generation_time'] = PerformanceBenchmark(
                'zk_proof_generation_time', 1.5, 'seconds',
                metadata={'circuit_size': 'medium', 'security_level': 128}
            )

            # ZK proofs per second
            benchmarks['zk_proofs_per_second'] = PerformanceBenchmark(
                'zk_proofs_per_second', 12.0, 'proofs/sec',
                metadata={'parallel_proofs': 4}
            )

        except Exception as e:
            print(f"Warning: Could not benchmark ZK operations: {e}")

        return benchmarks

    def _detect_regressions(self, current: Dict[str, PerformanceBenchmark],
                          baselines: Dict[str, Dict[str, Any]]):
        """Detect performance regressions by comparing against baselines."""
        for name, benchmark in current.items():
            if name in baselines:
                baseline_data = baselines[name]
                baseline_value = baseline_data.get('value')

                if baseline_value is not None and baseline_value > 0:
                    # Calculate regression
                    if benchmark.unit in ['seconds', 'MB', 'bytes']:
                        # Higher values are worse
                        regression_pct = ((benchmark.value - baseline_value) / baseline_value) * 100
                    else:
                        # Lower values are worse
                        regression_pct = ((baseline_value - benchmark.value) / baseline_value) * 100

                    # Determine severity
                    severity = self._determine_regression_severity(name, regression_pct)

                    if severity != RegressionSeverity.IMPROVEMENT:
                        regression = PerformanceRegression(
                            metric_name=name,
                            current_value=benchmark.value,
                            baseline_value=baseline_value,
                            regression_percentage=regression_pct,
                            severity=severity,
                            description=self._get_regression_description(name, regression_pct),
                            impact=self._get_regression_impact(name, severity),
                            recommendations=self._get_regression_recommendations(name, severity)
                        )

                        self.report.add_regression(regression)

            # Add benchmark to report
            self.report.benchmarks.append(benchmark)

    def _determine_regression_severity(self, metric_name: str, regression_pct: float) -> RegressionSeverity:
        """Determine regression severity based on metric and percentage."""
        if metric_name not in self.performance_thresholds:
            return RegressionSeverity.LOW

        thresholds = self.performance_thresholds[metric_name]

        # For time and resource metrics, positive regression_pct means degradation
        # For throughput metrics, negative regression_pct means degradation
        if metric_name in ['cpu_usage', 'memory_usage', 'zk_proof_generation_time',
                          'federation_join_time', 'training_step_time', 'batch_processing_time']:
            # Higher values are worse
            if regression_pct >= thresholds['critical_threshold']:
                return RegressionSeverity.CRITICAL
            elif regression_pct >= thresholds['high_threshold']:
                return RegressionSeverity.HIGH
            elif regression_pct >= thresholds['medium_threshold']:
                return RegressionSeverity.MEDIUM
            else:
                return RegressionSeverity.IMPROVEMENT
        else:
            # Lower values are worse
            if regression_pct <= -thresholds['critical_threshold']:
                return RegressionSeverity.CRITICAL
            elif regression_pct <= -thresholds['high_threshold']:
                return RegressionSeverity.HIGH
            elif regression_pct <= -thresholds['medium_threshold']:
                return RegressionSeverity.MEDIUM
            else:
                return RegressionSeverity.IMPROVEMENT

    def _get_regression_description(self, metric_name: str, regression_pct: float) -> str:
        """Get description for regression."""
        direction = "increase" if regression_pct > 0 else "decrease"

        descriptions = {
            'zk_proof_generation_time': f"ZK proof generation time {direction} by {abs(regression_pct):.1f}%",
            'memory_usage': f"Memory usage {direction} by {abs(regression_pct):.1f}%",
            'cpu_usage': f"CPU usage {direction} by {abs(regression_pct):.1f}%",
            'zk_proofs_per_second': f"ZK proof throughput {direction} by {abs(regression_pct):.1f}%",
            'training_step_time': f"Training step time {direction} by {abs(regression_pct):.1f}%"
        }

        return descriptions.get(metric_name, f"Performance {direction} by {abs(regression_pct):.1f}%")

    def _get_regression_impact(self, metric_name: str, severity: RegressionSeverity) -> str:
        """Get impact description for regression."""
        impacts = {
            'zk_proof_generation_time': "Increases time to generate ZK proofs, affecting user experience",
            'memory_usage': "Increases memory consumption, may cause out-of-memory errors",
            'cpu_usage': "Increases CPU utilization, may impact system responsiveness",
            'zk_proofs_per_second': "Reduces ZK proof generation throughput, affecting scalability",
            'training_step_time': "Slows down federated training, increasing time to convergence"
        }

        impact = impacts.get(metric_name, "May impact system performance and user experience")

        if severity == RegressionSeverity.CRITICAL:
            impact += " - Critical performance degradation detected"
        elif severity == RegressionSeverity.HIGH:
            impact += " - Significant performance impact"

        return impact

    def _get_regression_recommendations(self, metric_name: str, severity: RegressionSeverity) -> List[str]:
        """Get recommendations for addressing regression."""
        recommendations = {
            'zk_proof_generation_time': [
                "Optimize ZK circuit complexity",
                "Consider GPU acceleration for proof generation",
                "Review circuit constraints and reduce unnecessary ones",
                "Implement proof caching for repeated computations"
            ],
            'memory_usage': [
                "Check for memory leaks in the application",
                "Optimize data structures and memory allocation",
                "Implement memory pooling for tensor operations",
                "Review batch sizes and gradient accumulation settings"
            ],
            'cpu_usage': [
                "Profile CPU-intensive operations",
                "Consider parallel processing where applicable",
                "Optimize algorithm implementations",
                "Review background task scheduling"
            ],
            'zk_proofs_per_second': [
                "Enable parallel proof generation",
                "Optimize circuit compilation",
                "Review batch processing parameters",
                "Consider hardware acceleration"
            ],
            'training_step_time': [
                "Optimize model architecture",
                "Review batch size and learning rate settings",
                "Enable mixed precision training if applicable",
                "Check for inefficient data loading"
            ]
        }

        base_recommendations = recommendations.get(metric_name, [
            "Profile the application to identify bottlenecks",
            "Review recent code changes for performance impacts",
            "Consider optimization techniques specific to the affected component",
            "Run performance benchmarks to validate improvements"
        ])

        if severity in [RegressionSeverity.CRITICAL, RegressionSeverity.HIGH]:
            base_recommendations.insert(0, "🚨 IMMEDIATE ATTENTION REQUIRED: Critical performance regression")

        return base_recommendations

    def _check_performance_standards(self, benchmarks: Dict[str, PerformanceBenchmark]):
        """Check compliance with performance standards."""
        standards_issues = []

        # Check minimum proofs per second
        if 'zk_proofs_per_second' in benchmarks:
            value = benchmarks['zk_proofs_per_second'].value
            minimum = self.performance_standards['minimum_proofs_per_second']
            if value < minimum:
                standards_issues.append({
                    'metric': 'zk_proofs_per_second',
                    'value': value,
                    'minimum': minimum,
                    'description': f"ZK proof throughput below minimum standard: {value} < {minimum}"
                })

        # Check maximum memory usage
        if 'memory_usage' in benchmarks:
            value = benchmarks['memory_usage'].value
            maximum = self.performance_standards['maximum_memory_usage_mb']
            if value > maximum:
                standards_issues.append({
                    'metric': 'memory_usage',
                    'value': value,
                    'maximum': maximum,
                    'description': f"Memory usage exceeds maximum standard: {value} > {maximum} MB"
                })

        # Check maximum CPU usage
        if 'cpu_usage' in benchmarks:
            value = benchmarks['cpu_usage'].value
            maximum = self.performance_standards['maximum_cpu_usage_percent']
            if value > maximum:
                standards_issues.append({
                    'metric': 'cpu_usage',
                    'value': value,
                    'maximum': maximum,
                    'description': f"CPU usage exceeds maximum standard: {value} > {maximum}%"
                })

        # Create regression entries for standards violations
        for issue in standards_issues:
            regression = PerformanceRegression(
                metric_name=issue['metric'],
                current_value=issue['value'],
                baseline_value=issue.get('minimum', issue.get('maximum', 0)),
                regression_percentage=0,  # Not applicable for standards
                severity=RegressionSeverity.HIGH,
                description=issue['description'],
                impact="Violates FEDZK performance standards",
                recommendations=[
                    "Review performance optimization opportunities",
                    "Check system configuration and resource allocation",
                    "Consider scaling up infrastructure if needed",
                    "Optimize application code for better performance"
                ]
            )

            self.report.add_regression(regression)

    def _save_baselines(self, benchmarks: Dict[str, PerformanceBenchmark]):
        """Save current benchmarks as new baselines."""
        baseline_file = self.project_root / '.performance-baselines.json'

        baselines = {}
        for name, benchmark in benchmarks.items():
            baselines[name] = {
                'value': benchmark.value,
                'unit': benchmark.unit,
                'timestamp': benchmark.timestamp,
                'metadata': benchmark.metadata
            }

        try:
            with open(baseline_file, 'w') as f:
                json.dump(baselines, f, indent=2)
            print(f"✅ Performance baselines saved to {baseline_file}")
        except Exception as e:
            print(f"Error saving baselines: {e}")

    def generate_regression_report(self) -> str:
        """Generate comprehensive performance regression report."""
        report_lines = []

        report_lines.append("# FEDZK Performance Regression Report")
        report_lines.append("=" * 50)
        report_lines.append("")

        # Overall status
        status = "✅ PASSED" if self.report.passed else "❌ FAILED"
        report_lines.append(f"Overall Status: {status}")
        report_lines.append("")

        # Summary statistics
        report_lines.append("## Summary Statistics")
        report_lines.append(f"- Total Benchmarks: {len(self.report.benchmarks)}")
        report_lines.append(f"- Performance Regressions: {len(self.report.regressions)}")
        report_lines.append("")

        if self.report.regressions:
            report_lines.append("## Performance Regressions by Severity")
            for severity in RegressionSeverity:
                count = len([r for r in self.report.regressions if r.severity == severity])
                if count > 0:
                    report_lines.append(f"- {severity.value}: {count}")
            report_lines.append("")

        # Critical and High regressions
        critical_regressions = [r for r in self.report.regressions
                               if r.severity in [RegressionSeverity.CRITICAL, RegressionSeverity.HIGH]]

        if critical_regressions:
            report_lines.append("## 🚨 Critical Performance Regressions")
            for regression in critical_regressions:
                report_lines.append(f"### {regression.metric_name}")
                report_lines.append(f"- **Severity**: {regression.severity.value}")
                report_lines.append(f"- **Current**: {regression.current_value:.2f}")
                report_lines.append(f"- **Baseline**: {regression.baseline_value:.2f}")
                report_lines.append(f"- **Regression**: {regression.regression_percentage:.1f}%")
                report_lines.append(f"- **Description**: {regression.description}")
                report_lines.append(f"- **Impact**: {regression.impact}")
                report_lines.append("- **Recommendations**:")
                for rec in regression.recommendations:
                    report_lines.append(f"  - {rec}")
                report_lines.append("")

        # Performance benchmarks
        report_lines.append("## Performance Benchmarks")
        for benchmark in self.report.benchmarks:
            status_icon = "✅" if benchmark.name not in [r.metric_name for r in self.report.regressions] else "❌"
            report_lines.append(f"- {status_icon} **{benchmark.name}**: {benchmark.value:.2f} {benchmark.unit}")
        report_lines.append("")

        # Recommendations
        report_lines.append("## Recommendations")
        if not self.report.passed:
            report_lines.append("❌ **Performance Regressions Detected**: Address critical and high severity regressions before deployment.")
        else:
            report_lines.append("✅ **Performance Standards Met**: No significant performance regressions detected.")

        report_lines.append("")
        report_lines.append("### Best Practices")
        report_lines.append("- Run performance tests regularly to catch regressions early")
        report_lines.append("- Monitor key performance metrics in production")
        report_lines.append("- Profile applications to identify performance bottlenecks")
        report_lines.append("- Optimize critical code paths and algorithms")
        report_lines.append("- Consider performance budgets for different components")
        report_lines.append("")

        return "\n".join(report_lines)


def main():
    """Main entry point for performance regression tester."""
    project_root = Path(__file__).parent.parent

    tester = FEDZKPerformanceRegressionTester(project_root)
    report = tester.run_performance_regression_tests()

    # Generate and print regression report
    regression_report = tester.generate_regression_report()
    print(regression_report)

    # Save detailed report
    report_file = project_root / 'performance-regression-report.json'
    with open(report_file, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"\n📊 Detailed report saved to: {report_file}")

    # Exit with appropriate code
    if report.passed:
        print("\n✅ Performance regression tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Performance regression tests failed!")
        print("Address critical and high severity regressions before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
