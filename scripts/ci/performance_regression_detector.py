#!/usr/bin/env python3
"""
Performance Regression Detector for CI/CD
=========================================

Detects performance regressions in the FEDzk system by comparing
current performance metrics against historical baselines.
"""

import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
import psutil
import os


class PerformanceRegressionDetector:
    """Detect performance regressions in CI/CD environment."""

    def __init__(self, baseline_file: str = "performance-baseline.json",
                 current_results_file: str = "performance-results.json"):
        """Initialize detector with baseline and current results."""
        self.baseline_file = Path(baseline_file)
        self.current_results_file = Path(current_results_file)
        self.baseline_data = {}
        self.current_data = {}
        self.regression_thresholds = {
            'response_time': 1.1,  # 10% degradation allowed
            'memory_usage': 1.15,  # 15% increase allowed
            'cpu_usage': 1.2,      # 20% increase allowed
            'throughput': 0.9      # 10% decrease allowed
        }

    def load_baseline_data(self) -> bool:
        """Load historical baseline performance data."""
        if not self.baseline_file.exists():
            print(f"⚠️ Baseline file not found: {self.baseline_file}")
            return False

        try:
            with open(self.baseline_file, 'r') as f:
                self.baseline_data = json.load(f)
            print(f"✅ Loaded baseline data from {self.baseline_file}")
            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"❌ Failed to load baseline data: {e}")
            return False

    def load_current_results(self) -> bool:
        """Load current performance test results."""
        if not self.current_results_file.exists():
            print(f"⚠️ Current results file not found: {self.current_results_file}")
            return False

        try:
            with open(self.current_results_file, 'r') as f:
                self.current_data = json.load(f)
            print(f"✅ Loaded current results from {self.current_results_file}")
            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"❌ Failed to load current results: {e}")
            return False

    def detect_regressions(self) -> Dict[str, Any]:
        """Detect performance regressions by comparing current vs baseline."""
        regressions = {
            'detected': False,
            'critical_regressions': [],
            'warning_regressions': [],
            'improvements': [],
            'metrics_comparison': {}
        }

        if not self.baseline_data or not self.current_data:
            regressions['error'] = "Missing baseline or current data"
            return regressions

        # Compare key performance metrics
        metrics_to_compare = [
            'avg_response_time',
            'max_memory_usage',
            'avg_cpu_usage',
            'avg_throughput',
            'total_requests'
        ]

        for metric in metrics_to_compare:
            baseline_value = self.baseline_data.get(metric)
            current_value = self.current_data.get(metric)

            if baseline_value is None or current_value is None:
                continue

            # Calculate regression ratio
            if metric in ['response_time', 'memory_usage', 'cpu_usage']:
                # Higher values are worse for these metrics
                ratio = current_value / baseline_value
                threshold = self.regression_thresholds.get(metric, 1.1)
                is_regression = ratio > threshold
            else:
                # Higher values are better for these metrics
                ratio = current_value / baseline_value
                threshold = self.regression_thresholds.get(metric, 0.9)
                is_regression = ratio < threshold

            regressions['metrics_comparison'][metric] = {
                'baseline': baseline_value,
                'current': current_value,
                'ratio': ratio,
                'threshold': threshold,
                'is_regression': is_regression,
                'change_percent': (ratio - 1) * 100
            }

            if is_regression:
                regression_info = {
                    'metric': metric,
                    'baseline': baseline_value,
                    'current': current_value,
                    'ratio': ratio,
                    'threshold': threshold,
                    'change_percent': (ratio - 1) * 100
                }

                # Classify as critical or warning
                if metric in ['response_time'] and ratio > 1.25:
                    regressions['critical_regressions'].append(regression_info)
                else:
                    regressions['warning_regressions'].append(regression_info)

                regressions['detected'] = True
            elif ratio < 1.0:  # Improvement
                regressions['improvements'].append({
                    'metric': metric,
                    'baseline': baseline_value,
                    'current': current_value,
                    'improvement_percent': (1 - ratio) * 100
                })

        return regressions

    def generate_regression_report(self, regressions: Dict[str, Any]) -> str:
        """Generate detailed regression report."""
        report_lines = [
            "# Performance Regression Report",
            "",
            "## 📊 Regression Analysis",
            "",
            f"**Overall Status:** {'❌ REGRESSIONS DETECTED' if regressions['detected'] else '✅ NO REGRESSIONS'}",
            "",
            "## 🔍 Metrics Comparison",
            "",
            "| Metric | Baseline | Current | Ratio | Threshold | Status |",
            "|--------|----------|---------|-------|-----------|--------|"
        ]

        for metric, comparison in regressions['metrics_comparison'].items():
            status = "❌ REGRESSION" if comparison['is_regression'] else "✅ OK"
            report_lines.append(
                f"| {metric} | {comparison['baseline']:.4f} | "
                f"{comparison['current']:.4f} | {comparison['ratio']:.3f} | "
                f"{comparison['threshold']:.2f} | {status} |"
            )

        report_lines.extend([
            "",
            "## 🚨 Critical Regressions",
            ""
        ])

        if regressions['critical_regressions']:
            for regression in regressions['critical_regressions']:
                report_lines.extend([
                    f"### {regression['metric'].upper()}",
                    f"- **Baseline:** {regression['baseline']:.4f}",
                    f"- **Current:** {regression['current']:.4f}",
                    f"- **Change:** {regression['change_percent']:+.1f}%",
                    f"- **Threshold:** {(regression['threshold']-1)*100:.0f}% degradation allowed",
                    ""
                ])
        else:
            report_lines.append("✅ No critical regressions detected")
            report_lines.append("")

        report_lines.extend([
            "## ⚠️ Warning Regressions",
            ""
        ])

        if regressions['warning_regressions']:
            for regression in regressions['warning_regressions']:
                report_lines.extend([
                    f"### {regression['metric'].upper()}",
                    f"- **Baseline:** {regression['baseline']:.4f}",
                    f"- **Current:** {regression['current']:.4f}",
                    f"- **Change:** {regression['change_percent']:+.1f}%",
                    f"- **Threshold:** {(regression['threshold']-1)*100:.0f}% degradation allowed",
                    ""
                ])
        else:
            report_lines.append("✅ No warning regressions detected")
            report_lines.append("")

        report_lines.extend([
            "## 📈 Performance Improvements",
            ""
        ])

        if regressions['improvements']:
            for improvement in regressions['improvements']:
                report_lines.extend([
                    f"### {improvement['metric'].upper()}",
                    f"- **Baseline:** {improvement['baseline']:.4f}",
                    f"- **Current:** {improvement['current']:.4f}",
                    f"- **Improvement:** {improvement['improvement_percent']:.1f}%",
                    ""
                ])
        else:
            report_lines.append("ℹ️ No significant improvements detected")
            report_lines.append("")

        return "\n".join(report_lines)

    def update_baseline(self, new_baseline_data: Dict[str, Any]) -> bool:
        """Update baseline data with current results."""
        try:
            # Add timestamp and metadata
            new_baseline_data['timestamp'] = time.time()
            new_baseline_data['ci_commit'] = os.getenv('GITHUB_SHA', 'unknown')
            new_baseline_data['ci_run'] = os.getenv('GITHUB_RUN_ID', 'unknown')

            with open(self.baseline_file, 'w') as f:
                json.dump(new_baseline_data, f, indent=2)

            print(f"✅ Updated baseline data: {self.baseline_file}")
            return True
        except IOError as e:
            print(f"❌ Failed to update baseline: {e}")
            return False

    def run_regression_analysis(self) -> Dict[str, Any]:
        """Run complete regression analysis."""
        print("📈 Starting Performance Regression Analysis...")

        # Load data
        baseline_loaded = self.load_baseline_data()
        current_loaded = self.load_current_results()

        if not baseline_loaded:
            print("⚠️ No baseline data available - creating initial baseline")
            if current_loaded:
                self.update_baseline(self.current_data)
            return {'error': 'No baseline data available'}

        if not current_loaded:
            return {'error': 'No current results available'}

        # Detect regressions
        regressions = self.detect_regressions()

        # Generate report
        report = self.generate_regression_report(regressions)

        # Save detailed results
        detailed_results = {
            'timestamp': time.time(),
            'baseline_file': str(self.baseline_file),
            'current_file': str(self.current_results_file),
            'regressions': regressions,
            'report': report
        }

        results_file = Path("test_reports/performance_regression_analysis.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        print(f"📄 Detailed analysis saved: {results_file}")

        return {
            'regressions_detected': regressions['detected'],
            'critical_count': len(regressions['critical_regressions']),
            'warning_count': len(regressions['warning_regressions']),
            'improvement_count': len(regressions['improvements']),
            'report': report
        }


def main():
    """Main entry point for performance regression detection."""
    print("📈 FEDzk Performance Regression Detector")
    print("=" * 45)

    detector = PerformanceRegressionDetector()

    results = detector.run_regression_analysis()

    if 'error' in results:
        print(f"❌ Analysis failed: {results['error']}")
        return 1

    print("
📊 Analysis Summary:"    print(f"   Regressions Detected: {results['regressions_detected']}")
    print(f"   Critical Regressions: {results['critical_count']}")
    print(f"   Warning Regressions: {results['warning_count']}")
    print(f"   Performance Improvements: {results['improvement_count']}")

    if results['regressions_detected']:
        print("
❌ PERFORMANCE REGRESSIONS DETECTED"        if results['critical_count'] > 0:
            print(f"   🚨 {results['critical_count']} critical regressions require immediate attention")
        if results['warning_count'] > 0:
            print(f"   ⚠️ {results['warning_count']} warning regressions should be reviewed")
        return 1
    else:
        print("
✅ NO PERFORMANCE REGRESSIONS DETECTED"        if results['improvement_count'] > 0:
            print(f"   📈 {results['improvement_count']} performance improvements detected")
        return 0


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)

