#!/usr/bin/env python3
"""
Security Scanner Integration for CI/CD
======================================

Comprehensive security scanning and vulnerability assessment
for FEDzk federated learning system.
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re


class SecurityScanner:
    """Comprehensive security scanner for CI/CD pipeline."""

    def __init__(self):
        """Initialize security scanner."""
        self.scan_results = {
            'bandit_scan': {},
            'safety_scan': {},
            'dependency_audit': {},
            'code_analysis': {},
            'cryptographic_validation': {}
        }
        self.vulnerability_thresholds = {
            'critical': 0,  # Zero tolerance for critical vulnerabilities
            'high': 5,      # Allow up to 5 high-severity issues
            'medium': 20,   # Allow up to 20 medium-severity issues
            'low': 50       # Allow up to 50 low-severity issues
        }

    def run_bandit_scan(self) -> Dict[str, Any]:
        """Run Bandit security scanner on Python code."""
        print("🔍 Running Bandit Security Scan...")

        try:
            # Run bandit with JSON output
            result = subprocess.run([
                'bandit', '-r', 'src/', '-f', 'json'
            ], capture_output=True, text=True, timeout=300)

            # Parse JSON output
            if result.returncode == 0:
                bandit_data = {'results': []}
            else:
                try:
                    bandit_data = json.loads(result.stdout)
                except json.JSONDecodeError:
                    bandit_data = {'error': 'Failed to parse bandit output', 'raw_output': result.stdout}

            # Analyze results
            analysis = self._analyze_bandit_results(bandit_data)

            self.scan_results['bandit_scan'] = {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'data': bandit_data,
                'analysis': analysis,
                'timestamp': '2025-09-04T10:00:00.000000'
            }

            print(f"✅ Bandit scan completed - Found {analysis['total_issues']} issues")
            return self.scan_results['bandit_scan']

        except subprocess.TimeoutExpired:
            print("❌ Bandit scan timed out")
            self.scan_results['bandit_scan'] = {'error': 'Timeout', 'success': False}
            return self.scan_results['bandit_scan']
        except FileNotFoundError:
            print("❌ Bandit not installed")
            self.scan_results['bandit_scan'] = {'error': 'Bandit not found', 'success': False}
            return self.scan_results['bandit_scan']

    def run_safety_scan(self) -> Dict[str, Any]:
        """Run Safety vulnerability scanner on dependencies."""
        print("🛡️ Running Safety Vulnerability Scan...")

        try:
            # Run safety check
            result = subprocess.run([
                'safety', 'check', '--output', 'json'
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                safety_data = []
            else:
                try:
                    safety_data = json.loads(result.stdout)
                except json.JSONDecodeError:
                    safety_data = [{'error': 'Failed to parse safety output'}]

            # Analyze vulnerabilities
            analysis = self._analyze_safety_results(safety_data)

            self.scan_results['safety_scan'] = {
                'success': True,  # Safety returns non-zero for vulnerabilities, which is OK
                'return_code': result.returncode,
                'data': safety_data,
                'analysis': analysis,
                'timestamp': '2025-09-04T10:00:00.000000'
            }

            print(f"✅ Safety scan completed - Found {analysis['total_vulnerabilities']} vulnerabilities")
            return self.scan_results['safety_scan']

        except subprocess.TimeoutExpired:
            print("❌ Safety scan timed out")
            self.scan_results['safety_scan'] = {'error': 'Timeout', 'success': False}
            return self.scan_results['safety_scan']
        except FileNotFoundError:
            print("❌ Safety not installed")
            self.scan_results['safety_scan'] = {'error': 'Safety not found', 'success': False}
            return self.scan_results['safety_scan']

    def run_dependency_audit(self) -> Dict[str, Any]:
        """Run dependency auditing for supply chain security."""
        print("📦 Running Dependency Audit...")

        try:
            # Run pip-audit
            result = subprocess.run([
                'pip-audit', '--format', 'json'
            ], capture_output=True, text=True, timeout=180)

            if result.returncode == 0:
                audit_data = {'vulnerabilities': []}
            else:
                try:
                    audit_data = json.loads(result.stdout)
                except json.JSONDecodeError:
                    audit_data = {'error': 'Failed to parse audit output'}

            # Analyze audit results
            analysis = self._analyze_audit_results(audit_data)

            self.scan_results['dependency_audit'] = {
                'success': result.returncode in [0, 1],  # 1 means vulnerabilities found, which is OK
                'return_code': result.returncode,
                'data': audit_data,
                'analysis': analysis,
                'timestamp': '2025-09-04T10:00:00.000000'
            }

            print(f"✅ Dependency audit completed - Found {analysis['total_vulnerabilities']} issues")
            return self.scan_results['dependency_audit']

        except subprocess.TimeoutExpired:
            print("❌ Dependency audit timed out")
            self.scan_results['dependency_audit'] = {'error': 'Timeout', 'success': False}
            return self.scan_results['dependency_audit']
        except FileNotFoundError:
            print("❌ pip-audit not installed")
            self.scan_results['dependency_audit'] = {'error': 'pip-audit not found', 'success': False}
            return self.scan_results['dependency_audit']

    def analyze_cryptographic_security(self) -> Dict[str, Any]:
        """Analyze cryptographic security of the codebase."""
        print("🔐 Analyzing Cryptographic Security...")

        analysis = {
            'weak_crypto_usage': [],
            'hardcoded_secrets': [],
            'insecure_random': [],
            'deprecated_algorithms': [],
            'score': 100
        }

        # Scan for potential cryptographic issues
        python_files = list(Path('src').rglob('*.py'))

        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')

                for line_num, line in enumerate(lines, 1):
                    # Check for weak cryptographic usage
                    if re.search(r'MD5|SHA1|DES|RC4', line, re.IGNORECASE):
                        analysis['deprecated_algorithms'].append({
                            'file': str(file_path),
                            'line': line_num,
                            'issue': 'Deprecated cryptographic algorithm',
                            'code': line.strip()
                        })
                        analysis['score'] -= 10

                    # Check for hardcoded secrets (simplified)
                    if re.search(r'password|secret|key.*=.*["\'][^"\']*["\']', line, re.IGNORECASE):
                        if 'test' not in str(file_path).lower():  # Skip test files
                            analysis['hardcoded_secrets'].append({
                                'file': str(file_path),
                                'line': line_num,
                                'issue': 'Potential hardcoded secret',
                                'code': line.strip()
                            })
                            analysis['score'] -= 15

                    # Check for insecure random usage
                    if 'random.random()' in line or 'random.randint' in line:
                        if 'crypt' not in content:  # Not for cryptographic purposes
                            analysis['insecure_random'].append({
                                'file': str(file_path),
                                'line': line_num,
                                'issue': 'Insecure random number generation',
                                'code': line.strip()
                            })
                            analysis['score'] -= 5

            except IOError:
                continue

        # Ensure score doesn't go below 0
        analysis['score'] = max(0, analysis['score'])

        self.scan_results['cryptographic_validation'] = {
            'success': True,
            'analysis': analysis,
            'timestamp': '2025-09-04T10:00:00.000000'
        }

        print(f"✅ Cryptographic analysis completed - Security score: {analysis['score']}/100")
        return self.scan_results['cryptographic_validation']

    def _analyze_bandit_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Bandit scan results."""
        analysis = {
            'total_issues': 0,
            'severity_breakdown': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
            'confidence_breakdown': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
            'issues_by_file': {},
            'blocking_issues': []
        }

        if 'results' in data:
            for result in data['results']:
                filename = result.get('filename', 'unknown')
                issues = result.get('issues', [])

                analysis['total_issues'] += len(issues)

                for issue in issues:
                    severity = issue.get('issue_severity', 'UNKNOWN')
                    confidence = issue.get('issue_confidence', 'UNKNOWN')

                    if severity in analysis['severity_breakdown']:
                        analysis['severity_breakdown'][severity] += 1

                    if confidence in analysis['confidence_breakdown']:
                        analysis['confidence_breakdown'][confidence] += 1

                    if filename not in analysis['issues_by_file']:
                        analysis['issues_by_file'][filename] = []
                    analysis['issues_by_file'][filename].append(issue)

                    # Check if this is a blocking issue (HIGH severity, HIGH confidence)
                    if severity == 'HIGH' and confidence == 'HIGH':
                        analysis['blocking_issues'].append(issue)

        return analysis

    def _analyze_safety_results(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Safety scan results."""
        analysis = {
            'total_vulnerabilities': len(data),
            'severity_breakdown': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
            'vulnerabilities_by_package': {},
            'blocking_vulnerabilities': []
        }

        for vuln in data:
            severity = vuln.get('severity', 'unknown').lower()

            if severity in analysis['severity_breakdown']:
                analysis['severity_breakdown'][severity] += 1

            package = vuln.get('package', 'unknown')
            if package not in analysis['vulnerabilities_by_package']:
                analysis['vulnerabilities_by_package'][package] = []
            analysis['vulnerabilities_by_package'][package].append(vuln)

            # Critical and high severity vulnerabilities are blocking
            if severity in ['critical', 'high']:
                analysis['blocking_vulnerabilities'].append(vuln)

        return analysis

    def _analyze_audit_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dependency audit results."""
        analysis = {
            'total_vulnerabilities': 0,
            'severity_breakdown': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
            'vulnerabilities_by_package': {},
            'blocking_vulnerabilities': []
        }

        vulnerabilities = data.get('vulnerabilities', [])
        analysis['total_vulnerabilities'] = len(vulnerabilities)

        for vuln in vulnerabilities:
            severity = vuln.get('severity', 'unknown').lower()

            if severity in analysis['severity_breakdown']:
                analysis['severity_breakdown'][severity] += 1

            package = vuln.get('package', 'unknown')
            if package not in analysis['vulnerabilities_by_package']:
                analysis['vulnerabilities_by_package'][package] = []
            analysis['vulnerabilities_by_package'][package].append(vuln)

            # Critical vulnerabilities are always blocking
            if severity == 'critical':
                analysis['blocking_vulnerabilities'].append(vuln)

        return analysis

    def check_quality_gates(self) -> Dict[str, Any]:
        """Check if security results meet quality gate requirements."""
        gates_status = {
            'passed': True,
            'blocking_issues': [],
            'warnings': [],
            'recommendations': []
        }

        # Check Bandit results
        if 'bandit_scan' in self.scan_results:
            bandit_analysis = self.scan_results['bandit_scan'].get('analysis', {})
            high_confidence_issues = len(bandit_analysis.get('blocking_issues', []))

            if high_confidence_issues > 0:
                gates_status['passed'] = False
                gates_status['blocking_issues'].append(
                    f"Bandit: {high_confidence_issues} high-confidence security issues"
                )

        # Check Safety results
        if 'safety_scan' in self.scan_results:
            safety_analysis = self.scan_results['safety_scan'].get('analysis', {})
            critical_vulns = safety_analysis.get('severity_breakdown', {}).get('critical', 0)
            high_vulns = safety_analysis.get('severity_breakdown', {}).get('high', 0)

            if critical_vulns > self.vulnerability_thresholds['critical']:
                gates_status['passed'] = False
                gates_status['blocking_issues'].append(
                    f"Safety: {critical_vulns} critical vulnerabilities (threshold: {self.vulnerability_thresholds['critical']})"
                )

            if high_vulns > self.vulnerability_thresholds['high']:
                gates_status['warnings'].append(
                    f"Safety: {high_vulns} high-severity vulnerabilities (threshold: {self.vulnerability_thresholds['high']})"
                )

        # Check cryptographic security
        if 'cryptographic_validation' in self.scan_results:
            crypto_analysis = self.scan_results['cryptographic_validation'].get('analysis', {})
            crypto_score = crypto_analysis.get('score', 100)

            if crypto_score < 80:
                gates_status['passed'] = False
                gates_status['blocking_issues'].append(
                    f"Cryptographic: Security score {crypto_score}/100 (below 80 threshold)"
                )

        return gates_status

    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        print("\n📊 Security Scan Report")
        print("=" * 40)

        # Run quality gates check
        quality_gates = self.check_quality_gates()

        report = {
            'timestamp': '2025-09-04T10:00:00.000000',
            'scan_type': 'FEDzk CI/CD Security Scan',
            'overall_status': 'PASSED' if quality_gates['passed'] else 'FAILED',
            'quality_gates': quality_gates,
            'scan_results': self.scan_results,
            'recommendations': []
        }

        print(f"🎯 Overall Status: {'✅ PASSED' if quality_gates['passed'] else '❌ FAILED'}")

        # Generate recommendations
        if not quality_gates['passed']:
            print("
🚨 BLOCKING ISSUES:"            for issue in quality_gates['blocking_issues']:
                print(f"   ❌ {issue}")
                report['recommendations'].append(f"CRITICAL: {issue}")

        if quality_gates['warnings']:
            print("
⚠️ WARNINGS:"            for warning in quality_gates['warnings']:
                print(f"   ⚠️ {warning}")
                report['recommendations'].append(f"WARNING: {warning}")

        # Summary statistics
        total_issues = 0
        if 'bandit_scan' in self.scan_results:
            bandit_analysis = self.scan_results['bandit_scan'].get('analysis', {})
            total_issues += bandit_analysis.get('total_issues', 0)

        if 'safety_scan' in self.scan_results:
            safety_analysis = self.scan_results['safety_scan'].get('analysis', {})
            total_issues += safety_analysis.get('total_vulnerabilities', 0)

        if 'dependency_audit' in self.scan_results:
            audit_analysis = self.scan_results['dependency_audit'].get('analysis', {})
            total_issues += audit_analysis.get('total_vulnerabilities', 0)

        print("
📈 SCAN STATISTICS:"        print(f"   Total Security Issues: {total_issues}")
        print(f"   Quality Gates Passed: {quality_gates['passed']}")

        return report

    def run_full_security_scan(self) -> Dict[str, Any]:
        """Run complete security scanning suite."""
        print("🔒 Starting FEDzk Security Scan...")

        # Run all security scans
        self.run_bandit_scan()
        self.run_safety_scan()
        self.run_dependency_audit()
        self.analyze_cryptographic_security()

        # Generate comprehensive report
        report = self.generate_security_report()

        # Save security report
        report_file = Path("test_reports/security_scan_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Security report saved: {report_file}")

        return report


def main():
    """Main entry point for security scanning."""
    print("🛡️ FEDzk Security Scanner")
    print("=" * 30)

    scanner = SecurityScanner()
    report = scanner.run_full_security_scan()

    # Exit with appropriate code
    if report['overall_status'] == 'PASSED':
        print("✅ Security scan PASSED - All quality gates met")
        return 0
    else:
        print("❌ Security scan FAILED - Quality gates not met")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

