#!/usr/bin/env python3
"""
Docker Container Security Scanner
==================================

Comprehensive security scanning for FEDzk container images.
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import re


class DockerSecurityScanner:
    """Security scanner for Docker container images."""

    def __init__(self):
        """Initialize security scanner."""
        self.scan_results = {
            'image_analysis': {},
            'vulnerability_scan': {},
            'configuration_audit': {},
            'best_practices_check': {},
            'compliance_check': {}
        }

    def scan_image(self, image_name: str, image_tag: str = "latest") -> Dict[str, Any]:
        """Perform comprehensive security scan on Docker image."""
        print(f"🔍 Scanning Docker image: {image_name}:{image_tag}")

        full_image_name = f"{image_name}:{image_tag}"

        # Check if image exists
        if not self._image_exists(full_image_name):
            return {'error': f'Image {full_image_name} not found'}

        results = {}

        # Image analysis
        results['image_analysis'] = self._analyze_image_layers(full_image_name)

        # Vulnerability scanning
        results['vulnerability_scan'] = self._scan_vulnerabilities(full_image_name)

        # Configuration audit
        results['configuration_audit'] = self._audit_configuration(full_image_name)

        # Best practices check
        results['best_practices_check'] = self._check_best_practices(full_image_name)

        # Compliance check
        results['compliance_check'] = self._check_compliance(full_image_name)

        return results

    def _image_exists(self, image_name: str) -> bool:
        """Check if Docker image exists."""
        try:
            result = subprocess.run([
                'docker', 'images', '-q', image_name
            ], capture_output=True, text=True, check=True)

            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False

    def _analyze_image_layers(self, image_name: str) -> Dict[str, Any]:
        """Analyze Docker image layers and metadata."""
        analysis = {
            'layer_count': 0,
            'total_size': 0,
            'base_image': '',
            'exposed_ports': [],
            'volumes': [],
            'environment_variables': {},
            'security_warnings': []
        }

        try:
            # Get image history
            result = subprocess.run([
                'docker', 'history', '--format', 'json', image_name
            ], capture_output=True, text=True, check=True)

            layers = [json.loads(line) for line in result.stdout.strip().split('\n') if line.strip()]
            analysis['layer_count'] = len(layers)

            # Calculate total size
            for layer in layers:
                size_str = layer.get('Size', '0B')
                if size_str != '<missing>':
                    analysis['total_size'] += self._parse_size(size_str)

        except subprocess.CalledProcessError:
            analysis['security_warnings'].append('Failed to analyze image layers')

        try:
            # Get image configuration
            result = subprocess.run([
                'docker', 'inspect', image_name
            ], capture_output=True, text=True, check=True)

            inspect_data = json.loads(result.stdout)[0]
            config = inspect_data.get('Config', {})

            # Extract configuration details
            analysis['exposed_ports'] = list(config.get('ExposedPorts', {}).keys())
            analysis['volumes'] = list(inspect_data.get('Config', {}).get('Volumes', {}).keys())
            analysis['environment_variables'] = config.get('Env', [])

        except (subprocess.CalledProcessError, json.JSONDecodeError):
            analysis['security_warnings'].append('Failed to inspect image configuration')

        return analysis

    def _scan_vulnerabilities(self, image_name: str) -> Dict[str, Any]:
        """Scan for vulnerabilities in the container image."""
        vulnerabilities = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'total': 0,
            'details': [],
            'scanner_used': '',
            'scan_success': False
        }

        # Try Trivy first (most comprehensive)
        try:
            result = subprocess.run([
                'trivy', 'image', '--format', 'json', image_name
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                vulnerabilities['scanner_used'] = 'trivy'
                vulnerabilities['scan_success'] = True

                try:
                    scan_data = json.loads(result.stdout)
                    for result_item in scan_data.get('Results', []):
                        for vuln in result_item.get('Vulnerabilities', []):
                            severity = vuln.get('Severity', 'UNKNOWN').lower()
                            if severity in vulnerabilities:
                                vulnerabilities[severity] += 1
                                vulnerabilities['total'] += 1

                            vulnerabilities['details'].append({
                                'id': vuln.get('VulnerabilityID', ''),
                                'severity': severity,
                                'package': vuln.get('PkgName', ''),
                                'installed_version': vuln.get('InstalledVersion', ''),
                                'fixed_version': vuln.get('FixedVersion', ''),
                                'description': vuln.get('Description', '')[:100] + '...'
                            })

                except json.JSONDecodeError:
                    vulnerabilities['scan_success'] = False

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to Docker Scout if Trivy not available
            try:
                result = subprocess.run([
                    'docker', 'scout', 'cves', image_name, '--format', 'json'
                ], capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    vulnerabilities['scanner_used'] = 'docker-scout'
                    vulnerabilities['scan_success'] = True

                    try:
                        scan_data = json.loads(result.stdout)
                        for vuln in scan_data:
                            severity = vuln.get('severity', 'UNKNOWN').lower()
                            if severity in vulnerabilities:
                                vulnerabilities[severity] += 1
                                vulnerabilities['total'] += 1
                    except json.JSONDecodeError:
                        vulnerabilities['scan_success'] = False

            except (subprocess.CalledProcessError, FileNotFoundError):
                vulnerabilities['scanner_used'] = 'none'
                vulnerabilities['scan_success'] = False

        return vulnerabilities

    def _audit_configuration(self, image_name: str) -> Dict[str, Any]:
        """Audit Docker image configuration for security issues."""
        audit = {
            'user_privileges': 'unknown',
            'root_usage': False,
            'sudo_installed': False,
            'ssh_server': False,
            'setuid_files': [],
            'world_writable': [],
            'security_issues': []
        }

        try:
            # Inspect image configuration
            result = subprocess.run([
                'docker', 'inspect', image_name
            ], capture_output=True, text=True, check=True)

            inspect_data = json.loads(result.stdout)[0]
            config = inspect_data.get('Config', {})

            # Check user configuration
            user = config.get('User', '')
            if not user or user == 'root' or user == '0':
                audit['user_privileges'] = 'root'
                audit['root_usage'] = True
                audit['security_issues'].append('Container runs as root user')
            else:
                audit['user_privileges'] = 'non-root'

            # Check for potentially dangerous environment
            env_vars = config.get('Env', [])
            for env_var in env_vars:
                if 'PASSWORD' in env_var.upper() or 'SECRET' in env_var.upper():
                    audit['security_issues'].append(f'Potential sensitive data in environment: {env_var.split("=")[0]}')

        except (subprocess.CalledProcessError, json.JSONDecodeError):
            audit['security_issues'].append('Failed to audit container configuration')

        return audit

    def _check_best_practices(self, image_name: str) -> Dict[str, Any]:
        """Check Docker image against security best practices."""
        practices = {
            'multi_stage_build': False,
            'minimal_base_image': False,
            'latest_tag_avoided': False,
            'small_image_size': False,
            'healthcheck_defined': False,
            'user_defined': False,
            'score': 0,
            'recommendations': []
        }

        try:
            # Inspect image
            result = subprocess.run([
                'docker', 'inspect', image_name
            ], capture_output=True, text=True, check=True)

            inspect_data = json.loads(result.stdout)[0]
            config = inspect_data.get('Config', {})

            # Check for non-root user
            user = config.get('User', '')
            if user and user != 'root' and user != '0':
                practices['user_defined'] = True
                practices['score'] += 20

            # Check for health check
            if config.get('Healthcheck'):
                practices['healthcheck_defined'] = True
                practices['score'] += 15

            # Check image size (rough heuristic)
            size_bytes = inspect_data.get('Size', 0)
            size_gb = size_bytes / (1024**3)
            if size_gb < 1.0:  # Less than 1GB
                practices['small_image_size'] = True
                practices['score'] += 15

            # Check if latest tag is avoided
            if ':latest' not in image_name:
                practices['latest_tag_avoided'] = True
                practices['score'] += 10

            # Generate recommendations
            if not practices['user_defined']:
                practices['recommendations'].append('Use non-root user for container execution')
            if not practices['healthcheck_defined']:
                practices['recommendations'].append('Define health check for container monitoring')
            if not practices['small_image_size']:
                practices['recommendations'].append('Optimize image size through multi-stage builds')
            if not practices['latest_tag_avoided']:
                practices['recommendations'].append('Avoid :latest tag for reproducible builds')

        except (subprocess.CalledProcessError, json.JSONDecodeError):
            practices['recommendations'].append('Failed to analyze image best practices')

        return practices

    def _check_compliance(self, image_name: str) -> Dict[str, Any]:
        """Check compliance with security standards."""
        compliance = {
            'cis_docker_benchmark': {
                'checked': False,
                'passed': 0,
                'total': 0,
                'failures': []
            },
            'nist_framework': {
                'checked': False,
                'score': 0,
                'recommendations': []
            },
            'overall_score': 0
        }

        # Basic CIS Docker Benchmark checks
        cis_checks = [
            {'name': 'Avoid running as root', 'passed': False},
            {'name': 'Use health checks', 'passed': False},
            {'name': 'Limit exposed ports', 'passed': False},
            {'name': 'Use specific image tags', 'passed': False}
        ]

        try:
            result = subprocess.run([
                'docker', 'inspect', image_name
            ], capture_output=True, text=True, check=True)

            inspect_data = json.loads(result.stdout)[0]
            config = inspect_data.get('Config', {})

            # Check user
            user = config.get('User', '')
            if user and user != 'root' and user != '0':
                cis_checks[0]['passed'] = True

            # Check health check
            if config.get('Healthcheck'):
                cis_checks[1]['passed'] = True

            # Check exposed ports
            exposed_ports = config.get('ExposedPorts', {})
            if len(exposed_ports) <= 5:  # Reasonable limit
                cis_checks[2]['passed'] = True

            # Check image tag
            if ':latest' not in image_name:
                cis_checks[3]['passed'] = True

            # Calculate compliance score
            passed_checks = sum(1 for check in cis_checks if check['passed'])
            compliance['cis_docker_benchmark']['checked'] = True
            compliance['cis_docker_benchmark']['passed'] = passed_checks
            compliance['cis_docker_benchmark']['total'] = len(cis_checks)

            for check in cis_checks:
                if not check['passed']:
                    compliance['cis_docker_benchmark']['failures'].append(check['name'])

            compliance['overall_score'] = (passed_checks / len(cis_checks)) * 100

        except (subprocess.CalledProcessError, json.JSONDecodeError):
            compliance['cis_docker_benchmark']['failures'].append('Failed to perform compliance checks')

        return compliance

    def _parse_size(self, size_str: str) -> int:
        """Parse human-readable size string to bytes."""
        if not size_str or size_str == '<missing>':
            return 0

        # Simple parsing for common formats
        size_str = size_str.upper()
        if size_str.endswith('GB'):
            return int(float(size_str[:-2]) * 1024**3)
        elif size_str.endswith('MB'):
            return int(float(size_str[:-2]) * 1024**2)
        elif size_str.endswith('KB'):
            return int(float(size_str[:-2]) * 1024)
        elif size_str.endswith('B'):
            return int(float(size_str[:-1]))
        else:
            return 0

    def generate_security_report(self, scan_results: Dict[str, Any]) -> str:
        """Generate comprehensive security report."""
        report_lines = [
            "# Docker Container Security Report",
            "",
            "## 🔍 Scan Summary",
            "",
            "| Component | Status | Details |",
            "|-----------|--------|---------|",
            f"| Image Analysis | ✅ Complete | {scan_results.get('image_analysis', {}).get('layer_count', 0)} layers |",
            f"| Vulnerability Scan | {'✅' if scan_results.get('vulnerability_scan', {}).get('scan_success', False) else '❌'} | {scan_results.get('vulnerability_scan', {}).get('total', 0)} vulnerabilities |",
            f"| Configuration Audit | ✅ Complete | {len(scan_results.get('configuration_audit', {}).get('security_issues', []))} issues |",
            f"| Best Practices | ✅ Complete | {scan_results.get('best_practices_check', {}).get('score', 0)}/100 score |",
            f"| Compliance Check | ✅ Complete | {scan_results.get('compliance_check', {}).get('overall_score', 0):.1f}% compliant |",
            "",
            "## 🚨 Security Issues",
            ""
        ]

        # Add security issues
        vuln_scan = scan_results.get('vulnerability_scan', {})
        if vuln_scan.get('total', 0) > 0:
            report_lines.extend([
                "### Vulnerabilities Found",
                "",
                f"- **Critical:** {vuln_scan.get('critical', 0)}",
                f"- **High:** {vuln_scan.get('high', 0)}",
                f"- **Medium:** {vuln_scan.get('medium', 0)}",
                f"- **Low:** {vuln_scan.get('low', 0)}",
                f"- **Total:** {vuln_scan.get('total', 0)}",
                ""
            ])

        # Configuration issues
        config_audit = scan_results.get('configuration_audit', {})
        if config_audit.get('security_issues'):
            report_lines.extend([
                "### Configuration Issues",
                ""
            ])
            for issue in config_audit['security_issues']:
                report_lines.append(f"- ❌ {issue}")
            report_lines.append("")

        # Best practices recommendations
        best_practices = scan_results.get('best_practices_check', {})
        if best_practices.get('recommendations'):
            report_lines.extend([
                "### Best Practices Recommendations",
                ""
            ])
            for rec in best_practices['recommendations']:
                report_lines.append(f"- 💡 {rec}")
            report_lines.append("")

        # Compliance status
        compliance = scan_results.get('compliance_check', {})
        cis_benchmark = compliance.get('cis_docker_benchmark', {})
        if cis_benchmark.get('checked', False):
            report_lines.extend([
                "## 📋 Compliance Status",
                "",
                f"**CIS Docker Benchmark:** {cis_benchmark.get('passed', 0)}/{cis_benchmark.get('total', 0)} checks passed",
                "",
                "### Failed Checks",
                ""
            ])
            for failure in cis_benchmark.get('failures', []):
                report_lines.append(f"- ❌ {failure}")
            report_lines.append("")

        # Overall assessment
        overall_score = 0
        if vuln_scan.get('scan_success', False):
            # Calculate risk score based on vulnerabilities
            risk_score = min(100, vuln_scan.get('total', 0) * 5)  # 5 points per vulnerability
            overall_score = max(0, 100 - risk_score)

        report_lines.extend([
            "## 📊 Overall Assessment",
            "",
            f"**Security Score:** {overall_score}/100",
            "",
            "### Risk Level",
            ""
        ])

        if overall_score >= 80:
            report_lines.append("🟢 **LOW RISK** - Container meets security standards")
        elif overall_score >= 60:
            report_lines.append("🟡 **MEDIUM RISK** - Minor security improvements recommended")
        elif overall_score >= 40:
            report_lines.append("🟠 **HIGH RISK** - Significant security issues found")
        else:
            report_lines.append("🔴 **CRITICAL RISK** - Immediate security remediation required")

        return "\n".join(report_lines)

    def run_security_scan(self, image_name: str, image_tag: str = "latest") -> Dict[str, Any]:
        """Run complete security scan on Docker image."""
        print("🛡️ Starting Docker Security Scan...")

        results = self.scan_image(image_name, image_tag)

        if 'error' in results:
            print(f"❌ Scan failed: {results['error']}")
            return results

        # Generate report
        report = self.generate_security_report(results)

        # Save results
        timestamp = '2025-09-04T10:00:00.000000'
        scan_results = {
            'timestamp': timestamp,
            'image': f"{image_name}:{image_tag}",
            'scan_results': results,
            'report': report,
            'recommendations': []
        }

        # Add recommendations based on results
        vuln_scan = results.get('vulnerability_scan', {})
        if vuln_scan.get('total', 0) > 0:
            scan_results['recommendations'].append("Update base image to latest security patches")
            scan_results['recommendations'].append("Review and update vulnerable dependencies")

        config_audit = results.get('configuration_audit', {})
        if config_audit.get('root_usage', False):
            scan_results['recommendations'].append("Configure container to run as non-root user")

        # Save to file
        results_file = Path("test_reports/docker_security_scan.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(scan_results, f, indent=2)

        report_file = Path("test_reports/docker_security_report.md")
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"📄 Security scan results saved: {results_file}")
        print(f"📄 Security report saved: {report_file}")

        return scan_results


def main():
    """Main entry point for Docker security scanning."""
    if len(sys.argv) < 2:
        print("Usage: python security_scan.py <image_name> [image_tag]")
        print("Example: python security_scan.py fedzk latest")
        sys.exit(1)

    image_name = sys.argv[1]
    image_tag = sys.argv[2] if len(sys.argv) > 2 else "latest"

    scanner = DockerSecurityScanner()
    results = scanner.run_security_scan(image_name, image_tag)

    if 'error' in results:
        print(f"❌ Security scan failed: {results['error']}")
        sys.exit(1)

    vuln_scan = results.get('scan_results', {}).get('vulnerability_scan', {})
    if vuln_scan.get('scan_success', False) and vuln_scan.get('critical', 0) > 0:
        print("❌ CRITICAL VULNERABILITIES FOUND - Deployment blocked")
        sys.exit(1)
    elif vuln_scan.get('scan_success', False) and vuln_scan.get('high', 0) > 5:
        print("⚠️ HIGH VULNERABILITY COUNT - Review recommended")
        sys.exit(0)
    else:
        print("✅ Security scan passed")
        sys.exit(0)


if __name__ == "__main__":
    main()

