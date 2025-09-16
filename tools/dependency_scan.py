#!/usr/bin/env python3
"""
FEDZK Dependency Vulnerability Scanner

Comprehensive security scanning for project dependencies.
Checks for vulnerabilities, outdated packages, and license compliance.
"""

import os
import sys
import json
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import time


class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class DependencyIssue:
    """Represents a dependency-related issue."""

    package_name: str
    version: str
    issue_type: str
    severity: VulnerabilitySeverity
    description: str
    impact: str
    remediation: str = ""
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None
    published_date: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert issue to dictionary."""
        return {
            'package_name': self.package_name,
            'version': self.version,
            'issue_type': self.issue_type,
            'severity': self.severity.value,
            'description': self.description,
            'impact': self.impact,
            'remediation': self.remediation,
            'cve_id': self.cve_id,
            'cvss_score': self.cvss_score,
            'published_date': self.published_date
        }


@dataclass
class DependencyReport:
    """Comprehensive dependency analysis report."""

    total_packages: int = 0
    vulnerable_packages: int = 0
    outdated_packages: int = 0
    license_issues: int = 0
    issues: List[DependencyIssue] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    passed: bool = True

    def add_issue(self, issue: DependencyIssue):
        """Add a dependency issue."""
        self.issues.append(issue)
        severity = issue.severity.value
        self.summary[severity] = self.summary.get(severity, 0) + 1

        # Critical and High issues cause failure
        if issue.severity in [VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH]:
            self.passed = False

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            'total_packages': self.total_packages,
            'vulnerable_packages': self.vulnerable_packages,
            'outdated_packages': self.outdated_packages,
            'license_issues': self.license_issues,
            'total_issues': len(self.issues),
            'issues_by_severity': self.summary,
            'passed': self.passed,
            'issues': [i.to_dict() for i in self.issues]
        }


class FEDZKDependencyScanner:
    """Comprehensive FEDZK dependency scanner."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.report = DependencyReport()

        # Approved licenses for FEDZK
        self.approved_licenses = {
            'MIT', 'Apache-2.0', 'BSD-3-Clause', 'BSD-2-Clause',
            'ISC', 'Python-2.0', 'PostgreSQL'
        }

        # Forbidden licenses
        self.forbidden_licenses = {
            'GPL-3.0', 'GPL-2.0', 'LGPL-3.0', 'LGPL-2.1', 'AGPL-3.0'
        }

    def scan_dependencies(self) -> DependencyReport:
        """Perform comprehensive dependency scanning."""
        print("🔍 Scanning FEDZK dependencies...")

        # Scan Python dependencies
        self._scan_python_dependencies()

        # Scan for vulnerabilities
        self._scan_vulnerabilities()

        # Check license compliance
        self._check_license_compliance()

        # Check for outdated packages
        self._check_outdated_packages()

        return self.report

    def _scan_python_dependencies(self):
        """Scan Python dependencies from requirements and setup files."""
        try:
            # Read pyproject.toml
            pyproject_file = self.project_root / 'pyproject.toml'
            if pyproject_file.exists():
                self._parse_pyproject_toml(pyproject_file)

            # Read requirements.txt if it exists
            requirements_file = self.project_root / 'requirements.txt'
            if requirements_file.exists():
                self._parse_requirements_txt(requirements_file)

            # Get installed packages
            self._get_installed_packages()

        except Exception as e:
            print(f"Warning: Could not scan Python dependencies: {e}")

    def _parse_pyproject_toml(self, file_path: Path):
        """Parse dependencies from pyproject.toml."""
        try:
            import tomli
        except ImportError:
            print("Warning: tomli not installed, skipping pyproject.toml parsing")
            return

        try:
            with open(file_path, 'rb') as f:
                data = tomli.load(f)

            # Extract dependencies
            project_data = data.get('project', {})
            dependencies = project_data.get('dependencies', [])

            for dep in dependencies:
                # Parse package name and version
                package_info = self._parse_package_spec(dep)
                if package_info:
                    self.report.total_packages += 1

        except Exception as e:
            print(f"Warning: Could not parse pyproject.toml: {e}")

    def _parse_requirements_txt(self, file_path: Path):
        """Parse dependencies from requirements.txt."""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        package_info = self._parse_package_spec(line)
                        if package_info:
                            self.report.total_packages += 1

        except Exception as e:
            print(f"Warning: Could not parse requirements.txt: {e}")

    def _parse_package_spec(self, spec: str) -> Optional[Tuple[str, str]]:
        """Parse package specification into name and version."""
        # Remove comments
        spec = spec.split('#')[0].strip()

        # Handle various package spec formats
        if '==' in spec:
            name, version = spec.split('==', 1)
        elif '>=' in spec:
            name, version = spec.split('>=', 1)
            version = f">={version}"
        elif '>' in spec:
            name, version = spec.split('>', 1)
            version = f">{version}"
        elif '<=' in spec:
            name, version = spec.split('<=', 1)
            version = f"<={version}"
        elif '<' in spec:
            name, version = spec.split('<', 1)
            version = f"<{version}"
        else:
            # No version specified
            name = spec
            version = "latest"

        return (name.strip(), version.strip())

    def _get_installed_packages(self):
        """Get information about installed packages."""
        try:
            import pkg_resources

            for dist in pkg_resources.working_set:
                self.report.total_packages += 1

                # Check if package is outdated
                if hasattr(dist, 'version'):
                    # This is a simplified check - in practice you'd query PyPI
                    pass

        except ImportError:
            print("Warning: pkg_resources not available, skipping installed package analysis")

    def _scan_vulnerabilities(self):
        """Scan for known vulnerabilities in dependencies."""
        print("🔒 Scanning for vulnerabilities...")

        # Use safety tool for vulnerability scanning
        try:
            result = subprocess.run(
                ['safety', 'check', '--json'],
                capture_output=True, text=True,
                cwd=self.project_root
            )

            if result.returncode == 0:
                # Parse safety output
                if result.stdout:
                    safety_data = json.loads(result.stdout)
                    self._parse_safety_output(safety_data)

        except FileNotFoundError:
            print("Warning: safety tool not found, skipping vulnerability scan")
            print("Install with: pip install safety")
        except Exception as e:
            print(f"Warning: Could not run vulnerability scan: {e}")

    def _parse_safety_output(self, safety_data: List[Dict]):
        """Parse Safety tool output for vulnerabilities."""
        for vuln in safety_data:
            package_name = vuln.get('package', 'unknown')
            version = vuln.get('installed_version', 'unknown')
            vuln_id = vuln.get('vulnerability_id', '')
            description = vuln.get('vulnerability_description', '')
            severity = vuln.get('severity', 'medium')

            # Map severity to our enum
            severity_map = {
                'critical': VulnerabilitySeverity.CRITICAL,
                'high': VulnerabilitySeverity.HIGH,
                'medium': VulnerabilitySeverity.MEDIUM,
                'low': VulnerabilitySeverity.LOW
            }

            vuln_severity = severity_map.get(severity.lower(), VulnerabilitySeverity.MEDIUM)

            issue = DependencyIssue(
                package_name=package_name,
                version=version,
                issue_type='VULNERABILITY',
                severity=vuln_severity,
                description=f"Vulnerability found: {description}",
                impact='Package has known security vulnerability',
                remediation='Update to a patched version',
                cve_id=vuln_id
            )

            self.report.add_issue(issue)
            self.report.vulnerable_packages += 1

    def _check_license_compliance(self):
        """Check license compliance of dependencies."""
        print("📄 Checking license compliance...")

        try:
            # Use pip-licenses to check licenses
            result = subprocess.run(
                ['pip-licenses', '--format=json'],
                capture_output=True, text=True,
                cwd=self.project_root
            )

            if result.returncode == 0 and result.stdout:
                licenses_data = json.loads(result.stdout)
                self._analyze_licenses(licenses_data)

        except FileNotFoundError:
            print("Warning: pip-licenses not found, skipping license check")
            print("Install with: pip install pip-licenses")
        except Exception as e:
            print(f"Warning: Could not check licenses: {e}")

    def _analyze_licenses(self, licenses_data: List[Dict]):
        """Analyze license information for compliance."""
        for package in licenses_data:
            package_name = package.get('Name', '')
            license_type = package.get('License', '')

            # Check for forbidden licenses
            if license_type in self.forbidden_licenses:
                issue = DependencyIssue(
                    package_name=package_name,
                    version=package.get('Version', 'unknown'),
                    issue_type='LICENSE_VIOLATION',
                    severity=VulnerabilitySeverity.HIGH,
                    description=f"Forbidden license: {license_type}",
                    impact='License incompatible with FEDZK distribution terms',
                    remediation='Replace with package using approved license'
                )
                self.report.add_issue(issue)
                self.report.license_issues += 1

            # Check for unknown licenses
            elif license_type and license_type not in self.approved_licenses:
                issue = DependencyIssue(
                    package_name=package_name,
                    version=package.get('Version', 'unknown'),
                    issue_type='UNKNOWN_LICENSE',
                    severity=VulnerabilitySeverity.MEDIUM,
                    description=f"Unknown license: {license_type}",
                    impact='License needs manual review',
                    remediation='Verify license compatibility and add to approved list if acceptable'
                )
                self.report.add_issue(issue)
                self.report.license_issues += 1

    def _check_outdated_packages(self):
        """Check for outdated packages."""
        print("📦 Checking for outdated packages...")

        try:
            result = subprocess.run(
                ['pip', 'list', '--outdated', '--format=json'],
                capture_output=True, text=True,
                cwd=self.project_root
            )

            if result.returncode == 0 and result.stdout:
                outdated_data = json.loads(result.stdout)
                self._analyze_outdated_packages(outdated_data)

        except Exception as e:
            print(f"Warning: Could not check for outdated packages: {e}")

    def _analyze_outdated_packages(self, outdated_data: List[Dict]):
        """Analyze outdated package information."""
        for package in outdated_data:
            package_name = package.get('name', '')
            current_version = package.get('version', '')
            latest_version = package.get('latest_version', '')

            # Calculate how outdated the package is
            try:
                current_parts = [int(x) for x in current_version.split('.') if x.isdigit()]
                latest_parts = [int(x) for x in latest_version.split('.') if x.isdigit()]

                if len(current_parts) >= 1 and len(latest_parts) >= 1:
                    major_diff = latest_parts[0] - current_parts[0]

                    if major_diff >= 1:
                        severity = VulnerabilitySeverity.MEDIUM
                        description = f"Major version update available: {current_version} -> {latest_version}"
                    else:
                        severity = VulnerabilitySeverity.LOW
                        description = f"Minor update available: {current_version} -> {latest_version}"
                else:
                    severity = VulnerabilitySeverity.LOW
                    description = f"Update available: {current_version} -> {latest_version}"

            except (ValueError, IndexError):
                severity = VulnerabilitySeverity.LOW
                description = f"Update available: {current_version} -> {latest_version}"

            issue = DependencyIssue(
                package_name=package_name,
                version=current_version,
                issue_type='OUTDATED_PACKAGE',
                severity=severity,
                description=description,
                impact='Package may have bug fixes or security updates',
                remediation=f"Update to version {latest_version} or later"
            )

            self.report.add_issue(issue)
            self.report.outdated_packages += 1

    def generate_security_report(self) -> str:
        """Generate comprehensive security report."""
        report_lines = []

        report_lines.append("# FEDZK Dependency Security Report")
        report_lines.append("=" * 50)
        report_lines.append("")

        # Overall status
        status = "✅ PASSED" if self.report.passed else "❌ FAILED"
        report_lines.append(f"Overall Status: {status}")
        report_lines.append("")

        # Summary statistics
        report_lines.append("## Summary Statistics")
        report_lines.append(f"- Total Packages: {self.report.total_packages}")
        report_lines.append(f"- Vulnerable Packages: {self.report.vulnerable_packages}")
        report_lines.append(f"- Outdated Packages: {self.report.outdated_packages}")
        report_lines.append(f"- License Issues: {self.report.license_issues}")
        report_lines.append("")

        if self.report.summary:
            report_lines.append("## Issues by Severity")
            for severity, count in self.report.summary.items():
                report_lines.append(f"- {severity}: {count}")
            report_lines.append("")

        # Critical vulnerabilities
        critical_issues = [i for i in self.report.issues
                          if i.severity == VulnerabilitySeverity.CRITICAL]

        if critical_issues:
            report_lines.append("## 🚨 Critical Issues (Immediate Action Required)")
            for issue in critical_issues:
                report_lines.append(f"### {issue.package_name} v{issue.version}")
                report_lines.append(f"- **Type**: {issue.issue_type}")
                report_lines.append(f"- **Description**: {issue.description}")
                report_lines.append(f"- **Impact**: {issue.impact}")
                if issue.cve_id:
                    report_lines.append(f"- **CVE**: {issue.cve_id}")
                if issue.remediation:
                    report_lines.append(f"- **Remediation**: {issue.remediation}")
                report_lines.append("")

        # High severity issues
        high_issues = [i for i in self.report.issues
                      if i.severity == VulnerabilitySeverity.HIGH]

        if high_issues:
            report_lines.append("## ⚠️ High Priority Issues")
            for issue in high_issues:
                report_lines.append(f"- **{issue.package_name}**: {issue.description}")
                if issue.remediation:
                    report_lines.append(f"  - 💡 {issue.remediation}")
            report_lines.append("")

        # Recommendations
        report_lines.append("## Recommendations")
        if not self.report.passed:
            report_lines.append("❌ **Security Issues Found**: Address critical and high severity issues before deployment.")
        else:
            report_lines.append("✅ **Dependencies Secure**: No critical security issues found.")

        report_lines.append("")
        report_lines.append("### Best Practices")
        report_lines.append("- Keep dependencies updated regularly")
        report_lines.append("- Review license compatibility before adding new dependencies")
        report_lines.append("- Use tools like `safety` and `pip-audit` for ongoing monitoring")
        report_lines.append("- Consider using `pip-tools` for reproducible builds")
        report_lines.append("- Regularly audit your dependency tree")
        report_lines.append("")

        # Compliance status
        report_lines.append("## Compliance Status")
        report_lines.append("### Licenses")
        report_lines.append(f"- Approved licenses: {len(self.approved_licenses)}")
        report_lines.append(f"- Issues found: {self.report.license_issues}")
        report_lines.append("")

        report_lines.append("### Security")
        report_lines.append(f"- Packages scanned: {self.report.total_packages}")
        report_lines.append(f"- Vulnerabilities found: {self.report.vulnerable_packages}")
        report_lines.append("")

        return "\n".join(report_lines)


def main():
    """Main entry point for dependency scanner."""
    project_root = Path(__file__).parent.parent

    scanner = FEDZKDependencyScanner(project_root)
    report = scanner.scan_dependencies()

    # Generate and print security report
    security_report = scanner.generate_security_report()
    print(security_report)

    # Save detailed report
    report_file = project_root / 'dependency-security-report.json'
    with open(report_file, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"\n📋 Detailed report saved to: {report_file}")

    # Exit with appropriate code
    if report.passed:
        print("\n✅ Dependency security scan passed!")
        sys.exit(0)
    else:
        print("\n❌ Dependency security scan failed!")
        print("Address critical and high severity issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
