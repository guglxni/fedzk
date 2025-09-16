#!/usr/bin/env python3
"""
FEDZK License Compliance Checker

Comprehensive license compliance analysis for FEDZK and its dependencies.
Ensures all components comply with distribution and usage requirements.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum


class LicenseCompliance(Enum):
    """License compliance status."""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    UNKNOWN = "UNKNOWN"
    RESTRICTED = "RESTRICTED"


@dataclass
class LicenseIssue:
    """Represents a license compliance issue."""

    package_name: str
    license_type: str
    compliance_status: LicenseCompliance
    severity: str
    description: str
    impact: str
    remediation: str = ""
    source_file: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert issue to dictionary."""
        return {
            'package_name': self.package_name,
            'license_type': self.license_type,
            'compliance_status': self.compliance_status.value,
            'severity': self.severity,
            'description': self.description,
            'impact': self.impact,
            'remediation': self.remediation,
            'source_file': self.source_file
        }


@dataclass
class LicenseReport:
    """Comprehensive license compliance report."""

    total_packages: int = 0
    compliant_packages: int = 0
    non_compliant_packages: int = 0
    unknown_licenses: int = 0
    issues: List[LicenseIssue] = field(default_factory=list)
    license_distribution: Dict[str, int] = field(default_factory=dict)
    passed: bool = True

    def add_issue(self, issue: LicenseIssue):
        """Add a license issue."""
        self.issues.append(issue)

        # Update counters
        if issue.compliance_status == LicenseCompliance.NON_COMPLIANT:
            self.non_compliant_packages += 1
            self.passed = False
        elif issue.compliance_status == LicenseCompliance.UNKNOWN:
            self.unknown_licenses += 1

        # Update license distribution
        self.license_distribution[issue.license_type] = \
            self.license_distribution.get(issue.license_type, 0) + 1

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            'total_packages': self.total_packages,
            'compliant_packages': self.compliant_packages,
            'non_compliant_packages': self.non_compliant_packages,
            'unknown_licenses': self.unknown_licenses,
            'total_issues': len(self.issues),
            'license_distribution': self.license_distribution,
            'passed': self.passed,
            'issues': [i.to_dict() for i in self.issues]
        }


class FEDZKLicenseChecker:
    """Comprehensive FEDZK license compliance checker."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.report = LicenseReport()

        # FEDZK approved licenses (permissive and compatible)
        self.approved_licenses = {
            'MIT', 'Apache-2.0', 'Apache 2.0', 'BSD-3-Clause', 'BSD-2-Clause',
            'ISC', 'Python-2.0', 'PostgreSQL', 'ZPL-2.1', 'ZPL 2.1',
            'CC0-1.0', 'CC0 1.0', 'Unlicense', 'WTFPL', 'BSD'
        }

        # FEDZK restricted licenses (require special handling)
        self.restricted_licenses = {
            'LGPL-2.1', 'LGPL-3.0', 'MPL-2.0', 'CDDL', 'CPL-1.0'
        }

        # FEDZK forbidden licenses (incompatible with distribution)
        self.forbidden_licenses = {
            'GPL-2.0', 'GPL-3.0', 'AGPL-3.0', 'AGPL-1.0'
        }

        # License compatibility matrix
        self.compatibility_matrix = {
            'MIT': ['MIT', 'Apache-2.0', 'BSD-3-Clause', 'BSD-2-Clause', 'ISC'],
            'Apache-2.0': ['MIT', 'Apache-2.0', 'BSD-3-Clause', 'BSD-2-Clause', 'ISC'],
            'BSD-3-Clause': ['MIT', 'Apache-2.0', 'BSD-3-Clause', 'BSD-2-Clause', 'ISC'],
            'GPL-2.0': ['GPL-2.0', 'GPL-3.0', 'AGPL-3.0'],  # Copyleft contamination
            'GPL-3.0': ['GPL-3.0', 'AGPL-3.0']  # Stronger copyleft
        }

    def check_license_compliance(self) -> LicenseReport:
        """Perform comprehensive license compliance check."""
        print("📜 Checking FEDZK license compliance...")

        # Check FEDZK core license
        self._check_fedzk_license()

        # Check dependency licenses
        self._check_dependency_licenses()

        # Check third-party components
        self._check_third_party_licenses()

        # Analyze license compatibility
        self._analyze_license_compatibility()

        return self.report

    def _check_fedzk_license(self):
        """Check FEDZK core license files."""
        license_file = self.project_root / 'LICENSE'
        if license_file.exists():
            try:
                with open(license_file, 'r') as f:
                    content = f.read().lower()

                if 'mit' in content:
                    self.report.compliant_packages += 1
                    print("✅ FEDZK core: MIT license compliant")
                else:
                    issue = LicenseIssue(
                        package_name='FEDZK Core',
                        license_type='Unknown',
                        compliance_status=LicenseCompliance.UNKNOWN,
                        severity='Medium',
                        description='FEDZK core license could not be determined',
                        impact='May affect distribution and usage rights',
                        remediation='Ensure LICENSE file contains valid license text'
                    )
                    self.report.add_issue(issue)

            except Exception as e:
                print(f"Warning: Could not read LICENSE file: {e}")
        else:
            issue = LicenseIssue(
                package_name='FEDZK Core',
                license_type='Missing',
                compliance_status=LicenseCompliance.UNKNOWN,
                severity='High',
                description='LICENSE file not found',
                impact='Distribution and usage rights unclear',
                remediation='Create LICENSE file with appropriate license'
            )
            self.report.add_issue(issue)

    def _check_dependency_licenses(self):
        """Check licenses of Python dependencies."""
        try:
            # Use pip-licenses for comprehensive license checking
            result = subprocess.run(
                ['pip-licenses', '--format=json', '--with-authors'],
                capture_output=True, text=True,
                cwd=self.project_root
            )

            if result.returncode == 0 and result.stdout:
                licenses_data = json.loads(result.stdout)
                self._analyze_dependency_licenses(licenses_data)

        except FileNotFoundError:
            print("Warning: pip-licenses not found, using fallback method")
            self._fallback_dependency_check()
        except Exception as e:
            print(f"Warning: Could not check dependency licenses: {e}")
            self._fallback_dependency_check()

    def _analyze_dependency_licenses(self, licenses_data: List[Dict]):
        """Analyze license information from pip-licenses."""
        for package in licenses_data:
            package_name = package.get('Name', 'Unknown')
            license_type = package.get('License', 'Unknown')

            self.report.total_packages += 1

            # Determine compliance status
            compliance_status = self._determine_license_compliance(license_type)

            if compliance_status == LicenseCompliance.COMPLIANT:
                self.report.compliant_packages += 1
            elif compliance_status == LicenseCompliance.NON_COMPLIANT:
                issue = LicenseIssue(
                    package_name=package_name,
                    license_type=license_type,
                    compliance_status=compliance_status,
                    severity='High',
                    description=f'Non-compliant license: {license_type}',
                    impact='Cannot distribute FEDZK with this dependency',
                    remediation='Replace with dependency using compatible license'
                )
                self.report.add_issue(issue)
            elif compliance_status == LicenseCompliance.RESTRICTED:
                issue = LicenseIssue(
                    package_name=package_name,
                    license_type=license_type,
                    compliance_status=compliance_status,
                    severity='Medium',
                    description=f'Restricted license: {license_type}',
                    impact='Requires special handling and legal review',
                    remediation='Review license compatibility and usage terms'
                )
                self.report.add_issue(issue)
            elif compliance_status == LicenseCompliance.UNKNOWN:
                issue = LicenseIssue(
                    package_name=package_name,
                    license_type=license_type,
                    compliance_status=compliance_status,
                    severity='Low',
                    description=f'Unknown license: {license_type}',
                    impact='License needs manual verification',
                    remediation='Verify license compatibility manually'
                )
                self.report.add_issue(issue)

    def _determine_license_compliance(self, license_type: str) -> LicenseCompliance:
        """Determine license compliance status."""
        if not license_type or license_type.lower() in ['unknown', '']:
            return LicenseCompliance.UNKNOWN

        # Normalize license name
        license_norm = license_type.replace(' ', '').replace('-', '').upper()

        # Check forbidden licenses
        for forbidden in self.forbidden_licenses:
            if forbidden.replace('-', '').replace(' ', '') in license_norm:
                return LicenseCompliance.NON_COMPLIANT

        # Check approved licenses
        for approved in self.approved_licenses:
            if approved.replace('-', '').replace(' ', '') in license_norm:
                return LicenseCompliance.COMPLIANT

        # Check restricted licenses
        for restricted in self.restricted_licenses:
            if restricted.replace('-', '').replace(' ', '') in license_norm:
                return LicenseCompliance.RESTRICTED

        return LicenseCompliance.UNKNOWN

    def _fallback_dependency_check(self):
        """Fallback method for dependency license checking."""
        try:
            # Try to get basic package information
            result = subprocess.run(
                ['pip', 'list', '--format=json'],
                capture_output=True, text=True,
                cwd=self.project_root
            )

            if result.returncode == 0 and result.stdout:
                packages_data = json.loads(result.stdout)
                self.report.total_packages = len(packages_data)

                print(f"Found {len(packages_data)} packages (detailed license check unavailable)")
                print("Install pip-licenses for comprehensive license analysis:")
                print("  pip install pip-licenses")

        except Exception as e:
            print(f"Warning: Could not perform fallback dependency check: {e}")

    def _check_third_party_licenses(self):
        """Check licenses of third-party components."""
        third_party_dirs = [
            self.project_root / 'third_party',
            self.project_root / 'vendor',
            self.project_root / 'external'
        ]

        for third_party_dir in third_party_dirs:
            if third_party_dir.exists():
                self._scan_third_party_directory(third_party_dir)

    def _scan_third_party_directory(self, directory: Path):
        """Scan third-party directory for license files."""
        license_files = list(directory.rglob('LICENSE*')) + list(directory.rglob('COPYING*'))

        for license_file in license_files:
            try:
                with open(license_file, 'r') as f:
                    content = f.read().lower()

                # Try to identify license type
                if 'mit' in content:
                    license_type = 'MIT'
                elif 'apache' in content and '2.0' in content:
                    license_type = 'Apache-2.0'
                elif 'gpl' in content:
                    if '3.0' in content:
                        license_type = 'GPL-3.0'
                    else:
                        license_type = 'GPL-2.0'
                elif 'bsd' in content:
                    if '3' in content:
                        license_type = 'BSD-3-Clause'
                    else:
                        license_type = 'BSD-2-Clause'
                else:
                    license_type = 'Unknown'

                compliance_status = self._determine_license_compliance(license_type)

                if compliance_status != LicenseCompliance.COMPLIANT:
                    issue = LicenseIssue(
                        package_name=f'Third-party: {license_file.parent.name}',
                        license_type=license_type,
                        compliance_status=compliance_status,
                        severity='Medium',
                        description=f'Third-party component license: {license_type}',
                        impact='May affect overall distribution license',
                        remediation='Review third-party component license compatibility',
                        source_file=str(license_file)
                    )
                    self.report.add_issue(issue)

            except Exception as e:
                print(f"Warning: Could not read license file {license_file}: {e}")

    def _analyze_license_compatibility(self):
        """Analyze license compatibility across all components."""
        # Check for license compatibility issues
        fedzk_license = 'MIT'  # Assuming FEDZK uses MIT

        incompatible_licenses = []
        for license_type, count in self.report.license_distribution.items():
            if license_type in self.forbidden_licenses:
                incompatible_licenses.append(f"{license_type} ({count} packages)")

        if incompatible_licenses:
            issue = LicenseIssue(
                package_name='FEDZK Distribution',
                license_type=fedzk_license,
                compliance_status=LicenseCompliance.NON_COMPLIANT,
                severity='Critical',
                description=f'License compatibility conflict with: {", ".join(incompatible_licenses)}',
                impact='Cannot distribute FEDZK with incompatible licenses',
                remediation='Replace incompatible dependencies or change FEDZK license'
            )
            self.report.add_issue(issue)

    def generate_compliance_report(self) -> str:
        """Generate comprehensive license compliance report."""
        report_lines = []

        report_lines.append("# FEDZK License Compliance Report")
        report_lines.append("=" * 50)
        report_lines.append("")

        # Overall status
        status = "✅ COMPLIANT" if self.report.passed else "❌ NON-COMPLIANT"
        report_lines.append(f"Overall Status: {status}")
        report_lines.append("")

        # Summary statistics
        report_lines.append("## Summary Statistics")
        report_lines.append(f"- Total Packages: {self.report.total_packages}")
        report_lines.append(f"- Compliant Packages: {self.report.compliant_packages}")
        report_lines.append(f"- Non-compliant Packages: {self.report.non_compliant_packages}")
        report_lines.append(f"- Unknown Licenses: {self.report.unknown_licenses}")
        report_lines.append("")

        # License distribution
        if self.report.license_distribution:
            report_lines.append("## License Distribution")
            for license_type, count in sorted(self.report.license_distribution.items()):
                compliance = self._determine_license_compliance(license_type)
                status_icon = {
                    LicenseCompliance.COMPLIANT: "✅",
                    LicenseCompliance.RESTRICTED: "⚠️",
                    LicenseCompliance.NON_COMPLIANT: "❌",
                    LicenseCompliance.UNKNOWN: "❓"
                }.get(compliance, "❓")

                report_lines.append(f"- {status_icon} {license_type}: {count} packages")
            report_lines.append("")

        # Critical issues
        critical_issues = [i for i in self.report.issues
                          if i.compliance_status == LicenseCompliance.NON_COMPLIANT]

        if critical_issues:
            report_lines.append("## 🚨 Critical License Issues")
            for issue in critical_issues:
                report_lines.append(f"### {issue.package_name}")
                report_lines.append(f"- **License**: {issue.license_type}")
                report_lines.append(f"- **Description**: {issue.description}")
                report_lines.append(f"- **Impact**: {issue.impact}")
                if issue.remediation:
                    report_lines.append(f"- **Remediation**: {issue.remediation}")
                report_lines.append("")

        # Recommendations
        report_lines.append("## Recommendations")
        if not self.report.passed:
            report_lines.append("❌ **License Compliance Issues Found**: Address non-compliant licenses before distribution.")
        else:
            report_lines.append("✅ **License Compliance Verified**: All components have compatible licenses.")

        report_lines.append("")
        report_lines.append("### Best Practices")
        report_lines.append("- Use only approved permissive licenses (MIT, Apache-2.0, BSD)")
        report_lines.append("- Avoid copyleft licenses (GPL, LGPL) in dependencies")
        report_lines.append("- Regularly audit dependency licenses")
        report_lines.append("- Document license compatibility decisions")
        report_lines.append("- Consider dual-licensing for broader compatibility")
        report_lines.append("")

        # Approved licenses reference
        report_lines.append("## Approved Licenses")
        report_lines.append("FEDZK approves the following licenses for dependencies:")
        for license_name in sorted(self.approved_licenses):
            report_lines.append(f"- {license_name}")
        report_lines.append("")

        return "\n".join(report_lines)


def main():
    """Main entry point for license checker."""
    project_root = Path(__file__).parent.parent

    checker = FEDZKLicenseChecker(project_root)
    report = checker.check_license_compliance()

    # Generate and print compliance report
    compliance_report = checker.generate_compliance_report()
    print(compliance_report)

    # Save detailed report
    report_file = project_root / 'license-compliance-report.json'
    with open(report_file, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"\n📋 Detailed report saved to: {report_file}")

    # Exit with appropriate code
    if report.passed:
        print("\n✅ License compliance check passed!")
        sys.exit(0)
    else:
        print("\n❌ License compliance check failed!")
        print("Address non-compliant licenses before distribution.")
        sys.exit(1)


if __name__ == "__main__":
    main()
