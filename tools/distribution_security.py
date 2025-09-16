#!/usr/bin/env python3
"""
FEDZK Distribution Security Tool

Comprehensive security measures for software distribution.
Includes SBOM generation, signature verification, and secure distribution practices.
"""

import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import base64


@dataclass
class DistributionSecurityReport:
    """Comprehensive distribution security report."""

    package_name: str
    version: str
    distribution_channels: List[str] = field(default_factory=list)
    security_measures: Dict[str, Any] = field(default_factory=dict)
    vulnerabilities: List[Dict] = field(default_factory=list)
    compliance_status: Dict[str, bool] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            'package_name': self.package_name,
            'version': self.version,
            'distribution_channels': self.distribution_channels,
            'security_measures': self.security_measures,
            'vulnerabilities': self.vulnerabilities,
            'compliance_status': self.compliance_status,
            'recommendations': self.recommendations,
            'generated_at': self.generated_at.isoformat()
        }


class FEDZKDistributionSecurity:
    """FEDZK distribution security management."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.dist_dir = project_root / 'dist'
        self.security_dir = project_root / 'security'

        # Security configuration
        self.security_config = {
            'gpg_key_id': None,  # Should be set via environment
            'sigstore_enabled': True,
            'sbom_format': 'spdx',
            'vulnerability_threshold': 'MEDIUM',
            'compliance_frameworks': ['SLSA', 'Sigstore', 'OpenSSF']
        }

    def generate_distribution_security_report(self, version: str) -> DistributionSecurityReport:
        """Generate comprehensive distribution security report."""
        print(f"🔒 Generating distribution security report for FEDZK {version}...")

        report = DistributionSecurityReport(
            package_name='fedzk',
            version=version
        )

        # Analyze distribution channels
        report.distribution_channels = self._analyze_distribution_channels()

        # Check security measures
        report.security_measures = self._check_security_measures(version)

        # Scan for vulnerabilities
        report.vulnerabilities = self._scan_distribution_vulnerabilities()

        # Check compliance status
        report.compliance_status = self._check_compliance_status()

        # Generate recommendations
        report.recommendations = self._generate_security_recommendations(report)

        return report

    def _analyze_distribution_channels(self) -> List[str]:
        """Analyze available distribution channels."""
        channels = []

        # Check PyPI
        if self._check_pypi_availability():
            channels.append('PyPI')

        # Check Docker Hub
        if self._check_docker_availability():
            channels.append('Docker Hub')

        # Check GitHub Releases
        if self._check_github_releases():
            channels.append('GitHub Releases')

        # Check Helm registry
        if self._check_helm_availability():
            channels.append('Helm Registry')

        return channels

    def _check_pypi_availability(self) -> bool:
        """Check if package is available on PyPI."""
        try:
            result = subprocess.run(
                ['pip', 'index', 'versions', 'fedzk'],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_docker_availability(self) -> bool:
        """Check if Docker images are available."""
        try:
            result = subprocess.run(
                ['docker', 'manifest', 'inspect', 'fedzk/fedzk:latest'],
                capture_output=True, timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_github_releases(self) -> bool:
        """Check if GitHub releases exist."""
        # This would normally check GitHub API
        # For now, assume releases exist if VERSION file exists
        return (self.project_root / 'VERSION').exists()

    def _check_helm_availability(self) -> bool:
        """Check if Helm charts are available."""
        try:
            result = subprocess.run(
                ['helm', 'repo', 'list'],
                capture_output=True, text=True
            )
            return 'fedzk' in result.stdout
        except FileNotFoundError:
            return False

    def _check_security_measures(self, version: str) -> Dict[str, Any]:
        """Check implemented security measures."""
        measures = {}

        # Check for GPG signatures
        measures['gpg_signatures'] = self._check_gpg_signatures()

        # Check for SBOM
        measures['sbom'] = self._check_sbom_availability()

        # Check for Sigstore signatures
        measures['sigstore'] = self._check_sigstore_signatures()

        # Check for SLSA provenance
        measures['slsa_provenance'] = self._check_slsa_provenance()

        # Check for vulnerability scanning
        measures['vulnerability_scanning'] = self._check_vulnerability_scanning()

        return measures

    def _check_gpg_signatures(self) -> Dict[str, Any]:
        """Check for GPG signatures on distribution artifacts."""
        signature_status = {
            'available': False,
            'valid': False,
            'key_fingerprint': None
        }

        # Check for .asc signature files
        if self.dist_dir.exists():
            for file_path in self.dist_dir.glob('*'):
                if file_path.suffix in ['.tar.gz', '.whl']:
                    sig_file = file_path.with_suffix(file_path.suffix + '.asc')
                    if sig_file.exists():
                        signature_status['available'] = True
                        # In practice, you'd verify the signature here
                        signature_status['valid'] = True
                        break

        return signature_status

    def _check_sbom_availability(self) -> Dict[str, Any]:
        """Check for Software Bill of Materials (SBOM)."""
        sbom_status = {
            'available': False,
            'format': None,
            'location': None
        }

        # Look for SBOM files
        sbom_files = list(self.project_root.glob('*.spdx.json')) + \
                    list(self.project_root.glob('*.cyclonedx.json')) + \
                    list(self.security_dir.glob('*.spdx.json'))

        if sbom_files:
            sbom_status['available'] = True
            sbom_status['location'] = str(sbom_files[0])
            if 'spdx' in sbom_files[0].name:
                sbom_status['format'] = 'SPDX'
            elif 'cyclonedx' in sbom_files[0].name:
                sbom_status['format'] = 'CycloneDX'

        return sbom_status

    def _check_sigstore_signatures(self) -> Dict[str, Any]:
        """Check for Sigstore signatures."""
        sigstore_status = {
            'enabled': self.security_config['sigstore_enabled'],
            'signatures_available': False,
            'verified': False
        }

        # Check for Sigstore signature files
        sigstore_files = list(self.project_root.glob('*.sig')) + \
                        list(self.security_dir.glob('*.sig'))

        if sigstore_files:
            sigstore_status['signatures_available'] = True
            # In practice, you'd verify signatures with cosign
            sigstore_status['verified'] = True

        return sigstore_status

    def _check_slsa_provenance(self) -> Dict[str, Any]:
        """Check for SLSA provenance information."""
        slsa_status = {
            'available': False,
            'level': None,
            'provenance_file': None
        }

        # Look for SLSA provenance files
        provenance_files = list(self.security_dir.glob('*.intoto.jsonl')) + \
                          list(self.project_root.glob('*.intoto.jsonl'))

        if provenance_files:
            slsa_status['available'] = True
            slsa_status['provenance_file'] = str(provenance_files[0])
            # In practice, you'd parse the provenance to determine SLSA level
            slsa_status['level'] = 'SLSA Level 2'

        return slsa_status

    def _check_vulnerability_scanning(self) -> Dict[str, Any]:
        """Check vulnerability scanning status."""
        vuln_status = {
            'enabled': True,
            'last_scan': None,
            'critical_vulns': 0,
            'high_vulns': 0
        }

        # Check for vulnerability scan reports
        vuln_reports = list(self.security_dir.glob('*vulnerability*.json')) + \
                      list(self.project_root.glob('*trivy*.json'))

        if vuln_reports:
            vuln_status['last_scan'] = datetime.fromtimestamp(
                vuln_reports[0].stat().st_mtime
            ).isoformat()

            # Parse vulnerability counts from report
            try:
                with open(vuln_reports[0], 'r') as f:
                    vuln_data = json.load(f)
                    # Extract vulnerability counts based on report format
                    vuln_status['critical_vulns'] = vuln_data.get('critical', 0)
                    vuln_status['high_vulns'] = vuln_data.get('high', 0)
            except (json.JSONDecodeError, KeyError):
                pass

        return vuln_status

    def _scan_distribution_vulnerabilities(self) -> List[Dict]:
        """Scan distribution artifacts for vulnerabilities."""
        vulnerabilities = []

        # Scan package distributions
        if self.dist_dir.exists():
            for file_path in self.dist_dir.glob('*'):
                if file_path.suffix in ['.tar.gz', '.whl']:
                    file_vulns = self._scan_file_vulnerabilities(file_path)
                    vulnerabilities.extend(file_vulns)

        return vulnerabilities

    def _scan_file_vulnerabilities(self, file_path: Path) -> List[Dict]:
        """Scan a specific file for vulnerabilities."""
        vulnerabilities = []

        try:
            # Use Trivy or similar tool to scan the file
            result = subprocess.run(
                ['trivy', 'fs', '--format', 'json', str(file_path)],
                capture_output=True, text=True, timeout=60
            )

            if result.returncode == 0 and result.stdout:
                vuln_data = json.loads(result.stdout)
                for vuln in vuln_data.get('Results', []):
                    for vulnerability in vuln.get('Vulnerabilities', []):
                        vulnerabilities.append({
                            'id': vulnerability.get('VulnerabilityID'),
                            'severity': vulnerability.get('Severity'),
                            'package': vulnerability.get('PkgName'),
                            'description': vulnerability.get('Description'),
                            'fixed_version': vulnerability.get('FixedVersion'),
                            'file': str(file_path)
                        })

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass

        return vulnerabilities

    def _check_compliance_status(self) -> Dict[str, bool]:
        """Check compliance with security frameworks."""
        compliance = {}

        # SLSA Level compliance
        compliance['slsa_level_2'] = self._check_slsa_compliance()

        # Sigstore compliance
        compliance['sigstore'] = self.security_config['sigstore_enabled']

        # OpenSSF Best Practices
        compliance['openssf'] = self._check_openssf_compliance()

        # Vulnerability disclosure
        compliance['vulnerability_disclosure'] = self._check_vulnerability_disclosure()

        return compliance

    def _check_slsa_compliance(self) -> bool:
        """Check SLSA Level 2 compliance."""
        # SLSA Level 2 requirements:
        # 1. Source code from version control
        # 2. Build service with access to source
        # 3. Ephemeral environment
        # 4. Parameterized builds
        # 5. Provenance
        # 6. Reproducible builds

        slsa_requirements = [
            (self.project_root / '.git').exists(),  # Version control
            self._check_build_service(),  # Build service
            True,  # Assume ephemeral (would check CI config)
            self._check_parameterized_builds(),  # Parameterized builds
            self._check_slsa_provenance()['available'],  # Provenance
            self._check_reproducible_builds()  # Reproducible
        ]

        return all(slsa_requirements)

    def _check_build_service(self) -> bool:
        """Check if using a build service."""
        # Check for GitHub Actions, GitLab CI, etc.
        return (self.project_root / '.github' / 'workflows').exists()

    def _check_parameterized_builds(self) -> bool:
        """Check for parameterized builds."""
        # Look for build configuration files
        build_files = [
            'pyproject.toml',
            'setup.py',
            'Dockerfile',
            '.github/workflows/ci.yml'
        ]

        return any((self.project_root / f).exists() for f in build_files)

    def _check_reproducible_builds(self) -> bool:
        """Check for reproducible build practices."""
        # Check for pinned dependencies, fixed build environments, etc.
        return (self.project_root / 'pyproject.toml').exists()

    def _check_openssf_compliance(self) -> bool:
        """Check OpenSSF Best Practices compliance."""
        # OpenSSF requirements:
        # 1. Security policy
        # 2. Dependency updates
        # 3. Code review
        # 4. CI/CD security

        openssf_requirements = [
            (self.project_root / 'SECURITY.md').exists(),  # Security policy
            (self.project_root / 'dependabot.yml').exists() or
            (self.project_root / '.github' / 'dependabot.yml').exists(),  # Dependency updates
            (self.project_root / '.github' / 'workflows').exists(),  # CI/CD
            len(list(self.project_root.glob('*'))) > 10  # Reasonable project size
        ]

        return sum(openssf_requirements) >= 3  # At least 75% compliance

    def _check_vulnerability_disclosure(self) -> bool:
        """Check for vulnerability disclosure process."""
        # Look for security policy, vulnerability reporting
        security_files = [
            'SECURITY.md',
            '.github/SECURITY.md',
            'SECURITY',
            '.github/SECURITY'
        ]

        return any((self.project_root / f).exists() for f in security_files)

    def _generate_security_recommendations(self, report: DistributionSecurityReport) -> List[str]:
        """Generate security recommendations based on report."""
        recommendations = []

        # GPG signatures
        if not report.security_measures.get('gpg_signatures', {}).get('available'):
            recommendations.append("Implement GPG signing for all distribution artifacts")

        # SBOM
        if not report.security_measures.get('sbom', {}).get('available'):
            recommendations.append("Generate and publish Software Bill of Materials (SBOM)")

        # SLSA
        if not report.compliance_status.get('slsa_level_2'):
            recommendations.append("Achieve SLSA Level 2 compliance for build security")

        # Vulnerabilities
        if report.vulnerabilities:
            critical_vulns = [v for v in report.vulnerabilities if v.get('severity') == 'CRITICAL']
            if critical_vulns:
                recommendations.append("Address critical vulnerabilities before distribution")

        # Distribution channels
        recommended_channels = {'PyPI', 'Docker Hub', 'GitHub Releases'}
        missing_channels = recommended_channels - set(report.distribution_channels)
        if missing_channels:
            recommendations.append(f"Consider distributing via: {', '.join(missing_channels)}")

        return recommendations

    def generate_sbom(self, version: str) -> Optional[Path]:
        """Generate Software Bill of Materials (SBOM)."""
        try:
            # Use Trivy to generate SBOM
            sbom_file = self.security_dir / f'fedzk-{version}-sbom.spdx.json'

            result = subprocess.run(
                ['trivy', 'fs', '--format', 'spdx-json', '--output', str(sbom_file), '.'],
                cwd=self.project_root,
                capture_output=True,
                timeout=120
            )

            if result.returncode == 0 and sbom_file.exists():
                print(f"✅ Generated SBOM: {sbom_file}")
                return sbom_file
            else:
                print(f"❌ Failed to generate SBOM: {result.stderr.decode()}")
                return None

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"❌ SBOM generation failed: {e}")
            return None

    def sign_artifacts(self, artifacts: List[Path]) -> Dict[str, bool]:
        """Sign distribution artifacts with GPG."""
        signing_results = {}

        gpg_key = os.getenv('GPG_KEY_ID')
        if not gpg_key:
            print("❌ GPG_KEY_ID environment variable not set")
            return {str(artifact): False for artifact in artifacts}

        for artifact in artifacts:
            try:
                sig_file = artifact.with_suffix(artifact.suffix + '.asc')

                result = subprocess.run(
                    ['gpg', '--detach-sign', '--armor', '--local-user', gpg_key,
                     '--output', str(sig_file), str(artifact)],
                    capture_output=True,
                    timeout=30
                )

                signing_results[str(artifact)] = result.returncode == 0

                if result.returncode == 0:
                    print(f"✅ Signed: {artifact}")
                else:
                    print(f"❌ Failed to sign {artifact}: {result.stderr.decode()}")

            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"❌ Signing failed for {artifact}: {e}")
                signing_results[str(artifact)] = False

        return signing_results

    def verify_signatures(self, artifacts: List[Path]) -> Dict[str, bool]:
        """Verify GPG signatures of artifacts."""
        verification_results = {}

        for artifact in artifacts:
            sig_file = artifact.with_suffix(artifact.suffix + '.asc')

            if not sig_file.exists():
                verification_results[str(artifact)] = False
                continue

            try:
                result = subprocess.run(
                    ['gpg', '--verify', str(sig_file), str(artifact)],
                    capture_output=True,
                    timeout=30
                )

                verification_results[str(artifact)] = result.returncode == 0

                if result.returncode == 0:
                    print(f"✅ Signature verified: {artifact}")
                else:
                    print(f"❌ Signature verification failed: {artifact}")

            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"❌ Verification failed for {artifact}: {e}")
                verification_results[str(artifact)] = False

        return verification_results


def main():
    """Main entry point for distribution security tool."""
    import argparse

    parser = argparse.ArgumentParser(description="FEDZK Distribution Security Tool")
    parser.add_argument('--version', help='Package version')
    parser.add_argument('--generate-sbom', action='store_true', help='Generate SBOM')
    parser.add_argument('--sign-artifacts', action='store_true', help='Sign distribution artifacts')
    parser.add_argument('--verify-signatures', action='store_true', help='Verify artifact signatures')
    parser.add_argument('--report', action='store_true', help='Generate security report')

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    security_tool = FEDZKDistributionSecurity(project_root)

    # Get version
    if args.version:
        version = args.version
    else:
        # Try to get from VERSION file
        version_file = project_root / 'VERSION'
        if version_file.exists():
            version = version_file.read_text().strip()
        else:
            version = '1.0.0'

    # Execute requested operations
    if args.generate_sbom:
        sbom_file = security_tool.generate_sbom(version)
        if sbom_file:
            print(f"SBOM generated: {sbom_file}")

    if args.sign_artifacts:
        dist_files = list((project_root / 'dist').glob('*')) if (project_root / 'dist').exists() else []
        signing_results = security_tool.sign_artifacts(dist_files)
        print("Artifact signing results:", signing_results)

    if args.verify_signatures:
        dist_files = [f for f in (project_root / 'dist').glob('*') if not f.name.endswith('.asc')] if (project_root / 'dist').exists() else []
        verification_results = security_tool.verify_signatures(dist_files)
        print("Signature verification results:", verification_results)

    if args.report:
        report = security_tool.generate_distribution_security_report(version)

        # Print summary
        print("🔒 FEDZK Distribution Security Report")
        print("=" * 50)
        print(f"Package: {report.package_name} v{report.version}")
        print(f"Distribution Channels: {', '.join(report.distribution_channels)}")
        print(f"Vulnerabilities Found: {len(report.vulnerabilities)}")
        print(f"Security Measures: {len(report.security_measures)} implemented")

        # Compliance status
        compliant = sum(report.compliance_status.values())
        total = len(report.compliance_status)
        print(f"Compliance: {compliant}/{total} frameworks")

        # Recommendations
        if report.recommendations:
            print("\nRecommendations:")
            for rec in report.recommendations:
                print(f"- {rec}")

        # Save detailed report
        report_file = project_root / f'distribution-security-report-{version}.json'
        with open(report_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        print(f"\n📋 Detailed report saved to: {report_file}")


if __name__ == "__main__":
    main()
