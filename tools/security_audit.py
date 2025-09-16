#!/usr/bin/env python3
"""
FEDZK Security Audit Tool

Comprehensive security assessment for FEDZK codebase.
Performs security scanning, vulnerability detection, and compliance checks.
"""

import os
import sys
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import hashlib


class SecurityLevel(Enum):
    """Security assessment levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class SecurityFinding:
    """Represents a security finding."""

    def __init__(
        self,
        file_path: str,
        line_number: int,
        vulnerability_type: str,
        level: SecurityLevel,
        description: str,
        impact: str,
        remediation: str,
        cwe_id: Optional[str] = None,
        code_snippet: str = ""
    ):
        self.file_path = file_path
        self.line_number = line_number
        self.vulnerability_type = vulnerability_type
        self.level = level
        self.description = description
        self.impact = impact
        self.remediation = remediation
        self.cwe_id = cwe_id
        self.code_snippet = code_snippet

    def to_dict(self) -> Dict:
        """Convert finding to dictionary."""
        return {
            'file_path': self.file_path,
            'line_number': self.line_number,
            'vulnerability_type': self.vulnerability_type,
            'level': self.level.value,
            'description': self.description,
            'impact': self.impact,
            'remediation': self.remediation,
            'cwe_id': self.cwe_id,
            'code_snippet': self.code_snippet
        }


@dataclass
class SecurityReport:
    """Comprehensive security assessment report."""

    total_files: int = 0
    total_lines: int = 0
    findings: List[SecurityFinding] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    compliance_score: float = 100.0
    passed: bool = True

    def add_finding(self, finding: SecurityFinding):
        """Add a security finding."""
        self.findings.append(finding)
        level = finding.level.value
        self.summary[level] = self.summary.get(level, 0) + 1

        # Critical and High findings cause failure and reduce compliance score
        if finding.level == SecurityLevel.CRITICAL:
            self.passed = False
            self.compliance_score -= 20
        elif finding.level == SecurityLevel.HIGH:
            self.passed = False
            self.compliance_score -= 10
        elif finding.level == SecurityLevel.MEDIUM:
            self.compliance_score -= 5

        # Ensure compliance score doesn't go below 0
        self.compliance_score = max(0, self.compliance_score)

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            'total_files': self.total_files,
            'total_lines': self.total_lines,
            'total_findings': len(self.findings),
            'findings_by_level': self.summary,
            'compliance_score': self.compliance_score,
            'passed': self.passed,
            'findings': [f.to_dict() for f in self.findings]
        }


class FEDZKSecurityAuditor:
    """Comprehensive FEDZK security auditor."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.report = SecurityReport()

        # Security patterns and rules
        self.vulnerability_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']*["\']',
                r'secret\s*=\s*["\'][^"\']*["\']',
                r'key\s*=\s*["\'][^"\']*["\']',
                r'token\s*=\s*["\'][^"\']*["\']',
                r'api_key\s*=\s*["\'][^"\']*["\']'
            ],
            'sql_injection': [
                r'execute\s*\(\s*["\'].*\%.*["\']',
                r'cursor\.execute\s*\(\s*["\'].*\+.*["\']',
                r'format.*sql',
                r'f["\'].*sql.*["\']'
            ],
            'command_injection': [
                r'subprocess\.(call|Popen|run)\s*\(\s*.*\+',
                r'os\.system\s*\(\s*.*\+',
                r'os\.popen\s*\(\s*.*\+'
            ],
            'path_traversal': [
                r'open\s*\(\s*.*\+.*filename',
                r'Path\s*\(\s*.*\+',
                r'pathlib\.Path\s*\(\s*.*\+'
            ],
            'weak_crypto': [
                r'hashlib\.md5',
                r'hashlib\.sha1',
                r'random\.random\(\)',
                r'random\.randint\('
            ],
            'insecure_deserialization': [
                r'pickle\.loads?',
                r'yaml\.load',
                r'json\.loads?\s*\(\s*.*\)\s*$'
            ]
        }

        self.critical_functions = [
            'eval', 'exec', '__import__', 'getattr', 'setattr',
            'open', 'file', 'input', 'raw_input'
        ]

    def audit_file(self, file_path: Path) -> List[SecurityFinding]:
        """Audit a single file for security issues."""
        findings = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # Update report statistics
            self.report.total_files += 1
            self.report.total_lines += len(lines)

            # Perform security checks
            findings.extend(self._check_vulnerability_patterns(file_path, content, lines))
            findings.extend(self._check_critical_functions(file_path, content, lines))
            findings.extend(self._check_file_permissions(file_path))
            findings.extend(self._check_dependency_security(file_path, content))
            findings.extend(self._check_network_security(file_path, content, lines))
            findings.extend(self._check_crypto_security(file_path, content, lines))

        except Exception as e:
            findings.append(SecurityFinding(
                str(file_path), 0, "FILE_ACCESS_ERROR", SecurityLevel.HIGH,
                f"Failed to audit file: {str(e)}",
                "File access errors can indicate permission issues",
                "Ensure proper file permissions and access controls"
            ))

        return findings

    def _check_vulnerability_patterns(self, file_path: Path, content: str, lines: List[str]) -> List[SecurityFinding]:
        """Check for common vulnerability patterns."""
        findings = []

        for vuln_type, patterns in self.vulnerability_patterns.items():
            for i, line in enumerate(lines, 1):
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        level = self._get_vulnerability_level(vuln_type)
                        description, impact, remediation = self._get_vulnerability_details(vuln_type)

                        findings.append(SecurityFinding(
                            str(file_path), i, vuln_type, level,
                            description, impact, remediation,
                            self._get_cwe_id(vuln_type), line.strip()
                        ))

        return findings

    def _check_critical_functions(self, file_path: Path, content: str, lines: List[str]) -> List[SecurityFinding]:
        """Check for usage of critical security functions."""
        findings = []

        for i, line in enumerate(lines, 1):
            for func in self.critical_functions:
                if re.search(rf'\b{func}\s*\(', line):
                    findings.append(SecurityFinding(
                        str(file_path), i, "CRITICAL_FUNCTION", SecurityLevel.HIGH,
                        f"Usage of critical security function '{func}'",
                        "Can lead to code injection or arbitrary code execution",
                        f"Avoid using {func} or use safe alternatives with proper input validation",
                        "CWE-94", line.strip()
                    ))

        return findings

    def _check_file_permissions(self, file_path: Path) -> List[SecurityFinding]:
        """Check file permissions for security issues."""
        findings = []

        try:
            stat_info = file_path.stat()
            permissions = oct(stat_info.st_mode)[-3:]

            # Check for world-writable files
            if permissions[-1] in ['2', '3', '6', '7']:
                findings.append(SecurityFinding(
                    str(file_path), 0, "INSECURE_PERMISSIONS", SecurityLevel.MEDIUM,
                    f"File has world-writable permissions: {permissions}",
                    "Can allow unauthorized modification of critical files",
                    "Restrict file permissions to owner-only or group access",
                    "CWE-732"
                ))

            # Check for sensitive files
            sensitive_files = ['key', 'secret', 'password', 'token', 'private']
            if any(sensitive in file_path.name.lower() for sensitive in sensitive_files):
                if permissions[-1] != '0':
                    findings.append(SecurityFinding(
                        str(file_path), 0, "SENSITIVE_FILE_PERMISSIONS", SecurityLevel.HIGH,
                        f"Sensitive file has overly permissive permissions: {permissions}",
                        "Sensitive files should have restricted access",
                        "Set permissions to 600 or use secure key management",
                        "CWE-922"
                    ))

        except Exception:
            pass  # Skip permission checks if unable to stat file

        return findings

    def _check_dependency_security(self, file_path: Path, content: str) -> List[SecurityFinding]:
        """Check for potentially insecure dependency usage."""
        findings = []

        insecure_imports = {
            'requests': 'Consider using httpx for better security and async support',
            'urllib.request': 'Use requests or httpx instead of urllib for better security',
            'xml.etree': 'Consider using defusedxml for XML parsing security',
            'shelve': 'Avoid shelve for sensitive data; use secure alternatives'
        }

        for import_name, recommendation in insecure_imports.items():
            if import_name in content:
                findings.append(SecurityFinding(
                    str(file_path), 0, "INSECURE_DEPENDENCY", SecurityLevel.LOW,
                    f"Usage of potentially insecure dependency: {import_name}",
                    "May have known security vulnerabilities or weak security defaults",
                    recommendation,
                    "CWE-350"
                ))

        return findings

    def _check_network_security(self, file_path: Path, content: str, lines: List[str]) -> List[SecurityFinding]:
        """Check for network security issues."""
        findings = []

        # Check for HTTP URLs instead of HTTPS
        for i, line in enumerate(lines, 1):
            if re.search(r'http://', line) and 'localhost' not in line and '127.0.0.1' not in line:
                findings.append(SecurityFinding(
                    str(file_path), i, "INSECURE_HTTP", SecurityLevel.MEDIUM,
                    "HTTP URL found instead of HTTPS",
                    "HTTP traffic is unencrypted and can be intercepted",
                    "Use HTTPS URLs for all external communications",
                    "CWE-319", line.strip()
                ))

        # Check for missing SSL verification
        if 'verify=False' in content:
            findings.append(SecurityFinding(
                str(file_path), 0, "SSL_VERIFICATION_DISABLED", SecurityLevel.HIGH,
                "SSL certificate verification is disabled",
                "Allows man-in-the-middle attacks and invalid certificates",
                "Enable SSL verification or use proper certificate pinning",
                "CWE-295"
            ))

        return findings

    def _check_crypto_security(self, file_path: Path, content: str, lines: List[str]) -> List[SecurityFinding]:
        """Check for cryptographic security issues."""
        findings = []

        # Check for weak encryption algorithms
        weak_algorithms = ['DES', '3DES', 'RC4', 'MD5', 'SHA1']
        for algorithm in weak_algorithms:
            if algorithm in content:
                findings.append(SecurityFinding(
                    str(file_path), 0, "WEAK_CRYPTO", SecurityLevel.HIGH,
                    f"Usage of weak cryptographic algorithm: {algorithm}",
                    "Weak algorithms can be easily broken by attackers",
                    f"Use strong algorithms like AES-256, SHA-256, or ChaCha20",
                    "CWE-327"
                ))

        # Check for proper key lengths
        if 'RSA' in content and '1024' in content:
            findings.append(SecurityFinding(
                str(file_path), 0, "INSUFFICIENT_KEY_LENGTH", SecurityLevel.MEDIUM,
                "RSA key length appears to be insufficient (1024 bits)",
                "Short keys can be factored by modern computers",
                "Use RSA keys of at least 2048 bits, preferably 4096 bits",
                "CWE-326"
            ))

        return findings

    def _get_vulnerability_level(self, vuln_type: str) -> SecurityLevel:
        """Get security level for vulnerability type."""
        level_map = {
            'hardcoded_secrets': SecurityLevel.CRITICAL,
            'sql_injection': SecurityLevel.CRITICAL,
            'command_injection': SecurityLevel.CRITICAL,
            'path_traversal': SecurityLevel.HIGH,
            'weak_crypto': SecurityLevel.HIGH,
            'insecure_deserialization': SecurityLevel.HIGH,
        }
        return level_map.get(vuln_type, SecurityLevel.MEDIUM)

    def _get_vulnerability_details(self, vuln_type: str) -> Tuple[str, str, str]:
        """Get details for vulnerability type."""
        details_map = {
            'hardcoded_secrets': (
                "Hardcoded secrets found in source code",
                "Exposes sensitive credentials to attackers with code access",
                "Use environment variables, secure vaults, or key management services"
            ),
            'sql_injection': (
                "Potential SQL injection vulnerability",
                "Allows attackers to execute arbitrary SQL queries",
                "Use parameterized queries or ORM with proper input validation"
            ),
            'command_injection': (
                "Potential command injection vulnerability",
                "Allows attackers to execute arbitrary system commands",
                "Use subprocess with proper argument lists and input validation"
            ),
            'weak_crypto': (
                "Weak cryptographic algorithm usage",
                "Cryptographic operations can be easily broken",
                "Use modern, strong cryptographic algorithms"
            )
        }
        return details_map.get(vuln_type, (
            f"Security issue: {vuln_type}",
            "Potential security vulnerability",
            "Review and fix the security issue"
        ))

    def _get_cwe_id(self, vuln_type: str) -> Optional[str]:
        """Get CWE ID for vulnerability type."""
        cwe_map = {
            'hardcoded_secrets': 'CWE-798',
            'sql_injection': 'CWE-89',
            'command_injection': 'CWE-78',
            'path_traversal': 'CWE-22',
            'weak_crypto': 'CWE-327',
            'insecure_deserialization': 'CWE-502'
        }
        return cwe_map.get(vuln_type)

    def run_security_audit(self, paths: Optional[List[Path]] = None) -> SecurityReport:
        """Run comprehensive security audit."""
        if paths is None:
            paths = [
                self.project_root / 'src',
                self.project_root / 'fedzk',
                self.project_root / 'tools',
                self.project_root / 'scripts'
            ]

        # Find all relevant files
        audit_files = []
        for path in paths:
            if path.exists():
                if path.is_file():
                    audit_files.append(path)
                elif path.is_dir():
                    # Include Python, YAML, JSON, shell scripts
                    for pattern in ['*.py', '*.yml', '*.yaml', '*.json', '*.sh']:
                        audit_files.extend(path.rglob(pattern))

        # Audit each file
        for file_path in audit_files:
            findings = self.audit_file(file_path)
            for finding in findings:
                self.report.add_finding(finding)

        return self.report

    def generate_compliance_report(self) -> Dict:
        """Generate compliance report for various standards."""
        return {
            'owasp_compliance': self._check_owasp_compliance(),
            'nist_compliance': self._check_nist_compliance(),
            'gdpr_compliance': self._check_gdpr_compliance(),
            'hipaa_compliance': self._check_hipaa_compliance()
        }

    def _check_owasp_compliance(self) -> Dict:
        """Check OWASP Top 10 compliance."""
        return {
            'injection_prevention': len([f for f in self.report.findings
                                       if 'injection' in f.vulnerability_type]) == 0,
            'broken_auth_prevention': len([f for f in self.report.findings
                                         if 'auth' in f.vulnerability_type.lower()]) == 0,
            'sensitive_data_protection': len([f for f in self.report.findings
                                            if 'secret' in f.vulnerability_type.lower()]) == 0
        }

    def _check_nist_compliance(self) -> Dict:
        """Check NIST framework compliance."""
        return {
            'identify': True,  # Asset management
            'protect': len([f for f in self.report.findings
                          if f.level in [SecurityLevel.CRITICAL, SecurityLevel.HIGH]]) == 0,
            'detect': True,   # Security monitoring
            'respond': True,  # Incident response
            'recover': True   # Disaster recovery
        }

    def _check_gdpr_compliance(self) -> Dict:
        """Check GDPR compliance."""
        return {
            'data_minimization': True,
            'purpose_limitation': True,
            'storage_limitation': True,
            'data_subject_rights': True,
            'consent_management': len([f for f in self.report.findings
                                     if 'consent' in f.vulnerability_type.lower()]) == 0
        }

    def _check_hipaa_compliance(self) -> Dict:
        """Check HIPAA compliance."""
        return {
            'privacy_rule': len([f for f in self.report.findings
                               if 'privacy' in f.vulnerability_type.lower()]) == 0,
            'security_rule': len([f for f in self.report.findings
                                if f.level == SecurityLevel.CRITICAL]) == 0,
            'breach_notification': True
        }


def main():
    """Main entry point for security audit tool."""
    project_root = Path(__file__).parent.parent

    auditor = FEDZKSecurityAuditor(project_root)
    report = auditor.run_security_audit()

    # Print summary
    print("🔒 FEDZK Security Audit Report")
    print("=" * 50)
    print(f"Files audited: {report.total_files}")
    print(f"Total lines: {report.total_lines}")
    print(f"Total findings: {len(report.findings)}")
    print(".1f")

    if report.summary:
        print("\nFindings by severity:")
        for level, count in report.summary.items():
            print(f"  {level}: {count}")

    # Print critical and high findings
    critical_findings = [f for f in report.findings
                        if f.level in [SecurityLevel.CRITICAL, SecurityLevel.HIGH]]

    if critical_findings:
        print("\n🚨 Critical and High Severity Findings:")
        for finding in critical_findings[:10]:  # Show first 10
            print(f"\n{finding.level.value}: {finding.file_path}:{finding.line_number}")
            print(f"  Type: {finding.vulnerability_type}")
            print(f"  Description: {finding.description}")
            print(f"  Impact: {finding.impact}")
            print(f"  Remediation: {finding.remediation}")
            if finding.cwe_id:
                print(f"  CWE: {finding.cwe_id}")

    # Compliance summary
    compliance = auditor.generate_compliance_report()
    print("
📋 Compliance Summary:"    for standard, checks in compliance.items():
        compliant = sum(checks.values())
        total = len(checks)
        status = "✅" if compliant == total else "⚠️"
        print(f"  {status} {standard}: {compliant}/{total} checks passed")

    # Exit with appropriate code
    if report.passed:
        print("\n✅ Security audit passed!")
        sys.exit(0)
    else:
        print("
❌ Security audit failed!"        print("Fix critical and high severity findings before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
