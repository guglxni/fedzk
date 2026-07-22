"""
Security Auditor for FEDZK

This module provides comprehensive security auditing capabilities including
vulnerability scanning, security best practices validation, and risk assessment.
"""

import os
import re
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import ast
import inspect

logger = logging.getLogger(__name__)


class SecuritySeverity(Enum):
    """Security issue severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class VulnerabilityType(Enum):
    """Types of security vulnerabilities"""
    CRYPTOGRAPHIC_WEAKNESS = "cryptographic_weakness"
    INPUT_VALIDATION = "input_validation"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    AUTHORIZATION_FAILURE = "authorization_failure"
    DATA_LEAKAGE = "data_leakage"
    SECRETS_EXPOSURE = "secrets_exposure"
    DEPENDENCY_VULNERABILITY = "dependency_vulnerability"
    CODE_INJECTION = "code_injection"
    DENIAL_OF_SERVICE = "denial_of_service"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    RACE_CONDITION = "race_condition"


@dataclass
class SecurityFinding:
    """Represents a security finding from an audit"""
    id: str
    title: str
    description: str
    severity: SecuritySeverity
    vulnerability_type: VulnerabilityType
    file_path: str
    line_number: Optional[int]
    code_snippet: str
    recommendation: str
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    remediation_status: str = "open"
    discovered_at: datetime = field(default_factory=datetime.now)
    mitigated_at: Optional[datetime] = None


@dataclass
class AuditReport:
    """Comprehensive audit report"""
    audit_id: str
    target_system: str
    audit_start_time: datetime
    audit_end_time: Optional[datetime] = None
    total_files_scanned: int = 0
    findings: List[SecurityFinding] = field(default_factory=list)
    risk_score: float = 0.0
    compliance_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecurityAuditor:
    """
    Comprehensive security auditor for FEDZK system

    Performs static analysis, vulnerability scanning, and security best practices validation.
    """

    def __init__(self, target_directory: str = None):
        self.target_directory = Path(target_directory or Path(__file__).parent.parent.parent)
        self._patterns = self._initialize_patterns()
        self._findings: List[SecurityFinding] = []
        self._scanned_files: Set[str] = set()

    def _initialize_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize security vulnerability patterns"""
        return {
            'hardcoded_secrets': [
                {
                    'pattern': r'(?i)(password|secret|key|token)\s*[:=]\s*["\'][^"\']+["\']',
                    'severity': SecuritySeverity.HIGH,
                    'type': VulnerabilityType.SECRETS_EXPOSURE,
                    'title': 'Hardcoded Secret Detected',
                    'recommendation': 'Use environment variables or secure key management'
                }
            ],
            'sql_injection': [
                {
                    'pattern': r'(?i)(select|insert|update|delete).*?\+.*?(request\.|input)',
                    'severity': SecuritySeverity.CRITICAL,
                    'type': VulnerabilityType.CODE_INJECTION,
                    'title': 'Potential SQL Injection',
                    'recommendation': 'Use parameterized queries or ORM'
                }
            ],
            'weak_crypto': [
                {
                    'pattern': r'(?i)(md5|sha1)\(',
                    'severity': SecuritySeverity.HIGH,
                    'type': VulnerabilityType.CRYPTOGRAPHIC_WEAKNESS,
                    'title': 'Weak Cryptographic Hash',
                    'recommendation': 'Use SHA-256 or stronger hashing algorithms'
                }
            ],
            'unsafe_deserialization': [
                {
                    'pattern': r'(?i)(pickle\.load|yaml\.unsafe_load)',
                    'severity': SecuritySeverity.HIGH,
                    'type': VulnerabilityType.INSECURE_DESERIALIZATION,
                    'title': 'Unsafe Deserialization',
                    'recommendation': 'Use safe deserialization methods'
                }
            ],
            'missing_input_validation': [
                {
                    'pattern': r'def\s+\w+\([^)]*\):\s*$',
                    'severity': SecuritySeverity.MEDIUM,
                    'type': VulnerabilityType.INPUT_VALIDATION,
                    'title': 'Function Without Input Validation',
                    'recommendation': 'Add input validation and sanitization'
                }
            ]
        }

    def perform_comprehensive_audit(self) -> AuditReport:
        """
        Perform comprehensive security audit of the codebase

        Returns:
            AuditReport: Detailed audit results
        """
        audit_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report = AuditReport(
            audit_id=audit_id,
            target_system="FEDZK",
            audit_start_time=datetime.now(),
            metadata={
                'auditor_version': '1.0.0',
                'scan_type': 'comprehensive',
                'target_directory': str(self.target_directory)
            }
        )

        try:
            # Scan Python files
            self._scan_python_files()

            # Scan configuration files
            self._scan_config_files()

            # Scan dependencies
            self._scan_dependencies()

            # Perform cryptographic analysis
            self._analyze_cryptographic_usage()

            # Generate risk assessment
            report.findings = self._findings.copy()
            report.total_files_scanned = len(self._scanned_files)
            report.risk_score = self._calculate_risk_score()
            report.compliance_score = self._calculate_compliance_score()
            report.recommendations = self._generate_recommendations()

        except Exception as e:
            logger.error(f"Audit failed: {e}")
            report.metadata['error'] = str(e)

        report.audit_end_time = datetime.now()
        return report

    def _scan_python_files(self):
        """Scan Python files for security vulnerabilities"""
        python_files = list(self.target_directory.rglob("*.py"))

        for file_path in python_files:
            if str(file_path).endswith('__pycache__'):
                continue

            try:
                self._scan_single_file(file_path)
                self._scanned_files.add(str(file_path))
            except Exception as e:
                logger.warning(f"Failed to scan {file_path}: {e}")

    def _scan_single_file(self, file_path: Path):
        """Scan a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            lines = content.split('\n')

            # Check each pattern
            for category, patterns in self._patterns.items():
                for pattern_config in patterns:
                    matches = re.finditer(pattern_config['pattern'], content, re.MULTILINE)
                    for match in matches:
                        line_number = content[:match.start()].count('\n') + 1
                        code_snippet = self._extract_code_snippet(lines, line_number - 1, 3)

                        finding = SecurityFinding(
                            id=f"{pattern_config['type'].value}_{hashlib.md5(str(match.start()).encode()).hexdigest()[:8]}",
                            title=pattern_config['title'],
                            description=f"Detected {pattern_config['title']} in {file_path.name}",
                            severity=pattern_config['severity'],
                            vulnerability_type=pattern_config['type'],
                            file_path=str(file_path),
                            line_number=line_number,
                            code_snippet=code_snippet,
                            recommendation=pattern_config['recommendation'],
                            evidence={
                                'pattern_matched': match.group(),
                                'category': category
                            }
                        )
                        self._findings.append(finding)

            # Additional AST-based analysis
            self._analyze_ast(file_path, content)

        except Exception as e:
            logger.error(f"Error scanning {file_path}: {e}")

    def _analyze_ast(self, file_path: Path, content: str):
        """Analyze Python AST for security issues"""
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                # Check for dangerous imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ['pickle', 'subprocess', 'os.system']:
                            self._create_dangerous_import_finding(file_path, alias.name, node.lineno)

                # Check for exec/eval usage
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ['exec', 'eval']:
                        self._create_code_execution_finding(file_path, node.func.id, node.lineno)

        except SyntaxError:
            logger.warning(f"Syntax error in {file_path}, skipping AST analysis")

    def _create_dangerous_import_finding(self, file_path: Path, module: str, line_number: int):
        """Create finding for dangerous import"""
        finding = SecurityFinding(
            id=f"dangerous_import_{hashlib.md5(f'{file_path}_{line_number}'.encode()).hexdigest()[:8]}",
            title=f"Dangerous Import: {module}",
            description=f"Import of potentially dangerous module '{module}'",
            severity=SecuritySeverity.MEDIUM,
            vulnerability_type=VulnerabilityType.CODE_INJECTION,
            file_path=str(file_path),
            line_number=line_number,
            code_snippet=f"import {module}",
            recommendation=f"Review usage of {module} and consider safer alternatives",
            evidence={'module': module}
        )
        self._findings.append(finding)

    def _create_code_execution_finding(self, file_path: Path, function: str, line_number: int):
        """Create finding for code execution"""
        finding = SecurityFinding(
            id=f"code_execution_{hashlib.md5(f'{file_path}_{line_number}'.encode()).hexdigest()[:8]}",
            title=f"Dangerous Code Execution: {function}",
            description=f"Use of {function}() which can execute arbitrary code",
            severity=SecuritySeverity.CRITICAL,
            vulnerability_type=VulnerabilityType.CODE_INJECTION,
            file_path=str(file_path),
            line_number=line_number,
            code_snippet=f"{function}(...)",
            recommendation="Avoid using exec() or eval(). Use safer alternatives.",
            evidence={'function': function}
        )
        self._findings.append(finding)

    def _scan_config_files(self):
        """Scan configuration files for security issues"""
        config_patterns = ['*.json', '*.yaml', '*.yml', '*.toml', '*.ini', '*.cfg']

        for pattern in config_patterns:
            config_files = list(self.target_directory.rglob(pattern))

            for file_path in config_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Check for hardcoded secrets in config files
                    secret_patterns = [
                        r'["\']password["\']\s*:\s*["\'][^"\']+["\']',
                        r'["\']secret["\']\s*:\s*["\'][^"\']+["\']',
                        r'["\']key["\']\s*:\s*["\'][^"\']+["\']',
                        r'["\']token["\']\s*:\s*["\'][^"\']+["\']'
                    ]

                    for pattern in secret_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            finding = SecurityFinding(
                                id=f"config_secret_{hashlib.md5(f'{file_path}_{match.start()}'.encode()).hexdigest()[:8]}",
                                title="Hardcoded Secret in Configuration",
                                description=f"Potential hardcoded secret found in {file_path.name}",
                                severity=SecuritySeverity.HIGH,
                                vulnerability_type=VulnerabilityType.SECRETS_EXPOSURE,
                                file_path=str(file_path),
                                line_number=content[:match.start()].count('\n') + 1,
                                code_snippet=match.group(),
                                recommendation="Use environment variables or secure configuration management",
                                evidence={'config_file': True}
                            )
                            self._findings.append(finding)

                except Exception as e:
                    logger.warning(f"Failed to scan config file {file_path}: {e}")

    def _scan_dependencies(self):
        """Scan project dependencies for known vulnerabilities"""
        requirements_files = ['requirements.txt', 'pyproject.toml', 'setup.py', 'Pipfile']

        for req_file in requirements_files:
            req_path = self.target_directory / req_file
            if req_path.exists():
                try:
                    with open(req_path, 'r') as f:
                        content = f.read()

                    # Simple dependency analysis (in real implementation, use tools like safety)
                    vulnerable_deps = {
                        'insecure-package': 'Known vulnerability',
                        'old-crypto': 'Outdated cryptography library'
                    }

                    for dep, issue in vulnerable_deps.items():
                        if dep in content.lower():
                            finding = SecurityFinding(
                                id=f"vulnerable_dep_{hashlib.md5(f'{req_file}_{dep}'.encode()).hexdigest()[:8]}",
                                title=f"Vulnerable Dependency: {dep}",
                                description=f"Potentially vulnerable dependency '{dep}' found in {req_file}",
                                severity=SecuritySeverity.HIGH,
                                vulnerability_type=VulnerabilityType.DEPENDENCY_VULNERABILITY,
                                file_path=str(req_path),
                                line_number=None,
                                code_snippet=dep,
                                recommendation=f"Update {dep} to latest secure version or find alternative",
                                evidence={'dependency_file': req_file, 'issue': issue}
                            )
                            self._findings.append(finding)

                except Exception as e:
                    logger.warning(f"Failed to scan dependencies in {req_path}: {e}")

    def _analyze_cryptographic_usage(self):
        """Analyze cryptographic implementations"""
        crypto_files = list(self.target_directory.rglob("*crypto*.py")) + \
                      list(self.target_directory.rglob("*zk*.py"))

        for file_path in crypto_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for proper key sizes
                if 'AES' in content and '128' not in content and '256' not in content:
                    finding = SecurityFinding(
                        id=f"weak_key_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}",
                        title="Potential Weak Key Size",
                        description="AES encryption without explicit key size specification",
                        severity=SecuritySeverity.MEDIUM,
                        vulnerability_type=VulnerabilityType.CRYPTOGRAPHIC_WEAKNESS,
                        file_path=str(file_path),
                        line_number=None,
                        code_snippet="AES encryption",
                        recommendation="Specify explicit key size (AES-256 recommended)",
                        evidence={'algorithm': 'AES'}
                    )
                    self._findings.append(finding)

            except Exception as e:
                logger.warning(f"Failed to analyze crypto in {file_path}: {e}")

    def _extract_code_snippet(self, lines: List[str], start_line: int, context_lines: int = 3) -> str:
        """Extract code snippet with context"""
        start = max(0, start_line - context_lines)
        end = min(len(lines), start_line + context_lines + 1)
        snippet_lines = []

        for i in range(start, end):
            marker = ">>> " if i == start_line else "    "
            snippet_lines.append("2")

        return '\n'.join(snippet_lines)

    def _calculate_risk_score(self) -> float:
        """Calculate overall risk score based on findings"""
        if not self._findings:
            return 0.0

        severity_weights = {
            SecuritySeverity.CRITICAL: 10,
            SecuritySeverity.HIGH: 7,
            SecuritySeverity.MEDIUM: 4,
            SecuritySeverity.LOW: 2,
            SecuritySeverity.INFO: 1
        }

        total_weight = sum(severity_weights[finding.severity] for finding in self._findings)
        max_possible_weight = len(self._findings) * 10  # Max weight per finding

        return min(100.0, (total_weight / max_possible_weight) * 100) if max_possible_weight > 0 else 0.0

    def _calculate_compliance_score(self) -> float:
        """Calculate compliance score (inverse of risk score with adjustments)"""
        risk_score = self._calculate_risk_score()

        # Adjust for compliance factors
        compliance_bonus = 0

        # Bonus for having security features
        security_features = ['authentication', 'encryption', 'validation', 'audit']
        for feature in security_features:
            if any(feature in str(f.evidence) for f in self._findings):
                compliance_bonus += 5

        return max(0.0, min(100.0, 100.0 - risk_score + compliance_bonus))

    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings"""
        recommendations = []

        severity_counts = {}
        for finding in self._findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1

        if severity_counts.get(SecuritySeverity.CRITICAL, 0) > 0:
            recommendations.append("CRITICAL: Address critical security findings immediately")
        if severity_counts.get(SecuritySeverity.HIGH, 0) > 0:
            recommendations.append("HIGH: Implement fixes for high-severity vulnerabilities")
        if severity_counts.get(SecuritySeverity.MEDIUM, 0) > 5:
            recommendations.append("MEDIUM: Review and remediate medium-severity issues")

        # Specific recommendations
        if any(f.vulnerability_type == VulnerabilityType.SECRETS_EXPOSURE for f in self._findings):
            recommendations.append("Implement secure secrets management system")
        if any(f.vulnerability_type == VulnerabilityType.CRYPTOGRAPHIC_WEAKNESS for f in self._findings):
            recommendations.append("Upgrade to modern cryptographic standards")
        if any(f.vulnerability_type == VulnerabilityType.CODE_INJECTION for f in self._findings):
            recommendations.append("Implement comprehensive input validation and sanitization")

        return recommendations

    def export_report(self, report: AuditReport, output_format: str = 'json') -> str:
        """Export audit report in specified format"""
        if output_format == 'json':
            return json.dumps(self._report_to_dict(report), indent=2, default=str)
        elif output_format == 'html':
            return self._generate_html_report(report)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def _report_to_dict(self, report: AuditReport) -> Dict[str, Any]:
        """Convert audit report to dictionary"""
        return {
            'audit_id': report.audit_id,
            'target_system': report.target_system,
            'audit_start_time': report.audit_start_time.isoformat(),
            'audit_end_time': report.audit_end_time.isoformat() if report.audit_end_time else None,
            'total_files_scanned': report.total_files_scanned,
            'findings_count': len(report.findings),
            'risk_score': report.risk_score,
            'compliance_score': report.compliance_score,
            'findings': [
                {
                    'id': f.id,
                    'title': f.title,
                    'description': f.description,
                    'severity': f.severity.value,
                    'vulnerability_type': f.vulnerability_type.value,
                    'file_path': f.file_path,
                    'line_number': f.line_number,
                    'code_snippet': f.code_snippet,
                    'recommendation': f.recommendation,
                    'cwe_id': f.cwe_id,
                    'evidence': f.evidence,
                    'remediation_status': f.remediation_status,
                    'discovered_at': f.discovered_at.isoformat()
                }
                for f in report.findings
            ],
            'recommendations': report.recommendations,
            'metadata': report.metadata
        }

    def _generate_html_report(self, report: AuditReport) -> str:
        """Generate HTML audit report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FEDZK Security Audit Report - {report.audit_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .severity-critical {{ color: #dc3545; }}
                .severity-high {{ color: #fd7e14; }}
                .severity-medium {{ color: #ffc107; }}
                .severity-low {{ color: #28a745; }}
                .finding {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
                .code-snippet {{ background: #f8f9fa; padding: 10px; border-radius: 3px; font-family: monospace; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>FEDZK Security Audit Report</h1>
                <p><strong>Audit ID:</strong> {report.audit_id}</p>
                <p><strong>Target System:</strong> {report.target_system}</p>
                <p><strong>Audit Date:</strong> {report.audit_start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Files Scanned:</strong> {report.total_files_scanned}</p>
                <p><strong>Risk Score:</strong> {report.risk_score:.1f}/100</p>
                <p><strong>Compliance Score:</strong> {report.compliance_score:.1f}/100</p>
            </div>

            <h2>Findings Summary</h2>
            <p>Total findings: {len(report.findings)}</p>

            <h2>Detailed Findings</h2>
        """

        for finding in report.findings:
            html += f"""
            <div class="finding">
                <h3 class="severity-{finding.severity.value.lower()}">{finding.title}</h3>
                <p><strong>Severity:</strong> {finding.severity.value}</p>
                <p><strong>Type:</strong> {finding.vulnerability_type.value}</p>
                <p><strong>File:</strong> {finding.file_path}</p>
                {f'<p><strong>Line:</strong> {finding.line_number}</p>' if finding.line_number else ''}
                <p><strong>Description:</strong> {finding.description}</p>
                <p><strong>Recommendation:</strong> {finding.recommendation}</p>
                {f'<div class="code-snippet"><pre>{finding.code_snippet}</pre></div>' if finding.code_snippet else ''}
            </div>
            """

        html += """
            <h2>Recommendations</h2>
            <ul>
        """

        for rec in report.recommendations:
            html += f"<li>{rec}</li>"

        html += """
            </ul>
        </body>
        </html>
        """

        return html
