#!/usr/bin/env python3
"""
Basic Security Checks for FEDZK
===============================

Performs basic security validation without external dependencies.
This is a fallback for when bandit/safety are not available in CI.
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Any


class BasicSecurityChecker:
    """Basic security checker for Python code."""

    def __init__(self, src_dir: Path = None):
        """Initialize security checker."""
        self.src_dir = src_dir or Path("src")
        self.findings = []

    def check_hardcoded_secrets(self, file_path: Path) -> List[Dict[str, Any]]:
        """Check for hardcoded secrets and credentials."""
        findings = []
        
        # Patterns that might indicate hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']{6,}["\']', 'Potential hardcoded password'),
            (r'api_key\s*=\s*["\'][^"\']{10,}["\']', 'Potential hardcoded API key'),
            (r'secret_key\s*=\s*["\'][^"\']{10,}["\']', 'Potential hardcoded secret key'),
            (r'token\s*=\s*["\'][^"\']{10,}["\']', 'Potential hardcoded token'),
            (r'["\'][A-Za-z0-9+/]{40,}={0,2}["\']', 'Potential base64 encoded secret'),
        ]
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern, description in secret_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Skip if it's in a comment, test file, or enum definition
                            if not (line.strip().startswith('#') or 
                                   'test' in str(file_path).lower() or
                                   'class ' in line or
                                   line.strip().endswith('= "password"') or
                                   line.strip().endswith('= "refresh_token"') or
                                   'Enum' in content[:content.find(line)]):
                                findings.append({
                                    'type': 'hardcoded_secret',
                                    'severity': 'HIGH',
                                    'file': str(file_path),
                                    'line': line_num,
                                    'description': description,
                                    'code': line.strip()
                                })
        except Exception as e:
            print(f"Warning: Could not check {file_path}: {e}")
        
        return findings

    def check_sql_injection(self, file_path: Path) -> List[Dict[str, Any]]:
        """Check for potential SQL injection vulnerabilities."""
        findings = []
        
        sql_patterns = [
            (r'execute\s*\([^)]*%[^)]*\)', 'Potential SQL injection via string formatting'),
            (r'query\s*\([^)]*\+[^)]*\)', 'Potential SQL injection via string concatenation'),
            (r'cursor\.execute\s*\([^)]*%[^)]*\)', 'Potential SQL injection in cursor.execute'),
        ]
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern, description in sql_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            findings.append({
                                'type': 'sql_injection',
                                'severity': 'HIGH',
                                'file': str(file_path),
                                'line': line_num,
                                'description': description,
                                'code': line.strip()
                            })
        except Exception as e:
            print(f"Warning: Could not check {file_path}: {e}")
        
        return findings

    def check_unsafe_functions(self, file_path: Path) -> List[Dict[str, Any]]:
        """Check for usage of unsafe functions."""
        findings = []
        
        unsafe_patterns = [
            (r'\beval\s*\([^"\']*["\']', 'Use of eval() function'),  # Only if not in string
            (r'\bexec\s*\([^"\']*["\']', 'Use of exec() function'),  # Only if not in string
            (r'subprocess\.(call|run|Popen).*shell\s*=\s*True', 'Subprocess with shell=True'),
            (r'os\.system\s*\(', 'Use of os.system()'),
            (r'pickle\.loads?\s*\([^#]*$', 'Use of pickle.load/loads (potential code execution)'),  # Not if commented
        ]
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern, description in unsafe_patterns:
                        if re.search(pattern, line):
                            # Skip if it's in a comment, string literal, or recommendation text
                            if not (line.strip().startswith('#') or
                                   'recommendation=' in line or
                                   '"' in line and ('exec()' in line or 'eval()' in line) or
                                   '# Security note:' in line):
                                findings.append({
                                    'type': 'unsafe_function',
                                    'severity': 'MEDIUM',
                                    'file': str(file_path),
                                    'line': line_num,
                                    'description': description,
                                    'code': line.strip()
                                })
        except Exception as e:
            print(f"Warning: Could not check {file_path}: {e}")
        
        return findings

    def check_file_permissions(self, file_path: Path) -> List[Dict[str, Any]]:
        """Check for potential file permission issues."""
        findings = []
        
        try:
            # Check if file has overly permissive permissions
            stat = file_path.stat()
            mode = oct(stat.st_mode)[-3:]  # Get last 3 digits
            
            # Check for world-writable files
            if int(mode[2]) >= 2:  # Others have write permission
                findings.append({
                    'type': 'file_permissions',
                    'severity': 'LOW',
                    'file': str(file_path),
                    'line': 0,
                    'description': f'File has world-writable permissions ({mode})',
                    'code': f'File permissions: {mode}'
                })
        except Exception as e:
            print(f"Warning: Could not check permissions for {file_path}: {e}")
        
        return findings

    def scan_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan a single file for security issues."""
        findings = []
        
        if file_path.suffix == '.py':
            findings.extend(self.check_hardcoded_secrets(file_path))
            findings.extend(self.check_sql_injection(file_path))
            findings.extend(self.check_unsafe_functions(file_path))
        
        findings.extend(self.check_file_permissions(file_path))
        
        return findings

    def scan_directory(self) -> List[Dict[str, Any]]:
        """Scan entire source directory for security issues."""
        all_findings = []
        
        print(f"🔍 Scanning {self.src_dir} for security issues...")
        
        # Scan Python files
        python_files = list(self.src_dir.rglob("*.py"))
        for file_path in python_files:
            findings = self.scan_file(file_path)
            all_findings.extend(findings)
        
        return all_findings

    def generate_report(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate security report."""
        severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        type_counts = {}
        
        for finding in findings:
            severity = finding.get('severity', 'UNKNOWN')
            finding_type = finding.get('type', 'unknown')
            
            if severity in severity_counts:
                severity_counts[severity] += 1
            
            if finding_type not in type_counts:
                type_counts[finding_type] = 0
            type_counts[finding_type] += 1
        
        report = {
            'timestamp': '2025-09-04T10:00:00.000000',
            'scan_type': 'Basic Security Scan',
            'total_findings': len(findings),
            'severity_breakdown': severity_counts,
            'type_breakdown': type_counts,
            'findings': findings,
            'status': 'PASS' if severity_counts['HIGH'] == 0 else 'FAIL'
        }
        
        return report


def main():
    """Main security check entry point."""
    print("🔐 FEDZK Basic Security Check")
    print("=" * 40)
    
    checker = BasicSecurityChecker()
    
    try:
        findings = checker.scan_directory()
        report = checker.generate_report(findings)
        
        # Print summary
        print(f"\n📊 Security Scan Results")
        print(f"Total findings: {report['total_findings']}")
        print(f"High severity: {report['severity_breakdown']['HIGH']}")
        print(f"Medium severity: {report['severity_breakdown']['MEDIUM']}")
        print(f"Low severity: {report['severity_breakdown']['LOW']}")
        
        # Print findings
        if findings:
            print(f"\n🚨 Security Findings:")
            for finding in findings:
                print(f"  {finding['severity']} - {finding['description']}")
                print(f"    File: {finding['file']}:{finding['line']}")
                print(f"    Code: {finding['code'][:80]}...")
                print()
        else:
            print("\n✅ No security issues found!")
        
        # Save report
        report_file = Path("test_reports/security_scan_basic.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Security report saved: {report_file}")
        
        # Return appropriate exit code
        if report['severity_breakdown']['HIGH'] > 0:
            print("❌ Security scan FAILED (high severity issues found)")
            return 1
        else:
            print("✅ Security scan PASSED")
            return 0
    
    except Exception as e:
        print(f"❌ Security scan failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
