"""
Code Review Framework for FEDZK

This module provides automated code review capabilities, security best practices
validation, and code quality assessment tools.
"""

import ast
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import inspect
import hashlib

logger = logging.getLogger(__name__)


class ReviewSeverity(Enum):
    """Code review finding severity"""
    BLOCKER = "BLOCKER"
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    INFO = "INFO"


class ReviewCategory(Enum):
    """Code review categories"""
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    MAINTAINABILITY = "MAINTAINABILITY"
    RELIABILITY = "RELIABILITY"
    COMPLIANCE = "COMPLIANCE"
    BEST_PRACTICES = "BEST_PRACTICES"


@dataclass
class CodeReviewFinding:
    """Represents a finding from code review"""
    id: str
    title: str
    description: str
    severity: ReviewSeverity
    category: ReviewCategory
    file_path: str
    line_number: Optional[int]
    code_snippet: str
    recommendation: str
    rule_id: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    context_lines: List[str] = field(default_factory=list)


@dataclass
class CodeReviewReport:
    """Comprehensive code review report"""
    review_id: str
    target_directory: str
    review_start_time: str
    review_end_time: Optional[str] = None
    total_files_reviewed: int = 0
    total_lines_reviewed: int = 0
    findings: List[CodeReviewFinding] = field(default_factory=list)
    quality_score: float = 0.0
    coverage_percentage: float = 0.0
    rule_violations: Dict[str, int] = field(default_factory=dict)


class CodeReviewFramework:
    """
    Automated code review framework for FEDZK

    Performs static analysis, security checks, and best practices validation.
    """

    def __init__(self, target_directory: str = None):
        self.target_directory = Path(target_directory or Path(__file__).parent.parent.parent)
        self._rules = self._initialize_review_rules()
        self._findings: List[CodeReviewFinding] = []

    def _initialize_review_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize code review rules"""
        return {
            'SEC001': {
                'title': 'Hardcoded Password/Secret',
                'description': 'Avoid hardcoded passwords, secrets, or sensitive data',
                'severity': ReviewSeverity.CRITICAL,
                'category': ReviewCategory.SECURITY,
                'pattern': r'(?i)(password|secret|key|token)\s*[:=]\s*["\'][^"\']+["\']',
                'recommendation': 'Use environment variables or secure key management systems'
            },
            'SEC002': {
                'title': 'SQL Injection Risk',
                'description': 'Potential SQL injection vulnerability',
                'severity': ReviewSeverity.CRITICAL,
                'category': ReviewCategory.SECURITY,
                'pattern': r'(?i)(select|insert|update|delete).*?\+.*?(request\.|input)',
                'recommendation': 'Use parameterized queries or ORM with proper sanitization'
            },
            'SEC003': {
                'title': 'Weak Cryptographic Hash',
                'description': 'Use of weak cryptographic hash functions',
                'severity': ReviewSeverity.MAJOR,
                'category': ReviewCategory.SECURITY,
                'pattern': r'(?i)(md5|sha1)\(',
                'recommendation': 'Use SHA-256 or stronger hashing algorithms'
            },
            'SEC004': {
                'title': 'Unsafe Deserialization',
                'description': 'Use of unsafe deserialization methods',
                'severity': ReviewSeverity.CRITICAL,
                'category': ReviewCategory.SECURITY,
                'pattern': r'(?i)(pickle\.load|yaml\.unsafe_load)',
                'recommendation': 'Use safe deserialization methods or validate input'
            },
            'SEC005': {
                'title': 'Missing Input Validation',
                'description': 'Function without input validation',
                'severity': ReviewSeverity.MAJOR,
                'category': ReviewCategory.SECURITY,
                'pattern': r'def\s+\w+\([^)]*\):\s*$',
                'recommendation': 'Add comprehensive input validation and sanitization'
            },
            'PERF001': {
                'title': 'Inefficient List Comprehension',
                'description': 'Consider using generator expressions for large datasets',
                'severity': ReviewSeverity.MINOR,
                'category': ReviewCategory.PERFORMANCE,
                'pattern': r'\[.*for.*in.*\].*len\(',
                'recommendation': 'Use generator expressions or iterators for memory efficiency'
            },
            'MAINT001': {
                'title': 'Function Too Long',
                'description': 'Function exceeds recommended length limit',
                'severity': ReviewSeverity.MAJOR,
                'category': ReviewCategory.MAINTAINABILITY,
                'check_function': self._check_function_length,
                'recommendation': 'Break down large functions into smaller, focused functions'
            },
            'MAINT002': {
                'title': 'Missing Docstring',
                'description': 'Function or class missing documentation',
                'severity': ReviewSeverity.MINOR,
                'category': ReviewCategory.MAINTAINABILITY,
                'check_function': self._check_missing_docstring,
                'recommendation': 'Add comprehensive docstrings following PEP 257'
            },
            'BEST001': {
                'title': 'Unused Import',
                'description': 'Import statement not used in code',
                'severity': ReviewSeverity.MINOR,
                'category': ReviewCategory.BEST_PRACTICES,
                'check_function': self._check_unused_imports,
                'recommendation': 'Remove unused imports to reduce namespace pollution'
            },
            'COMP001': {
                'title': 'Logging Compliance',
                'description': 'Ensure sensitive data is not logged',
                'severity': ReviewSeverity.MAJOR,
                'category': ReviewCategory.COMPLIANCE,
                'pattern': r'(?i)log\.(debug|info|warning|error).*?(password|secret|key|token)',
                'recommendation': 'Never log sensitive information like passwords or secrets'
            }
        }

    def perform_code_review(self, file_patterns: List[str] = None) -> CodeReviewReport:
        """
        Perform comprehensive code review

        Args:
            file_patterns: List of file patterns to review (default: Python files)

        Returns:
            CodeReviewReport: Detailed review results
        """
        import time
        import uuid

        review_id = f"review_{uuid.uuid4().hex[:8]}"
        report = CodeReviewReport(
            review_id=review_id,
            target_directory=str(self.target_directory),
            review_start_time=time.strftime('%Y-%m-%d %H:%M:%S')
        )

        try:
            # Default to Python files if no patterns specified
            if file_patterns is None:
                file_patterns = ['*.py']

            files_to_review = []
            for pattern in file_patterns:
                files_to_review.extend(self.target_directory.rglob(pattern))

            # Remove duplicates and exclude common directories
            files_to_review = list(set(files_to_review))
            files_to_review = [f for f in files_to_review if not any(skip in str(f) for skip in ['__pycache__', '.git', 'node_modules'])]

            total_lines = 0

            for file_path in files_to_review:
                try:
                    lines_reviewed = self._review_single_file(file_path)
                    total_lines += lines_reviewed
                    report.total_files_reviewed += 1
                except Exception as e:
                    logger.warning(f"Failed to review {file_path}: {e}")

            report.total_lines_reviewed = total_lines
            report.findings = self._findings.copy()
            report.quality_score = self._calculate_quality_score()
            report.coverage_percentage = (report.total_files_reviewed / len(files_to_review)) * 100 if files_to_review else 100
            report.rule_violations = self._calculate_rule_violations()

        except Exception as e:
            logger.error(f"Code review failed: {e}")

        import time
        report.review_end_time = time.strftime('%Y-%m-%d %H:%M:%S')
        return report

    def _review_single_file(self, file_path: Path) -> int:
        """Review a single file and return number of lines reviewed"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            lines = content.split('\n')
            lines_reviewed = len(lines)

            # Apply pattern-based rules
            self._apply_pattern_rules(file_path, content, lines)

            # Apply AST-based rules
            self._apply_ast_rules(file_path, content)

            # Apply custom rules
            self._apply_custom_rules(file_path, content)

            return lines_reviewed

        except Exception as e:
            logger.error(f"Error reviewing {file_path}: {e}")
            return 0

    def _apply_pattern_rules(self, file_path: Path, content: str, lines: List[str]):
        """Apply pattern-based review rules"""
        for rule_id, rule in self._rules.items():
            if 'pattern' not in rule:
                continue

            matches = re.finditer(rule['pattern'], content, re.MULTILINE)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                code_snippet = self._extract_code_snippet(lines, line_number - 1)

                finding = CodeReviewFinding(
                    id=f"{rule_id}_{hashlib.md5(f'{file_path}_{line_number}'.encode()).hexdigest()[:8]}",
                    title=rule['title'],
                    description=rule['description'],
                    severity=rule['severity'],
                    category=rule['category'],
                    file_path=str(file_path),
                    line_number=line_number,
                    code_snippet=code_snippet,
                    recommendation=rule['recommendation'],
                    rule_id=rule_id,
                    evidence={'matched_pattern': match.group()},
                    context_lines=self._get_context_lines(lines, line_number - 1)
                )
                self._findings.append(finding)

    def _apply_ast_rules(self, file_path: Path, content: str):
        """Apply AST-based review rules"""
        try:
            tree = ast.parse(content)

            for rule_id, rule in self._rules.items():
                if 'check_function' in rule:
                    check_func = rule['check_function']
                    if hasattr(self, check_func.__name__):
                        findings = getattr(self, check_func.__name__)(tree, file_path)
                        self._findings.extend(findings)

        except SyntaxError:
            logger.warning(f"Syntax error in {file_path}, skipping AST analysis")

    def _apply_custom_rules(self, file_path: Path, content: str):
        """Apply custom review rules"""
        # Check for TODO comments
        if 'TODO' in content.upper():
            finding = CodeReviewFinding(
                id=f"TODO_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}",
                title="TODO Comment Found",
                description="Code contains TODO comments that should be addressed",
                severity=ReviewSeverity.INFO,
                category=ReviewCategory.MAINTAINABILITY,
                file_path=str(file_path),
                line_number=None,
                code_snippet="Contains TODO comments",
                recommendation="Address or document TODO items",
                rule_id="MAINT003",
                evidence={'todo_count': content.upper().count('TODO')}
            )
            self._findings.append(finding)

    def _check_function_length(self, tree: ast.AST, file_path: Path) -> List[CodeReviewFinding]:
        """Check for functions that are too long"""
        findings = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count lines in function
                if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                    function_length = node.end_lineno - node.lineno
                    if function_length > 50:  # Configurable threshold
                        finding = CodeReviewFinding(
                            id=f"MAINT001_{hashlib.md5(f'{file_path}_{node.lineno}'.encode()).hexdigest()[:8]}",
                            title="Function Too Long",
                            description=f"Function '{node.name}' is {function_length} lines long",
                            severity=ReviewSeverity.MAJOR,
                            category=ReviewCategory.MAINTAINABILITY,
                            file_path=str(file_path),
                            line_number=node.lineno,
                            code_snippet=f"def {node.name}(...):",
                            recommendation="Break down into smaller functions",
                            rule_id="MAINT001",
                            evidence={'function_length': function_length}
                        )
                        findings.append(finding)

        return findings

    def _check_missing_docstring(self, tree: ast.AST, file_path: Path) -> List[CodeReviewFinding]:
        """Check for missing docstrings"""
        findings = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                has_docstring = False
                if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Str)):
                    has_docstring = True

                if not has_docstring:
                    finding = CodeReviewFinding(
                        id=f"MAINT002_{hashlib.md5(f'{file_path}_{node.lineno}'.encode()).hexdigest()[:8]}",
                        title="Missing Docstring",
                        description=f"{'Function' if isinstance(node, ast.FunctionDef) else 'Class'} '{node.name}' missing docstring",
                        severity=ReviewSeverity.MINOR,
                        category=ReviewCategory.MAINTAINABILITY,
                        file_path=str(file_path),
                        line_number=node.lineno,
                        code_snippet=f"{'def' if isinstance(node, ast.FunctionDef) else 'class'} {node.name}:",
                        recommendation="Add comprehensive docstring",
                        rule_id="MAINT002"
                    )
                    findings.append(finding)

        return findings

    def _check_unused_imports(self, tree: ast.AST, file_path: Path) -> List[CodeReviewFinding]:
        """Check for unused imports"""
        findings = []

        # Get all imports
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name] = {'line': node.lineno, 'used': False}
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports[alias.asname or alias.name] = {'line': node.lineno, 'used': False}

        # Check usage
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id in imports:
                    imports[node.id]['used'] = True

        # Report unused imports
        for import_name, info in imports.items():
            if not info['used'] and import_name != '*':
                finding = CodeReviewFinding(
                    id="BEST001_" + hashlib.md5(f"{file_path}_{info["line"]}".encode()).hexdigest()[:8],
                    title="Unused Import",
                    description=f"Import '{import_name}' is not used",
                    severity=ReviewSeverity.MINOR,
                    category=ReviewCategory.BEST_PRACTICES,
                    file_path=str(file_path),
                    line_number=info['line'],
                    code_snippet=f"import {import_name}",
                    recommendation="Remove unused import",
                    rule_id="BEST001"
                )
                findings.append(finding)

        return findings

    def _extract_code_snippet(self, lines: List[str], line_number: int, context: int = 2) -> str:
        """Extract code snippet with context"""
        start = max(0, line_number - context)
        end = min(len(lines), line_number + context + 1)

        snippet_lines = []
        for i in range(start, end):
            marker = ">>> " if i == line_number else "    "
            snippet_lines.append("2")

        return '\n'.join(snippet_lines)

    def _get_context_lines(self, lines: List[str], line_number: int, context: int = 3) -> List[str]:
        """Get context lines around a finding"""
        start = max(0, line_number - context)
        end = min(len(lines), line_number + context + 1)
        return lines[start:end]

    def _calculate_quality_score(self) -> float:
        """Calculate code quality score"""
        if not self._findings:
            return 100.0

        # Weight findings by severity
        severity_weights = {
            ReviewSeverity.BLOCKER: 10,
            ReviewSeverity.CRITICAL: 8,
            ReviewSeverity.MAJOR: 5,
            ReviewSeverity.MINOR: 2,
            ReviewSeverity.INFO: 1
        }

        total_penalty = sum(severity_weights[finding.severity] for finding in self._findings)
        max_reasonable_penalty = len(self._findings) * 10

        return max(0.0, 100.0 - (total_penalty / max_reasonable_penalty) * 100)

    def _calculate_rule_violations(self) -> Dict[str, int]:
        """Calculate violations per rule"""
        violations = {}
        for finding in self._findings:
            violations[finding.rule_id] = violations.get(finding.rule_id, 0) + 1
        return violations

    def generate_review_summary(self, report: CodeReviewReport) -> Dict[str, Any]:
        """Generate a summary of the code review"""
        severity_counts = {}
        category_counts = {}

        for finding in report.findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
            category_counts[finding.category] = category_counts.get(finding.category, 0) + 1

        return {
            'review_id': report.review_id,
            'total_files': report.total_files_reviewed,
            'total_lines': report.total_lines_reviewed,
            'total_findings': len(report.findings),
            'quality_score': report.quality_score,
            'coverage_percentage': report.coverage_percentage,
            'findings_by_severity': severity_counts,
            'findings_by_category': category_counts,
            'rule_violations': report.rule_violations,
            'top_rules': sorted(report.rule_violations.items(), key=lambda x: x[1], reverse=True)[:5]
        }
