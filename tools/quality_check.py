#!/usr/bin/env python3
"""
FEDZK Quality Check Tool

Comprehensive code quality assessment for FEDZK codebase.
Implements quality gates and automated code review checks.
"""

import os
import sys
import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class QualityLevel(Enum):
    """Quality assessment levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class QualityIssue:
    """Represents a quality issue found in code."""

    def __init__(
        self,
        file_path: str,
        line_number: int,
        column: int,
        issue_type: str,
        level: QualityLevel,
        message: str,
        suggestion: str = "",
        code_snippet: str = ""
    ):
        self.file_path = file_path
        self.line_number = line_number
        self.column = column
        self.issue_type = issue_type
        self.level = level
        self.message = message
        self.suggestion = suggestion
        self.code_snippet = code_snippet

    def to_dict(self) -> Dict:
        """Convert issue to dictionary for JSON serialization."""
        return {
            'file_path': self.file_path,
            'line_number': self.line_number,
            'column': self.column,
            'issue_type': self.issue_type,
            'level': self.level.value,
            'message': self.message,
            'suggestion': self.suggestion,
            'code_snippet': self.code_snippet
        }


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""

    total_files: int = 0
    total_lines: int = 0
    issues: List[QualityIssue] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    passed: bool = True

    def add_issue(self, issue: QualityIssue):
        """Add an issue to the report."""
        self.issues.append(issue)
        level = issue.level.value
        self.summary[level] = self.summary.get(level, 0) + 1

        # Critical and High issues cause failure
        if issue.level in [QualityLevel.CRITICAL, QualityLevel.HIGH]:
            self.passed = False

    def to_dict(self) -> Dict:
        """Convert report to dictionary for JSON serialization."""
        return {
            'total_files': self.total_files,
            'total_lines': self.total_lines,
            'total_issues': len(self.issues),
            'issues_by_level': self.summary,
            'passed': self.passed,
            'issues': [issue.to_dict() for issue in self.issues]
        }


class FEDZKQualityChecker:
    """Comprehensive FEDZK code quality checker."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.report = QualityReport()

        # FEDZK-specific patterns and rules
        self.mock_patterns = [
            r'\bmock\b',
            r'\bstub\b',
            r'\bfake\b',
            r'\bdummy\b',
            r'\btest_mode\b',
            r'\bdebug_mode\b'
        ]

        self.critical_imports = [
            'unittest.mock',
            'pytest.mock'
        ]

    def check_file(self, file_path: Path) -> List[QualityIssue]:
        """Check a single file for quality issues."""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # Update report statistics
            self.report.total_files += 1
            self.report.total_lines += len(lines)

            # Perform various checks
            issues.extend(self._check_mock_usage(file_path, content, lines))
            issues.extend(self._check_critical_imports(file_path, content))
            issues.extend(self._check_security_issues(file_path, content, lines))
            issues.extend(self._check_code_complexity(file_path, content))
            issues.extend(self._check_documentation(file_path, content))
            issues.extend(self._check_naming_conventions(file_path, content))
            issues.extend(self._check_error_handling(file_path, content))

        except Exception as e:
            issues.append(QualityIssue(
                str(file_path), 0, 0, "FILE_READ_ERROR",
                QualityLevel.CRITICAL,
                f"Failed to read file: {str(e)}"
            ))

        return issues

    def _check_mock_usage(self, file_path: Path, content: str, lines: List[str]) -> List[QualityIssue]:
        """Check for inappropriate mock usage in production code."""
        issues = []

        # Skip test files
        if 'test' in str(file_path).lower():
            return issues

        for i, line in enumerate(lines, 1):
            line_lower = line.lower()

            # Check for mock-related keywords
            for pattern in self.mock_patterns:
                if re.search(pattern, line_lower) and not self._is_comment_or_docstring(line):
                    issues.append(QualityIssue(
                        str(file_path), i, 0, "MOCK_USAGE",
                        QualityLevel.CRITICAL,
                        f"Found mock-related keyword '{pattern}' in production code",
                        "Remove mock implementations and use real cryptographic operations",
                        line.strip()
                    ))

        return issues

    def _check_critical_imports(self, file_path: Path, content: str) -> List[QualityIssue]:
        """Check for critical imports that shouldn't be in production code."""
        issues = []

        # Skip test files and specific allowed files
        if ('test' in str(file_path).lower() or
            'conftest.py' in str(file_path) or
            'test_' in str(file_path).name):
            return issues

        for critical_import in self.critical_imports:
            if critical_import in content:
                issues.append(QualityIssue(
                    str(file_path), 0, 0, "CRITICAL_IMPORT",
                    QualityLevel.CRITICAL,
                    f"Found critical import '{critical_import}' in production code",
                    "Remove mock imports and use real implementations",
                    critical_import
                ))

        return issues

    def _check_security_issues(self, file_path: Path, content: str, lines: List[str]) -> List[QualityIssue]:
        """Check for security-related issues."""
        issues = []

        # Check for hardcoded secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']*["\']',
            r'secret\s*=\s*["\'][^"\']*["\']',
            r'key\s*=\s*["\'][^"\']*["\']',
            r'token\s*=\s*["\'][^"\']*["\']'
        ]

        for i, line in enumerate(lines, 1):
            for pattern in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(QualityIssue(
                        str(file_path), i, 0, "HARDCODED_SECRET",
                        QualityLevel.CRITICAL,
                        "Found potential hardcoded secret",
                        "Use environment variables or secure secret management",
                        line.strip()
                    ))

        # Check for insecure random usage
        if 'random.random()' in content or 'random.randint(' in content:
            issues.append(QualityIssue(
                str(file_path), 0, 0, "INSECURE_RANDOM",
                QualityLevel.HIGH,
                "Found insecure random number generation",
                "Use secrets module or cryptography library for secure random generation"
            ))

        return issues

    def _check_code_complexity(self, file_path: Path, content: str) -> List[QualityIssue]:
        """Check code complexity metrics."""
        issues = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check function complexity
                    complexity = self._calculate_complexity(node)
                    if complexity > 15:
                        issues.append(QualityIssue(
                            str(file_path), node.lineno, 0, "HIGH_COMPLEXITY",
                            QualityLevel.MEDIUM,
                            f"Function '{node.name}' has high complexity ({complexity})",
                            "Consider breaking down into smaller functions",
                            f"def {node.name}(...):"
                        ))

                    # Check function length
                    if len(node.body) > 50:
                        issues.append(QualityIssue(
                            str(file_path), node.lineno, 0, "LONG_FUNCTION",
                            QualityLevel.MEDIUM,
                            f"Function '{node.name}' is too long ({len(node.body)} lines)",
                            "Consider breaking down into smaller functions"
                        ))

        except SyntaxError:
            pass  # Skip files with syntax errors

        return issues

    def _check_documentation(self, file_path: Path, content: str) -> List[QualityIssue]:
        """Check documentation quality."""
        issues = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    # Check for docstrings
                    if not ast.get_docstring(node):
                        node_type = type(node).__name__.replace('Def', '').lower()
                        issues.append(QualityIssue(
                            str(file_path), node.lineno, 0, "MISSING_DOCSTRING",
                            QualityLevel.LOW,
                            f"Missing docstring for {node_type} '{getattr(node, 'name', 'module')}'",
                            "Add comprehensive docstring following Google style"
                        ))

        except SyntaxError:
            pass

        return issues

    def _check_naming_conventions(self, file_path: Path, content: str) -> List[QualityIssue]:
        """Check naming convention compliance."""
        issues = []

        # Check for non-snake_case function names
        function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        for match in re.finditer(function_pattern, content):
            func_name = match.group(1)
            if not re.match(r'^[a-z_][a-z0-9_]*$', func_name):
                issues.append(QualityIssue(
                    str(file_path), 0, 0, "NAMING_CONVENTION",
                    QualityLevel.LOW,
                    f"Function '{func_name}' doesn't follow snake_case convention",
                    "Use snake_case for function names"
                ))

        return issues

    def _check_error_handling(self, file_path: Path, content: str) -> List[QualityIssue]:
        """Check error handling practices."""
        issues = []

        # Check for bare except clauses
        if 'except:' in content or 'except :' in content:
            issues.append(QualityIssue(
                str(file_path), 0, 0, "BARE_EXCEPT",
                QualityLevel.MEDIUM,
                "Found bare except clause",
                "Specify exception types explicitly (e.g., 'except ValueError:')"
            ))

        # Check for print statements in production code
        if 'print(' in content and 'test' not in str(file_path).lower():
            issues.append(QualityIssue(
                str(file_path), 0, 0, "PRINT_STATEMENT",
                QualityLevel.LOW,
                "Found print statement in production code",
                "Use proper logging instead of print statements"
            ))

        return issues

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1

        return complexity

    def _is_comment_or_docstring(self, line: str) -> bool:
        """Check if a line is a comment or docstring."""
        line = line.strip()
        return line.startswith('#') or line.startswith('"""') or line.startswith("'''")

    def run_quality_check(self, paths: Optional[List[Path]] = None) -> QualityReport:
        """Run comprehensive quality check on specified paths."""
        if paths is None:
            # Default to source code directories
            paths = [
                self.project_root / 'src',
                self.project_root / 'fedzk'
            ]

        # Find all Python files
        python_files = []
        for path in paths:
            if path.exists():
                if path.is_file() and path.suffix == '.py':
                    python_files.append(path)
                elif path.is_dir():
                    python_files.extend(path.rglob('*.py'))

        # Check each file
        for file_path in python_files:
            issues = self.check_file(file_path)
            for issue in issues:
                self.report.add_issue(issue)

        return self.report


def main():
    """Main entry point for quality check tool."""
    project_root = Path(__file__).parent.parent

    checker = FEDZKQualityChecker(project_root)
    report = checker.run_quality_check()

    # Print summary
    print("🔍 FEDZK Quality Check Report")
    print("=" * 50)
    print(f"Files analyzed: {report.total_files}")
    print(f"Total lines: {report.total_lines}")
    print(f"Total issues: {len(report.issues)}")
    print()

    if report.summary:
        print("Issues by severity:")
        for level, count in report.summary.items():
            print(f"  {level}: {count}")
        print()

    # Print issues
    for issue in report.issues[:20]:  # Show first 20 issues
        print(f"{issue.level.value}: {issue.file_path}:{issue.line_number}")
        print(f"  {issue.message}")
        if issue.suggestion:
            print(f"  💡 {issue.suggestion}")
        print()

    if len(report.issues) > 20:
        print(f"... and {len(report.issues) - 20} more issues")
        print()

    # Exit with appropriate code
    if report.passed:
        print("✅ All quality checks passed!")
        sys.exit(0)
    else:
        print("❌ Quality checks failed!")
        print("Fix critical and high severity issues before committing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
