#!/usr/bin/env python3
"""
FEDZK Automated Code Review Tool

Comprehensive automated code review system for FEDZK.
Provides intelligent code analysis, style checking, and quality assessment.
"""

import os
import sys
import ast
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import difflib


class ReviewSeverity(Enum):
    """Code review severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class CodeReviewComment:
    """Represents a code review comment."""

    file_path: str
    line_number: int
    severity: ReviewSeverity
    category: str
    message: str
    suggestion: str = ""
    code_snippet: str = ""
    rule_id: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        """Convert comment to dictionary."""
        return {
            'file_path': self.file_path,
            'line_number': self.line_number,
            'severity': self.severity.value,
            'category': self.category,
            'message': self.message,
            'suggestion': self.suggestion,
            'code_snippet': self.code_snippet,
            'rule_id': self.rule_id,
            'confidence': self.confidence
        }


@dataclass
class CodeReviewReport:
    """Comprehensive code review report."""

    comments: List[CodeReviewComment] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    files_reviewed: int = 0
    total_lines: int = 0
    passed: bool = True

    def add_comment(self, comment: CodeReviewComment):
        """Add a review comment."""
        self.comments.append(comment)
        severity = comment.severity.value
        self.summary[severity] = self.summary.get(severity, 0) + 1

        # Critical and High comments cause failure
        if comment.severity in [ReviewSeverity.CRITICAL, ReviewSeverity.HIGH]:
            self.passed = False

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            'files_reviewed': self.files_reviewed,
            'total_lines': self.total_lines,
            'total_comments': len(self.comments),
            'comments_by_severity': self.summary,
            'passed': self.passed,
            'comments': [c.to_dict() for c in self.comments]
        }


class FEDZKCodeReviewer:
    """Comprehensive FEDZK code reviewer."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.report = CodeReviewReport()

        # FEDZK-specific review rules
        self.rules = {
            'security': self._check_security_issues,
            'performance': self._check_performance_issues,
            'maintainability': self._check_maintainability,
            'best_practices': self._check_best_practices,
            'zk_specific': self._check_zk_specific_issues,
            'cryptography': self._check_cryptography_issues
        }

    def review_codebase(self, changed_files: Optional[List[Path]] = None) -> CodeReviewReport:
        """Review the entire codebase or specific changed files."""
        if changed_files is None:
            # Review all Python files
            changed_files = list(self.project_root.rglob('*.py'))

        for file_path in changed_files:
            if file_path.exists() and file_path.suffix == '.py':
                self._review_file(file_path)

        return self.report

    def review_pull_request(self, base_branch: str = 'main', head_branch: str = 'HEAD') -> CodeReviewReport:
        """Review changes in a pull request."""
        try:
            # Get changed files
            result = subprocess.run(
                ['git', 'diff', '--name-only', f'{base_branch}..{head_branch}'],
                capture_output=True, text=True, cwd=self.project_root
            )

            changed_files = []
            if result.returncode == 0:
                for file_path in result.stdout.strip().split('\n'):
                    if file_path and file_path.endswith('.py'):
                        changed_files.append(self.project_root / file_path)

            # Get diff for detailed analysis
            result = subprocess.run(
                ['git', 'diff', base_branch, head_branch],
                capture_output=True, text=True, cwd=self.project_root
            )

            diff_content = result.stdout if result.returncode == 0 else ""

            # Review changed files with diff context
            for file_path in changed_files:
                if file_path.exists():
                    self._review_file_with_diff(file_path, diff_content)

        except Exception as e:
            print(f"Warning: Could not analyze pull request: {e}")

        return self.report

    def _review_file(self, file_path: Path):
        """Review a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            self.report.files_reviewed += 1
            self.report.total_lines += len(lines)

            # Apply all review rules
            for rule_name, rule_func in self.rules.items():
                comments = rule_func(file_path, content, lines)
                for comment in comments:
                    self.report.add_comment(comment)

        except Exception as e:
            self.report.add_comment(CodeReviewComment(
                str(file_path), 0, ReviewSeverity.HIGH, "FILE_ERROR",
                f"Failed to review file: {str(e)}",
                "Ensure file is properly formatted and accessible"
            ))

    def _review_file_with_diff(self, file_path: Path, diff_content: str):
        """Review a file with diff context for focused analysis."""
        # First do regular review
        self._review_file(file_path)

        # Then analyze diff-specific issues
        diff_comments = self._analyze_diff_issues(file_path, diff_content)
        for comment in diff_comments:
            self.report.add_comment(comment)

    def _check_security_issues(self, file_path: Path, content: str, lines: List[str]) -> List[CodeReviewComment]:
        """Check for security-related issues."""
        comments = []

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
                    comments.append(CodeReviewComment(
                        str(file_path), i, ReviewSeverity.CRITICAL, "SECURITY",
                        "Potential hardcoded secret detected",
                        "Use environment variables or secure secret management",
                        line.strip(), "SEC001"
                    ))

        # Check for insecure imports
        insecure_imports = ['pickle', 'shelve', 'marshal']
        for insecure_import in insecure_imports:
            if insecure_import in content:
                comments.append(CodeReviewComment(
                    str(file_path), 0, ReviewSeverity.HIGH, "SECURITY",
                    f"Insecure import: {insecure_import}",
                    "Use secure alternatives or validate input carefully",
                    insecure_import, "SEC002"
                ))

        return comments

    def _check_performance_issues(self, file_path: Path, content: str, lines: List[str]) -> List[CodeReviewComment]:
        """Check for performance-related issues."""
        comments = []

        # Check for inefficient patterns
        inefficient_patterns = [
            (r'for\s+.*\s+in\s+range\(len\(.*\)\)', "PERF001", "Use enumerate() instead of range(len())"),
            (r'\.append\(.*\)\s*$', "PERF002", "Consider using list comprehension for multiple appends"),
            (r'print\(.*\)', "PERF003", "Remove debug print statements in production code")
        ]

        for i, line in enumerate(lines, 1):
            for pattern, rule_id, suggestion in inefficient_patterns:
                if re.search(pattern, line):
                    comments.append(CodeReviewComment(
                        str(file_path), i, ReviewSeverity.MEDIUM, "PERFORMANCE",
                        f"Performance issue: {suggestion}",
                        suggestion, line.strip(), rule_id
                    ))

        # Check for large data structures in memory
        if 'list(' in content and len(content) > 10000:
            comments.append(CodeReviewComment(
                str(file_path), 0, ReviewSeverity.MEDIUM, "PERFORMANCE",
                "Large data structures detected",
                "Consider using generators or streaming for large datasets",
                "Large data structure", "PERF004"
            ))

        return comments

    def _check_maintainability(self, file_path: Path, content: str, lines: List[str]) -> List[CodeReviewComment]:
        """Check for maintainability issues."""
        comments = []

        # Check function length
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if len(node.body) > 50:
                        comments.append(CodeReviewComment(
                            str(file_path), node.lineno, ReviewSeverity.MEDIUM, "MAINTAINABILITY",
                            f"Function '{node.name}' is too long ({len(node.body)} lines)",
                            "Break down into smaller, focused functions",
                            f"def {node.name}(...):", "MAINT001"
                        ))

                    # Check for missing docstrings
                    if not ast.get_docstring(node):
                        comments.append(CodeReviewComment(
                            str(file_path), node.lineno, ReviewSeverity.LOW, "MAINTAINABILITY",
                            f"Missing docstring for function '{node.name}'",
                            "Add comprehensive docstring following Google style",
                            f"def {node.name}(...):", "MAINT002"
                        ))

        except SyntaxError:
            pass  # Skip files with syntax errors

        # Check for TODO comments
        for i, line in enumerate(lines, 1):
            if 'TODO' in line or 'FIXME' in line or 'XXX' in line:
                comments.append(CodeReviewComment(
                    str(file_path), i, ReviewSeverity.LOW, "MAINTAINABILITY",
                    "TODO/FIXME comment found",
                    "Address the TODO item or create a proper issue",
                    line.strip(), "MAINT003"
                ))

        return comments

    def _check_best_practices(self, file_path: Path, content: str, lines: List[str]) -> List[CodeReviewComment]:
        """Check for best practices violations."""
        comments = []

        # Check for type hints
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if function has type hints
                    has_type_hints = (
                        node.returns is not None or
                        any(arg.annotation for arg in node.args.args)
                    )

                    if not has_type_hints and len(node.body) > 10:
                        comments.append(CodeReviewComment(
                            str(file_path), node.lineno, ReviewSeverity.LOW, "BEST_PRACTICE",
                            f"Function '{node.name}' missing type hints",
                            "Add type hints for better code documentation and IDE support",
                            f"def {node.name}(...):", "BEST001"
                        ))

        except SyntaxError:
            pass

        # Check for magic numbers
        magic_number_pattern = r'\b\d{2,}\b'
        for i, line in enumerate(lines, 1):
            matches = re.findall(magic_number_pattern, line)
            for match in matches:
                # Skip common non-magic numbers
                if match in ['0', '1', '10', '100', '1000', '60', '24', '365']:
                    continue

                comments.append(CodeReviewComment(
                    str(file_path), i, ReviewSeverity.LOW, "BEST_PRACTICE",
                    f"Magic number '{match}' found",
                    "Replace with named constant for better readability",
                    line.strip(), "BEST002"
                ))

        return comments

    def _check_zk_specific_issues(self, file_path: Path, content: str, lines: List[str]) -> List[CodeReviewComment]:
        """Check for ZK-specific issues."""
        comments = []

        # Check for mock implementations
        mock_keywords = ['mock', 'stub', 'fake', 'dummy', 'test_mode']
        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            for keyword in mock_keywords:
                if keyword in line_lower and 'test' not in str(file_path).lower():
                    comments.append(CodeReviewComment(
                        str(file_path), i, ReviewSeverity.CRITICAL, "ZK_SPECIFIC",
                        f"Mock implementation found in production code: '{keyword}'",
                        "Remove mock implementations and use real ZK operations",
                        line.strip(), "ZK001"
                    ))

        # Check for proper ZK circuit validation
        if 'zk' in content.lower() and 'circuit' in content.lower():
            if 'validate' not in content.lower():
                comments.append(CodeReviewComment(
                    str(file_path), 0, ReviewSeverity.HIGH, "ZK_SPECIFIC",
                    "ZK circuit usage without validation",
                    "Add proper circuit validation and error handling",
                    "ZK circuit usage", "ZK002"
                ))

        return comments

    def _check_cryptography_issues(self, file_path: Path, content: str, lines: List[str]) -> List[CodeReviewComment]:
        """Check for cryptography-related issues."""
        comments = []

        # Check for weak cryptographic algorithms
        weak_algorithms = ['MD5', 'SHA1', 'DES', 'RC4']
        for algorithm in weak_algorithms:
            if algorithm in content:
                comments.append(CodeReviewComment(
                    str(file_path), 0, ReviewSeverity.CRITICAL, "CRYPTOGRAPHY",
                    f"Weak cryptographic algorithm: {algorithm}",
                    "Use modern, strong cryptographic algorithms",
                    algorithm, "CRYPTO001"
                ))

        # Check for proper key lengths
        if 'RSA' in content and '1024' in content:
            comments.append(CodeReviewComment(
                str(file_path), 0, ReviewSeverity.HIGH, "CRYPTOGRAPHY",
                "RSA key length appears insufficient (1024 bits)",
                "Use RSA keys of at least 2048 bits, preferably 4096 bits",
                "RSA 1024", "CRYPTO002"
            ))

        # Check for secure random usage
        if 'random.random()' in content or 'random.randint(' in content:
            comments.append(CodeReviewComment(
                str(file_path), 0, ReviewSeverity.MEDIUM, "CRYPTOGRAPHY",
                "Insecure random number generation detected",
                "Use secrets module or cryptography library for secure random generation",
                "random.random()", "CRYPTO003"
            ))

        return comments

    def _analyze_diff_issues(self, file_path: Path, diff_content: str) -> List[CodeReviewComment]:
        """Analyze diff content for specific issues."""
        comments = []

        # Check for large changes that might need review
        diff_lines = diff_content.split('\n')
        additions = sum(1 for line in diff_lines if line.startswith('+'))
        deletions = sum(1 for line in diff_lines if line.startswith('-'))

        if additions > 500 or deletions > 500:
            comments.append(CodeReviewComment(
                str(file_path), 0, ReviewSeverity.INFO, "LARGE_CHANGE",
                f"Large change detected ({additions} additions, {deletions} deletions)",
                "Consider breaking down into smaller, focused changes",
                f"Large change: +{additions} -{deletions}", "REVIEW001"
            ))

        # Check for new dependencies
        new_imports = []
        for line in diff_lines:
            if line.startswith('+') and ('import ' in line or 'from ' in line):
                new_imports.append(line[1:].strip())

        if len(new_imports) > 5:
            comments.append(CodeReviewComment(
                str(file_path), 0, ReviewSeverity.MEDIUM, "NEW_DEPENDENCIES",
                f"Multiple new imports added ({len(new_imports)})",
                "Review new dependencies for security and necessity",
                ", ".join(new_imports[:3]) + ("..." if len(new_imports) > 3 else ""),
                "REVIEW002"
            ))

        return comments

    def generate_review_summary(self) -> str:
        """Generate a comprehensive review summary."""
        summary_lines = []

        summary_lines.append("# FEDZK Automated Code Review Report")
        summary_lines.append("=" * 50)
        summary_lines.append("")

        # Overall status
        status = "✅ PASSED" if self.report.passed else "❌ FAILED"
        summary_lines.append(f"Overall Status: {status}")
        summary_lines.append("")

        # Summary statistics
        summary_lines.append("## Summary Statistics")
        summary_lines.append(f"- Files Reviewed: {self.report.files_reviewed}")
        summary_lines.append(f"- Total Lines: {self.report.total_lines}")
        summary_lines.append(f"- Total Comments: {len(self.report.comments)}")
        summary_lines.append("")

        if self.report.summary:
            summary_lines.append("## Comments by Severity")
            for severity, count in self.report.summary.items():
                summary_lines.append(f"- {severity}: {count}")
            summary_lines.append("")

        # Critical and High severity comments
        critical_comments = [c for c in self.report.comments
                           if c.severity in [ReviewSeverity.CRITICAL, ReviewSeverity.HIGH]]

        if critical_comments:
            summary_lines.append("## Critical Issues Requiring Attention")
            for comment in critical_comments:
                summary_lines.append(f"### {comment.category}: {comment.file_path}:{comment.line_number}")
                summary_lines.append(f"**{comment.message}**")
                if comment.suggestion:
                    summary_lines.append(f"💡 {comment.suggestion}")
                if comment.code_snippet:
                    summary_lines.append(f"```python\n{comment.code_snippet}\n```")
                summary_lines.append("")

        # Recommendations
        summary_lines.append("## Recommendations")
        if not self.report.passed:
            summary_lines.append("❌ **Blocking Issues Found**: Address critical and high severity comments before merging.")
        else:
            summary_lines.append("✅ **Ready for Review**: No blocking issues found.")

        summary_lines.append("")
        summary_lines.append("### Best Practices")
        summary_lines.append("- Ensure all public APIs have comprehensive docstrings")
        summary_lines.append("- Use type hints for better code documentation")
        summary_lines.append("- Avoid magic numbers; use named constants")
        summary_lines.append("- Remove debug print statements")
        summary_lines.append("- Add proper error handling and logging")
        summary_lines.append("")

        return "\n".join(summary_lines)


def main():
    """Main entry point for code review tool."""
    project_root = Path(__file__).parent.parent

    reviewer = FEDZKCodeReviewer(project_root)

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--pr':
            # Review pull request
            base_branch = sys.argv[2] if len(sys.argv) > 2 else 'main'
            head_branch = sys.argv[3] if len(sys.argv) > 3 else 'HEAD'
            report = reviewer.review_pull_request(base_branch, head_branch)
        elif sys.argv[1] == '--files':
            # Review specific files
            file_paths = [Path(f) for f in sys.argv[2:]]
            report = reviewer.review_codebase(file_paths)
        else:
            print("Usage: python code_review.py [--pr <base> <head>] [--files <file1> <file2> ...]")
            sys.exit(1)
    else:
        # Review entire codebase
        report = reviewer.review_codebase()

    # Generate and print summary
    summary = reviewer.generate_review_summary()
    print(summary)

    # Save detailed report
    report_file = project_root / 'code-review-report.json'
    with open(report_file, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"\n📋 Detailed report saved to: {report_file}")

    # Exit with appropriate code
    if report.passed:
        print("\n✅ Code review passed!")
        sys.exit(0)
    else:
        print("\n❌ Code review failed!")
        print("Address critical and high severity issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
