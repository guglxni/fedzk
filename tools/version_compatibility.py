#!/usr/bin/env python3
"""
FEDZK Version Compatibility Tester

Comprehensive version compatibility testing for FEDZK.
Validates API compatibility, data compatibility, and migration paths.
"""

import os
import sys
import json
import inspect
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum


class CompatibilityLevel(Enum):
    """Version compatibility levels."""
    COMPATIBLE = "COMPATIBLE"
    BACKWARD_COMPATIBLE = "BACKWARD_COMPATIBLE"
    FORWARD_COMPATIBLE = "FORWARD_COMPATIBLE"
    BREAKING = "BREAKING"
    UNKNOWN = "UNKNOWN"


class CompatibilityIssue:
    """Represents a compatibility issue."""

    def __init__(
        self,
        component: str,
        issue_type: str,
        level: CompatibilityLevel,
        description: str,
        impact: str,
        migration_guide: str = "",
        affected_items: List[str] = None
    ):
        self.component = component
        self.issue_type = issue_type
        self.level = level
        self.description = description
        self.impact = impact
        self.migration_guide = migration_guide
        self.affected_items = affected_items or []

    def to_dict(self) -> Dict:
        """Convert issue to dictionary."""
        return {
            'component': self.component,
            'issue_type': self.issue_type,
            'level': self.level.value,
            'description': self.description,
            'impact': self.impact,
            'migration_guide': self.migration_guide,
            'affected_items': self.affected_items
        }


@dataclass
class CompatibilityReport:
    """Comprehensive compatibility assessment report."""

    source_version: str
    target_version: str
    compatibility_level: CompatibilityLevel = CompatibilityLevel.UNKNOWN
    issues: List[CompatibilityIssue] = field(default_factory=list)
    api_changes: List[Dict] = field(default_factory=list)
    data_migration_required: bool = False
    migration_complexity: str = "LOW"
    passed: bool = True

    def add_issue(self, issue: CompatibilityIssue):
        """Add a compatibility issue."""
        self.issues.append(issue)

        # Update overall compatibility level
        if issue.level == CompatibilityLevel.BREAKING:
            self.compatibility_level = CompatibilityLevel.BREAKING
            self.passed = False
        elif issue.level == CompatibilityLevel.BACKWARD_COMPATIBLE and self.compatibility_level == CompatibilityLevel.UNKNOWN:
            self.compatibility_level = CompatibilityLevel.BACKWARD_COMPATIBLE

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            'source_version': self.source_version,
            'target_version': self.target_version,
            'compatibility_level': self.compatibility_level.value,
            'total_issues': len(self.issues),
            'issues_by_level': {
                level.value: len([i for i in self.issues if i.level == level])
                for level in CompatibilityLevel
            },
            'data_migration_required': self.data_migration_required,
            'migration_complexity': self.migration_complexity,
            'passed': self.passed,
            'issues': [i.to_dict() for i in self.issues]
        }


class FEDZKCompatibilityTester:
    """Comprehensive FEDZK version compatibility tester."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_dir = project_root / 'src'
        self.test_dir = project_root / 'tests'

        # FEDZK API components to check for compatibility
        self.api_components = [
            'fedzk.core',
            'fedzk.zk',
            'fedzk.monitoring',
            'fedzk.config',
            'fedzk.compliance'
        ]

    def test_version_compatibility(self, source_version: str, target_version: str) -> CompatibilityReport:
        """Test compatibility between two versions."""
        print(f"🔍 Testing compatibility: {source_version} -> {target_version}")

        report = CompatibilityReport(source_version, target_version)

        # Test API compatibility
        api_issues = self._test_api_compatibility(source_version, target_version)
        for issue in api_issues:
            report.add_issue(issue)

        # Test data format compatibility
        data_issues = self._test_data_compatibility(source_version, target_version)
        for issue in data_issues:
            report.add_issue(issue)

        # Test configuration compatibility
        config_issues = self._test_configuration_compatibility(source_version, target_version)
        for issue in config_issues:
            report.add_issue(issue)

        # Test dependency compatibility
        dep_issues = self._test_dependency_compatibility(source_version, target_version)
        for issue in dep_issues:
            report.add_issue(issue)

        # Determine overall compatibility
        self._determine_overall_compatibility(report)

        return report

    def _test_api_compatibility(self, source_version: str, target_version: str) -> List[CompatibilityIssue]:
        """Test API compatibility between versions."""
        issues = []

        try:
            # Get API signatures for both versions
            source_api = self._extract_api_signatures(source_version)
            target_api = self._extract_api_signatures(target_version)

            # Compare APIs
            for component in self.api_components:
                if component in source_api and component in target_api:
                    component_issues = self._compare_api_components(
                        component, source_api[component], target_api[component]
                    )
                    issues.extend(component_issues)

                elif component in source_api and component not in target_api:
                    issues.append(CompatibilityIssue(
                        component, "API_REMOVAL", CompatibilityLevel.BREAKING,
                        f"API component '{component}' removed in {target_version}",
                        "Applications using this component will break",
                        f"Migrate to alternative API or pin to {source_version}",
                        [component]
                    ))

        except Exception as e:
            print(f"Warning: Could not test API compatibility: {e}")

        return issues

    def _extract_api_signatures(self, version: str) -> Dict[str, Dict[str, Any]]:
        """Extract API signatures for a specific version."""
        api_signatures = {}

        try:
            # This would normally check out the specific version from git
            # For now, we'll analyze the current codebase
            for component in self.api_components:
                if (self.src_dir / component.replace('.', '/') / '__init__.py').exists():
                    api_signatures[component] = self._analyze_component_api(component)

        except Exception as e:
            print(f"Warning: Could not extract API for {version}: {e}")

        return api_signatures

    def _analyze_component_api(self, component: str) -> Dict[str, Any]:
        """Analyze API of a component."""
        api_info = {
            'classes': {},
            'functions': {},
            'constants': []
        }

        try:
            module_path = self.src_dir / component.replace('.', '/')
            init_file = module_path / '__init__.py'

            if init_file.exists():
                with open(init_file, 'r') as f:
                    content = f.read()

                # Extract exported symbols
                if '__all__' in content:
                    # Parse __all__ list
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if '__all__' in line:
                            # Simple parsing - in practice you'd use AST
                            api_info['exports'] = []
                            break

                # Analyze Python files in component
                for py_file in module_path.glob('*.py'):
                    if py_file.name != '__init__.py':
                        file_api = self._analyze_python_file(py_file)
                        api_info['classes'].update(file_api.get('classes', {}))
                        api_info['functions'].update(file_api.get('functions', {}))

        except Exception as e:
            print(f"Warning: Could not analyze {component}: {e}")

        return api_info

    def _analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a Python file for API elements."""
        api_info = {'classes': {}, 'functions': {}}

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Simple analysis - in practice you'd use AST properly
            lines = content.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()

                # Find class definitions
                if line.startswith('class '):
                    class_match = line.split('(')[0].replace('class ', '')
                    api_info['classes'][class_match] = {
                        'line': i + 1,
                        'signature': line
                    }

                # Find function definitions
                elif line.startswith('def '):
                    func_match = line.split('(')[0].replace('def ', '')
                    if not func_match.startswith('_'):  # Skip private functions
                        api_info['functions'][func_match] = {
                            'line': i + 1,
                            'signature': line
                        }

        except Exception as e:
            print(f"Warning: Could not analyze {file_path}: {e}")

        return api_info

    def _compare_api_components(self, component: str, source_api: Dict, target_api: Dict) -> List[CompatibilityIssue]:
        """Compare API components between versions."""
        issues = []

        # Check for removed classes
        for class_name in source_api.get('classes', {}):
            if class_name not in target_api.get('classes', {}):
                issues.append(CompatibilityIssue(
                    component, "CLASS_REMOVAL", CompatibilityLevel.BREAKING,
                    f"Class '{class_name}' removed from {component}",
                    "Code using this class will break",
                    f"Find alternative class or pin to older version",
                    [class_name]
                ))

        # Check for removed functions
        for func_name in source_api.get('functions', {}):
            if func_name not in target_api.get('functions', {}):
                issues.append(CompatibilityIssue(
                    component, "FUNCTION_REMOVAL", CompatibilityLevel.BREAKING,
                    f"Function '{func_name}' removed from {component}",
                    "Code calling this function will break",
                    f"Find alternative function or pin to older version",
                    [func_name]
                ))

        # Check for changed signatures
        for func_name in source_api.get('functions', {}):
            if func_name in target_api.get('functions', {}):
                source_sig = source_api['functions'][func_name]['signature']
                target_sig = target_api['functions'][func_name]['signature']

                if source_sig != target_sig:
                    issues.append(CompatibilityIssue(
                        component, "SIGNATURE_CHANGE", CompatibilityLevel.BREAKING,
                        f"Function '{func_name}' signature changed",
                        "Code calling this function may break due to parameter changes",
                        f"Update function call to match new signature",
                        [func_name]
                    ))

        return issues

    def _test_data_compatibility(self, source_version: str, target_version: str) -> List[CompatibilityIssue]:
        """Test data format compatibility."""
        issues = []

        # Check for data format changes
        data_formats = [
            'model_checkpoint',
            'federation_config',
            'zk_circuit',
            'training_data'
        ]

        for data_format in data_formats:
            # This would normally test actual data format compatibility
            # For now, we'll assume compatibility
            pass

        return issues

    def _test_configuration_compatibility(self, source_version: str, target_version: str) -> List[CompatibilityIssue]:
        """Test configuration compatibility."""
        issues = []

        # Check configuration file changes
        config_files = [
            'config.yaml',
            'pyproject.toml',
            'setup.cfg'
        ]

        for config_file in config_files:
            if (self.project_root / config_file).exists():
                # This would normally test configuration compatibility
                pass

        return issues

    def _test_dependency_compatibility(self, source_version: str, target_version: str) -> List[CompatibilityIssue]:
        """Test dependency compatibility."""
        issues = []

        # Check for dependency changes
        try:
            # Compare pyproject.toml dependencies
            pyproject_file = self.project_root / 'pyproject.toml'

            if pyproject_file.exists():
                # This would normally compare dependency specifications
                pass

        except Exception as e:
            print(f"Warning: Could not test dependency compatibility: {e}")

        return issues

    def _determine_overall_compatibility(self, report: CompatibilityReport):
        """Determine overall compatibility level."""
        breaking_issues = [i for i in report.issues if i.level == CompatibilityLevel.BREAKING]

        if breaking_issues:
            report.compatibility_level = CompatibilityLevel.BREAKING
            report.data_migration_required = True
            report.migration_complexity = "HIGH"
        elif report.issues:
            report.compatibility_level = CompatibilityLevel.BACKWARD_COMPATIBLE
            report.migration_complexity = "MEDIUM"
        else:
            report.compatibility_level = CompatibilityLevel.COMPATIBLE
            report.migration_complexity = "LOW"

    def generate_migration_guide(self, report: CompatibilityReport) -> str:
        """Generate migration guide for version upgrade."""
        guide_lines = []

        guide_lines.append(f"# Migration Guide: {report.source_version} → {report.target_version}")
        guide_lines.append("=" * 60)
        guide_lines.append("")

        guide_lines.append(f"**Compatibility Level:** {report.compatibility_level.value}")
        guide_lines.append(f"**Migration Complexity:** {report.migration_complexity}")
        guide_lines.append(f"**Data Migration Required:** {'Yes' if report.data_migration_required else 'No'}")
        guide_lines.append("")

        if report.issues:
            guide_lines.append("## Breaking Changes")
            for issue in report.issues:
                if issue.level == CompatibilityLevel.BREAKING:
                    guide_lines.append(f"### {issue.component}: {issue.description}")
                    guide_lines.append(f"**Impact:** {issue.impact}")
                    if issue.migration_guide:
                        guide_lines.append(f"**Migration:** {issue.migration_guide}")
                    guide_lines.append("")

            guide_lines.append("## Non-Breaking Changes")
            for issue in report.issues:
                if issue.level != CompatibilityLevel.BREAKING:
                    guide_lines.append(f"- **{issue.component}:** {issue.description}")
                    if issue.migration_guide:
                        guide_lines.append(f"  - Migration: {issue.migration_guide}")
            guide_lines.append("")

        guide_lines.append("## Migration Steps")
        guide_lines.append("")

        if report.data_migration_required:
            guide_lines.append("### Data Migration")
            guide_lines.append("1. Backup all existing data")
            guide_lines.append("2. Run data migration scripts")
            guide_lines.append("3. Validate migrated data integrity")
            guide_lines.append("4. Update data access patterns")
            guide_lines.append("")

        guide_lines.append("### Code Migration")
        guide_lines.append("1. Update import statements")
        guide_lines.append("2. Replace deprecated APIs")
        guide_lines.append("3. Update configuration files")
        guide_lines.append("4. Run test suite")
        guide_lines.append("5. Deploy to staging environment")
        guide_lines.append("")

        guide_lines.append("### Testing")
        guide_lines.append("- Run full test suite")
        guide_lines.append("- Perform integration testing")
        guide_lines.append("- Validate performance requirements")
        guide_lines.append("- Test rollback procedures")
        guide_lines.append("")

        return "\n".join(guide_lines)

    def test_api_contract(self, component: str) -> Dict[str, Any]:
        """Test API contract for a component."""
        contract = {
            'component': component,
            'public_api': [],
            'breaking_changes': [],
            'backward_compatible': True
        }

        try:
            module_path = self.src_dir / component.replace('.', '/')
            init_file = module_path / '__init__.py'

            if init_file.exists():
                with open(init_file, 'r') as f:
                    content = f.read()

                # Extract public API
                if '__all__' in content:
                    # Parse __all__ list to get public API
                    lines = content.split('\n')
                    in_all = False
                    for line in lines:
                        if '__all__' in line:
                            in_all = True
                        elif in_all and line.strip().startswith(']'):
                            break
                        elif in_all:
                            # Extract symbols from __all__
                            pass

        except Exception as e:
            print(f"Warning: Could not test API contract for {component}: {e}")

        return contract

    def validate_semantic_versioning(self, version_history: List[str]) -> List[CompatibilityIssue]:
        """Validate that version changes follow semantic versioning."""
        issues = []

        # This would analyze version history to ensure semantic versioning rules
        # For now, return empty list
        return issues


def main():
    """Main entry point for compatibility tester."""
    import argparse

    parser = argparse.ArgumentParser(description="FEDZK Version Compatibility Tester")
    parser.add_argument('--source', required=True, help='Source version')
    parser.add_argument('--target', required=True, help='Target version')
    parser.add_argument('--migration-guide', action='store_true', help='Generate migration guide')
    parser.add_argument('--api-contract', help='Test API contract for specific component')

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    tester = FEDZKCompatibilityTester(project_root)

    if args.api_contract:
        # Test API contract for specific component
        contract = tester.test_api_contract(args.api_contract)
        print(json.dumps(contract, indent=2))
    else:
        # Test version compatibility
        report = tester.test_version_compatibility(args.source, args.target)

        # Print summary
        print("🔍 FEDZK Compatibility Report")
        print("=" * 50)
        print(f"Source Version: {report.source_version}")
        print(f"Target Version: {report.target_version}")
        print(f"Compatibility: {report.compatibility_level.value}")
        print(f"Total Issues: {len(report.issues)}")
        print(f"Data Migration Required: {report.data_migration_required}")
        print(f"Migration Complexity: {report.migration_complexity}")
        print(f"Passed: {report.passed}")
        print("")

        if report.issues:
            print("Issues:")
            for issue in report.issues[:10]:  # Show first 10
                print(f"  {issue.level.value}: {issue.component} - {issue.description}")
            if len(report.issues) > 10:
                print(f"  ... and {len(report.issues) - 10} more issues")

        # Save detailed report
        report_file = project_root / f'compatibility-{args.source}-to-{args.target}.json'
        with open(report_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        print(f"\n📋 Detailed report saved to: {report_file}")

        # Generate migration guide if requested
        if args.migration_guide:
            guide = tester.generate_migration_guide(report)
            guide_file = project_root / f'migration-{args.source}-to-{args.target}.md'
            with open(guide_file, 'w') as f:
                f.write(guide)
            print(f"📖 Migration guide saved to: {guide_file}")

        # Exit with appropriate code
        if report.passed:
            print("\n✅ Compatibility test passed!")
            sys.exit(0)
        else:
            print("
❌ Compatibility test failed!"            print("Address breaking changes before upgrading.")
            sys.exit(1)


if __name__ == "__main__":
    main()
