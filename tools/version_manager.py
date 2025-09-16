#!/usr/bin/env python3
"""
FEDZK Semantic Version Manager

Comprehensive semantic versioning system for FEDZK.
Implements version management, compatibility checking, and release automation.
"""

import os
import sys
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import semver


class VersionComponent(Enum):
    """Semantic version components."""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    PRE_RELEASE = "pre_release"
    BUILD = "build"


class ChangeType(Enum):
    """Types of changes that affect versioning."""
    BREAKING = "breaking"          # Breaking changes (major version bump)
    FEATURE = "feature"            # New features (minor version bump)
    FIX = "fix"                   # Bug fixes (patch version bump)
    DOCS = "docs"                 # Documentation changes (patch version bump)
    STYLE = "style"               # Code style changes (patch version bump)
    REFACTOR = "refactor"          # Code refactoring (patch version bump)
    TEST = "test"                 # Test changes (patch version bump)
    BUILD = "build"               # Build system changes (patch version bump)
    CI = "ci"                     # CI/CD changes (patch version bump)
    CHORE = "chore"               # Maintenance changes (patch version bump)


@dataclass
class VersionInfo:
    """Represents semantic version information."""

    major: int
    minor: int
    patch: int
    pre_release: Optional[str] = None
    build: Optional[str] = None

    def __str__(self) -> str:
        """Convert to semantic version string."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_release:
            version += f"-{self.pre_release}"
        if self.build:
            version += f"+{self.build}"
        return version

    def bump(self, component: VersionComponent, pre_release: Optional[str] = None) -> 'VersionInfo':
        """Bump version component."""
        if component == VersionComponent.MAJOR:
            return VersionInfo(
                major=self.major + 1,
                minor=0,
                patch=0,
                pre_release=pre_release
            )
        elif component == VersionComponent.MINOR:
            return VersionInfo(
                major=self.major,
                minor=self.minor + 1,
                patch=0,
                pre_release=pre_release
            )
        elif component == VersionComponent.PATCH:
            return VersionInfo(
                major=self.major,
                minor=self.minor,
                patch=self.patch + 1,
                pre_release=pre_release
            )
        elif component == VersionComponent.PRE_RELEASE:
            return VersionInfo(
                major=self.major,
                minor=self.minor,
                patch=self.patch,
                pre_release=pre_release
            )
        else:
            return self

    @classmethod
    def from_string(cls, version_str: str) -> 'VersionInfo':
        """Create VersionInfo from semantic version string."""
        try:
            # Parse with semver library if available
            parsed = semver.VersionInfo.parse(version_str)

            return cls(
                major=parsed.major,
                minor=parsed.minor,
                patch=parsed.patch,
                pre_release=parsed.prerelease if parsed.prerelease else None,
                build=parsed.build if parsed.build else None
            )

        except Exception:
            # Fallback parsing
            match = re.match(r'(\d+)\.(\d+)\.(\d+)(?:-([^+\s]+))?(?:\+(.+))?', version_str)
            if match:
                return cls(
                    major=int(match.group(1)),
                    minor=int(match.group(2)),
                    patch=int(match.group(3)),
                    pre_release=match.group(4),
                    build=match.group(5)
                )
            else:
                raise ValueError(f"Invalid semantic version: {version_str}")


@dataclass
class VersionChange:
    """Represents a version-affecting change."""

    commit_hash: str
    change_type: ChangeType
    description: str
    breaking: bool = False
    scope: Optional[str] = None

    def affects_version(self) -> bool:
        """Check if this change affects versioning."""
        return self.change_type in [ChangeType.BREAKING, ChangeType.FEATURE, ChangeType.FIX]


@dataclass
class VersionAnalysis:
    """Analysis of changes for version bumping."""

    current_version: VersionInfo
    changes: List[VersionChange] = field(default_factory=list)
    recommended_bump: VersionComponent = VersionComponent.PATCH
    breaking_changes: List[VersionChange] = field(default_factory=list)
    features: List[VersionChange] = field(default_factory=list)
    fixes: List[VersionChange] = field(default_factory=list)

    def analyze_changes(self):
        """Analyze changes to determine version bump."""
        for change in self.changes:
            if change.breaking or change.change_type == ChangeType.BREAKING:
                self.breaking_changes.append(change)
            elif change.change_type == ChangeType.FEATURE:
                self.features.append(change)
            elif change.change_type == ChangeType.FIX:
                self.fixes.append(change)

        # Determine recommended bump
        if self.breaking_changes:
            self.recommended_bump = VersionComponent.MAJOR
        elif self.features:
            self.recommended_bump = VersionComponent.MINOR
        else:
            self.recommended_bump = VersionComponent.PATCH

    def get_next_version(self) -> VersionInfo:
        """Get the next version based on analysis."""
        return self.current_version.bump(self.recommended_bump)


class FEDZKVersionManager:
    """Comprehensive FEDZK version management system."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.version_file = project_root / 'VERSION'
        self.changelog_file = project_root / 'CHANGELOG.md'

        # Conventional commit patterns
        self.commit_patterns = {
            ChangeType.BREAKING: [
                r'^break(?:ing)?:',
                r'^.*?!\s*:',  # Breaking change indicator
                r'^feat(?:\([^)]+\))?!:',  # Breaking feature
                r'^refactor(?:\([^)]+\))?!:',  # Breaking refactor
            ],
            ChangeType.FEATURE: [
                r'^feat(?:\([^)]+\))?:',
            ],
            ChangeType.FIX: [
                r'^fix(?:\([^)]+\))?:',
                r'^perf(?:\([^)]+\))?:',  # Performance fixes
            ],
            ChangeType.DOCS: [
                r'^docs(?:\([^)]+\))?:',
            ],
            ChangeType.STYLE: [
                r'^style(?:\([^)]+\))?:',
            ],
            ChangeType.REFACTOR: [
                r'^refactor(?:\([^)]+\))?:',
            ],
            ChangeType.TEST: [
                r'^test(?:\([^)]+\))?:',
            ],
            ChangeType.BUILD: [
                r'^build(?:\([^)]+\))?:',
            ],
            ChangeType.CI: [
                r'^ci(?:\([^)]+\))?:',
            ],
            ChangeType.CHORE: [
                r'^chore(?:\([^)]+\))?:',
            ]
        }

    def get_current_version(self) -> VersionInfo:
        """Get the current version from VERSION file."""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r') as f:
                    version_str = f.read().strip()
                    return VersionInfo.from_string(version_str)
            except Exception as e:
                print(f"Warning: Could not read version file: {e}")

        # Fallback to pyproject.toml
        pyproject_file = self.project_root / 'pyproject.toml'
        if pyproject_file.exists():
            try:
                import tomli
                with open(pyproject_file, 'rb') as f:
                    data = tomli.load(f)
                    version_str = data.get('project', {}).get('version', '0.1.0')
                    return VersionInfo.from_string(version_str)
            except Exception as e:
                print(f"Warning: Could not read pyproject.toml: {e}")

        # Default version
        return VersionInfo(major=0, minor=1, patch=0)

    def set_version(self, version: VersionInfo):
        """Set the version in VERSION file and pyproject.toml."""
        version_str = str(version)

        # Update VERSION file
        with open(self.version_file, 'w') as f:
            f.write(version_str + '\n')

        # Update pyproject.toml
        pyproject_file = self.project_root / 'pyproject.toml'
        if pyproject_file.exists():
            try:
                import tomli
                with open(pyproject_file, 'rb') as f:
                    data = tomli.load(f)

                if 'project' not in data:
                    data['project'] = {}
                data['project']['version'] = version_str

                import tomllib
                with open(pyproject_file, 'w') as f:
                    # This is a simplified write - in practice you'd preserve formatting
                    f.write('[build-system]\n')
                    f.write('requires = ["setuptools>=61.0", "wheel"]\n')
                    f.write('build-backend = "setuptools.build_meta"\n\n')
                    f.write('[project]\n')
                    f.write(f'version = "{version_str}"\n')
                    # Add other project data...

            except Exception as e:
                print(f"Warning: Could not update pyproject.toml: {e}")

        print(f"✅ Version updated to {version_str}")

    def analyze_commits(self, since_tag: Optional[str] = None) -> VersionAnalysis:
        """Analyze commits to determine version bump."""
        current_version = self.get_current_version()
        analysis = VersionAnalysis(current_version=current_version)

        try:
            # Get commits since last tag or from beginning
            cmd = ['git', 'log', '--oneline', '--pretty=format:%H %s']

            if since_tag:
                cmd.extend([f'{since_tag}..HEAD'])
            else:
                # Get all commits if no since_tag
                pass

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        commit_hash, message = line.split(' ', 1)
                        change = self._parse_commit_message(commit_hash, message)
                        if change:
                            analysis.changes.append(change)

            analysis.analyze_changes()

        except Exception as e:
            print(f"Warning: Could not analyze commits: {e}")

        return analysis

    def _parse_commit_message(self, commit_hash: str, message: str) -> Optional[VersionChange]:
        """Parse commit message to determine change type."""
        message = message.strip()

        # Check for conventional commit patterns
        for change_type, patterns in self.commit_patterns.items():
            for pattern in patterns:
                if re.match(pattern, message, re.IGNORECASE):
                    # Extract scope if present
                    scope_match = re.match(r'^\w+(?:\(([^)]+)\))?!?:', message)
                    scope = scope_match.group(1) if scope_match and scope_match.group(1) else None

                    # Check for breaking change indicators
                    breaking = '!' in message or 'BREAKING' in message.upper()

                    return VersionChange(
                        commit_hash=commit_hash,
                        change_type=change_type,
                        description=message,
                        breaking=breaking,
                        scope=scope
                    )

        # Default to chore for unrecognized patterns
        return VersionChange(
            commit_hash=commit_hash,
            change_type=ChangeType.CHORE,
            description=message
        )

    def bump_version(self, bump_type: Optional[VersionComponent] = None,
                    pre_release: Optional[str] = None) -> VersionInfo:
        """Bump version based on changes or specified type."""
        if bump_type:
            # Manual bump
            current_version = self.get_current_version()
            new_version = current_version.bump(bump_type, pre_release)
        else:
            # Analyze commits for automatic bump
            analysis = self.analyze_commits()
            new_version = analysis.get_next_version()

            print("📊 Version Analysis:")
            print(f"  Breaking changes: {len(analysis.breaking_changes)}")
            print(f"  Features: {len(analysis.features)}")
            print(f"  Fixes: {len(analysis.fixes)}")
            print(f"  Recommended bump: {analysis.recommended_bump.value}")

        self.set_version(new_version)
        return new_version

    def create_release_tag(self, version: VersionInfo, message: Optional[str] = None) -> bool:
        """Create a git tag for the release."""
        version_str = str(version)
        tag_name = f"v{version_str}"

        if message is None:
            message = f"Release {version_str}"

        try:
            # Create annotated tag
            subprocess.run(
                ['git', 'tag', '-a', tag_name, '-m', message],
                cwd=self.project_root,
                check=True
            )

            print(f"✅ Created release tag: {tag_name}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create release tag: {e}")
            return False

    def get_version_history(self) -> List[Dict[str, Any]]:
        """Get version history from git tags."""
        try:
            result = subprocess.run(
                ['git', 'tag', '--sort=-version:refname', '--format=%(refname:short) %(creatordate) %(subject)'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            if result.returncode == 0:
                history = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(' ', 2)
                        if len(parts) >= 2:
                            tag = parts[0]
                            date = ' '.join(parts[1:-1]) if len(parts) > 2 else ''
                            message = parts[-1] if len(parts) > 2 else ''

                            # Parse version from tag
                            if tag.startswith('v'):
                                try:
                                    version = VersionInfo.from_string(tag[1:])
                                    history.append({
                                        'version': str(version),
                                        'tag': tag,
                                        'date': date,
                                        'message': message
                                    })
                                except ValueError:
                                    pass

                return history

        except Exception as e:
            print(f"Warning: Could not get version history: {e}")

        return []

    def check_version_compatibility(self, version1: VersionInfo, version2: VersionInfo) -> Dict[str, Any]:
        """Check version compatibility between two versions."""
        compatibility = {
            'compatible': True,
            'breaking_change': False,
            'compatibility_level': 'patch',
            'warnings': []
        }

        # Major version changes are breaking
        if version1.major != version2.major:
            compatibility['compatible'] = False
            compatibility['breaking_change'] = True
            compatibility['compatibility_level'] = 'breaking'
            compatibility['warnings'].append("Major version change indicates breaking changes")

        # Minor version changes are backward compatible
        elif version1.minor != version2.minor:
            compatibility['compatibility_level'] = 'minor'
            if version2.minor > version1.minor:
                compatibility['warnings'].append("Minor version increase - check for new features")
            else:
                compatibility['warnings'].append("Minor version decrease - potential compatibility issues")

        # Patch versions should be fully compatible
        elif version1.patch != version2.patch:
            compatibility['compatibility_level'] = 'patch'
            compatibility['warnings'].append("Patch version change - bug fixes only")

        return compatibility

    def validate_version_string(self, version_str: str) -> bool:
        """Validate semantic version string."""
        try:
            VersionInfo.from_string(version_str)
            return True
        except ValueError:
            return False

    def suggest_next_version(self, include_pre_release: bool = False) -> VersionInfo:
        """Suggest next version based on commit analysis."""
        analysis = self.analyze_commits()
        next_version = analysis.get_next_version()

        if include_pre_release:
            # Add pre-release identifier
            next_version.pre_release = "rc.1"

        return next_version

    def generate_version_report(self) -> str:
        """Generate comprehensive version management report."""
        current_version = self.get_current_version()
        analysis = self.analyze_commits()
        history = self.get_version_history()

        report_lines = []

        report_lines.append("# FEDZK Version Management Report")
        report_lines.append("=" * 50)
        report_lines.append("")

        # Current version
        report_lines.append("## Current Version")
        report_lines.append(f"- Version: {current_version}")
        report_lines.append(f"- Major: {current_version.major}")
        report_lines.append(f"- Minor: {current_version.minor}")
        report_lines.append(f"- Patch: {current_version.patch}")
        if current_version.pre_release:
            report_lines.append(f"- Pre-release: {current_version.pre_release}")
        if current_version.build:
            report_lines.append(f"- Build: {current_version.build}")
        report_lines.append("")

        # Commit analysis
        report_lines.append("## Commit Analysis")
        report_lines.append(f"- Total commits analyzed: {len(analysis.changes)}")
        report_lines.append(f"- Breaking changes: {len(analysis.breaking_changes)}")
        report_lines.append(f"- Features: {len(analysis.features)}")
        report_lines.append(f"- Fixes: {len(analysis.fixes)}")
        report_lines.append(f"- Recommended bump: {analysis.recommended_bump.value}")
        report_lines.append("")

        # Next version suggestion
        next_version = analysis.get_next_version()
        report_lines.append("## Suggested Next Version")
        report_lines.append(f"- Next version: {next_version}")
        report_lines.append(f"- Bump type: {analysis.recommended_bump.value}")
        report_lines.append("")

        # Version history
        if history:
            report_lines.append("## Version History")
            for entry in history[:10]:  # Show last 10 versions
                report_lines.append(f"- **{entry['version']}** ({entry['date'][:10]}) - {entry['message']}")
            report_lines.append("")

        # Recommendations
        report_lines.append("## Recommendations")
        if analysis.breaking_changes:
            report_lines.append("⚠️ **Breaking Changes Detected**: Consider major version bump")
        elif analysis.features:
            report_lines.append("✨ **New Features Added**: Consider minor version bump")
        else:
            report_lines.append("🐛 **Bug Fixes Only**: Consider patch version bump")

        report_lines.append("")
        report_lines.append("### Version Management Best Practices")
        report_lines.append("- Follow semantic versioning (MAJOR.MINOR.PATCH)")
        report_lines.append("- Use conventional commits for automatic versioning")
        report_lines.append("- Test version compatibility before releases")
        report_lines.append("- Document breaking changes in release notes")
        report_lines.append("- Use pre-release versions for beta testing")
        report_lines.append("")

        return "\n".join(report_lines)


def main():
    """Main entry point for version manager."""
    import argparse

    parser = argparse.ArgumentParser(description="FEDZK Version Manager")
    parser.add_argument('command', choices=['current', 'bump', 'analyze', 'suggest', 'tag', 'report'])
    parser.add_argument('--bump-type', choices=['major', 'minor', 'patch'], help='Version bump type')
    parser.add_argument('--pre-release', help='Pre-release identifier')
    parser.add_argument('--message', help='Release message')
    parser.add_argument('--since', help='Analyze commits since this tag')

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    manager = FEDZKVersionManager(project_root)

    if args.command == 'current':
        version = manager.get_current_version()
        print(f"Current version: {version}")

    elif args.command == 'bump':
        bump_type = None
        if args.bump_type:
            bump_type = VersionComponent(args.bump_type)

        new_version = manager.bump_version(bump_type, args.pre_release)
        print(f"Version bumped to: {new_version}")

    elif args.command == 'analyze':
        analysis = manager.analyze_commits(args.since)
        print(f"Analyzed {len(analysis.changes)} commits")
        print(f"Breaking changes: {len(analysis.breaking_changes)}")
        print(f"Features: {len(analysis.features)}")
        print(f"Fixes: {len(analysis.fixes)}")
        print(f"Recommended bump: {analysis.recommended_bump.value}")

    elif args.command == 'suggest':
        next_version = manager.suggest_next_version(bool(args.pre_release))
        print(f"Suggested next version: {next_version}")

    elif args.command == 'tag':
        current_version = manager.get_current_version()
        if manager.create_release_tag(current_version, args.message):
            print(f"✅ Created release tag v{current_version}")
        else:
            print("❌ Failed to create release tag")
            sys.exit(1)

    elif args.command == 'report':
        report = manager.generate_version_report()
        print(report)

        # Save report
        report_file = project_root / 'version-report.md'
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n📋 Report saved to: {report_file}")


if __name__ == "__main__":
    main()
