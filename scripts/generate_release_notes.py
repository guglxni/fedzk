#!/usr/bin/env python3
"""
Generate release notes for FEDZK releases.
"""

import sys
import subprocess
from datetime import datetime
from pathlib import Path


def get_git_commits(since_tag=None):
    """Get git commits since the last tag."""
    cmd = ["git", "log", "--oneline", "--pretty=format:%h %s"]

    if since_tag:
        cmd.extend([f"{since_tag}..HEAD"])

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip().split('\n') if result.stdout else []


def get_prs_merged(since_tag=None):
    """Get pull requests merged since the last tag."""
    # This is a simplified version. In practice, you'd use GitHub API
    commits = get_git_commits(since_tag)
    prs = []

    for commit in commits:
        if 'Merge pull request' in commit or '#' in commit:
            prs.append(commit)

    return prs


def categorize_changes(commits):
    """Categorize commits by type."""
    categories = {
        'Features': [],
        'Bug Fixes': [],
        'Documentation': [],
        'Performance': [],
        'Security': [],
        'Breaking Changes': [],
        'Other': []
    }

    for commit in commits:
        commit_lower = commit.lower()

        if any(keyword in commit_lower for keyword in ['feat:', 'feature:', 'add:']):
            categories['Features'].append(commit)
        elif any(keyword in commit_lower for keyword in ['fix:', 'bug:', 'hotfix:']):
            categories['Bug Fixes'].append(commit)
        elif any(keyword in commit_lower for keyword in ['docs:', 'documentation:']):
            categories['Documentation'].append(commit)
        elif any(keyword in commit_lower for keyword in ['perf:', 'performance:', 'optimize:']):
            categories['Performance'].append(commit)
        elif any(keyword in commit_lower for keyword in ['security:', 'sec:']):
            categories['Security'].append(commit)
        elif any(keyword in commit_lower for keyword in ['breaking:', 'break:']):
            categories['Breaking Changes'].append(commit)
        else:
            categories['Other'].append(commit)

    return categories


def generate_release_notes(version, since_tag=None):
    """Generate release notes."""

    commits = get_git_commits(since_tag)
    categories = categorize_changes(commits)

    release_notes = f"""# FEDZK Release {version}

**Released:** {datetime.now().strftime('%Y-%m-%d')}

## Overview
Federated Learning with Zero-Knowledge Proofs - Enterprise-grade privacy-preserving ML.

## What's New

"""

    # Add categorized changes
    for category, changes in categories.items():
        if changes:
            release_notes += f"### {category}\n\n"
            for change in changes:
                # Clean up commit message
                commit_hash, message = change.split(' ', 1)
                release_notes += f"- {message} ([{commit_hash[:7]}](https://github.com/fedzk/fedzk/commit/{commit_hash}))\n"
            release_notes += "\n"

    # Add technical details
    release_notes += """## Technical Details

### Compatibility
- **Python:** 3.8, 3.9, 3.10, 3.11
- **Operating Systems:** Linux, macOS, Windows
- **Architectures:** x86_64, ARM64

### Dependencies
- Updated cryptography library to v3.4.0+
- Enhanced GPU support with CUDA 11.8
- Improved performance with optimized ZK circuits

## Installation

```bash
# Install from PyPI
pip install fedzk=={version}

# Install with GPU support
pip install fedzk[gpu]=={version}

# Install with development tools
pip install fedzk[dev]=={version}
```

## Docker

```bash
docker pull fedzk/fedzk:{version}
```

## Documentation
📖 [Full Documentation](https://docs.fedzk.io)

## Security
For security-related updates and patches, please see our [Security Policy](https://github.com/fedzk/fedzk/security/policy).

## Contributing
We welcome contributions! Please see our [Contributing Guide](https://github.com/fedzk/fedzk/blob/main/CONTRIBUTING.md).

## Support
- 📧 [Email Support](mailto:support@fedzk.io)
- 💬 [Community Forum](https://community.fedzk.io)
- 🐛 [Bug Reports](https://github.com/fedzk/fedzk/issues)

---

**Checksums:**
```
SHA256 checksums will be available in the release assets.
```

**Previous Release:** [{since_tag}](https://github.com/fedzk/fedzk/releases/tag/{since_tag}) | **All Releases**"""

    return release_notes


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python generate_release_notes.py <version>")
        sys.exit(1)

    version = sys.argv[1]

    # Find the previous tag
    result = subprocess.run(["git", "describe", "--tags", "--abbrev=0"],
                          capture_output=True, text=True)
    since_tag = result.stdout.strip() if result.returncode == 0 else None

    # Generate release notes
    release_notes = generate_release_notes(version, since_tag)

    # Save to file
    output_file = Path(f"RELEASE_NOTES_{version}.md")
    output_file.write_text(release_notes)

    print(f"✅ Release notes generated: {output_file}")
    print("\n" + "="*50)
    print(release_notes)


if __name__ == "__main__":
    main()
