#!/bin/bash

# Script to clean up and organize the FedZK repository

set -e  # Exit on error

# Colors for output
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${GREEN}FedZK Repository Cleanup Script${RESET}"
echo -e "${YELLOW}This script cleans up and organizes the repository files${RESET}"
echo ""

# Create directories if they don't exist
mkdir -p build/artifacts
mkdir -p build/temp
mkdir -p docs/examples
mkdir -p scripts/deployment
mkdir -p scripts/utils

# Clean up temporary files and directories
echo -e "${YELLOW}Cleaning up temporary files...${RESET}"

# Remove macOS system files
find . -name ".DS_Store" -delete
find . -name "._*" -delete
find . -name ".!*" -delete
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Move benchmark results to build/artifacts
echo -e "${YELLOW}Moving benchmark results to build/artifacts...${RESET}"
for file in *benchmark_results*; do
    if [ -f "$file" ]; then
        echo -e "Moving $file to build/artifacts/"
        mv "$file" build/artifacts/
    fi
done

# Clean up setup scripts - keep only the ones we need
echo -e "${YELLOW}Organizing setup scripts...${RESET}"
if [ -f "setup_zk.sh" ] && [ -f "setup_minimal.sh" ]; then
    if cmp -s "setup_zk.sh" "setup_minimal.sh"; then
        echo -e "setup_zk.sh and setup_minimal.sh are identical, keeping only setup_zk.sh"
        rm setup_minimal.sh
    else
        echo -e "Moving setup scripts to scripts/"
        mv setup_zk.sh scripts/
        mv setup_minimal.sh scripts/
    fi
elif [ -f "setup_zk.sh" ]; then
    echo -e "Moving setup_zk.sh to scripts/"
    mv setup_zk.sh scripts/
elif [ -f "setup_minimal.sh" ]; then
    echo -e "Moving setup_minimal.sh to scripts/"
    mv setup_minimal.sh scripts/
fi

# Move .sym files to build/artifacts
echo -e "${YELLOW}Moving symbol files to build/artifacts...${RESET}"
for file in *.sym; do
    if [ -f "$file" ] && [ "$file" != "*.sym" ]; then
        echo -e "Moving $file to build/artifacts/"
        mv "$file" build/artifacts/
    fi
done

# Move .circom files to circuits
echo -e "${YELLOW}Moving circuit files to circuits/...${RESET}"
for file in *.circom; do
    if [ -f "$file" ] && [ "$file" != "*.circom" ]; then
        echo -e "Moving $file to circuits/"
        mv "$file" circuits/
    fi
done

# Move build_benchmark.sh to scripts
if [ -f "build_benchmark.sh" ]; then
    echo -e "${YELLOW}Moving build_benchmark.sh to scripts/...${RESET}"
    mv build_benchmark.sh scripts/
fi

# Move organize_project.sh to scripts if it exists
if [ -f "organize_project.sh" ]; then
    echo -e "${YELLOW}Moving organize_project.sh to scripts/...${RESET}"
    mv organize_project.sh scripts/
fi

# Handle src_backup directory - keep it for now but add a README explaining it
if [ -d "src_backup" ]; then
    echo -e "${YELLOW}Adding README to src_backup/...${RESET}"
    cat > src_backup/README.md << EOF
# Backup of Source Directory

This directory contains a backup of the source code from a previous structure.
It's kept for reference during the transition to the new directory structure.

**This directory will be removed after the transition is complete.**
EOF
fi

# Create a .gitignore specifically for build artifacts if it doesn't exist
if [ ! -f "build/.gitignore" ]; then
    echo -e "${YELLOW}Creating .gitignore for build directory...${RESET}"
    cat > build/.gitignore << EOF
# Ignore all files in this directory
*
# Except for this .gitignore file
!.gitignore
# And except for the artifacts directory
!artifacts/
# But do ignore the contents of artifacts
artifacts/*
# Except for .gitkeep
!artifacts/.gitkeep
EOF

    # Create .gitkeep to preserve artifacts directory
    mkdir -p build/artifacts
    touch build/artifacts/.gitkeep
fi

# Add GitHub issue templates if they don't exist
if [ ! -d ".github/ISSUE_TEMPLATE" ]; then
    echo -e "${YELLOW}Creating GitHub issue templates...${RESET}"
    mkdir -p .github/ISSUE_TEMPLATE
    
    cat > .github/ISSUE_TEMPLATE/bug_report.md << EOF
---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Install '...'
2. Run command '....'
3. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment:**
 - OS: [e.g. Ubuntu 22.04]
 - Python Version: [e.g. 3.10]
 - FedZK Version: [e.g. 1.0.0]

**Additional context**
Add any other context about the problem here.
EOF

    cat > .github/ISSUE_TEMPLATE/feature_request.md << EOF
---
name: Feature request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: enhancement
assignees: ''

---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
EOF
fi

# Clean up the migration scripts, move them to scripts/migration
echo -e "${YELLOW}Moving migration scripts to scripts/migration/...${RESET}"
mkdir -p scripts/migration
if [ -f "fix_project_structure.sh" ]; then
    mv fix_project_structure.sh scripts/migration/
fi
if [ -f "commit_structure_changes.sh" ]; then
    mv commit_structure_changes.sh scripts/migration/
fi

echo -e "\n${GREEN}Repository cleanup complete!${RESET}"
echo -e "${YELLOW}Next steps:${RESET}"
echo -e "1. Review the changes"
echo -e "2. Run the tests to ensure everything works properly"
echo -e "3. Commit and push the changes" 