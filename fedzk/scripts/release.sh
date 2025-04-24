#!/bin/bash
# FedZK Release Script

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

# Default version if not provided
VERSION=${1:-"0.1.0"}
VERSION_TAG="v$VERSION"

echo -e "${YELLOW}Starting release process for FedZK ${VERSION_TAG}${NC}"

# Check if we're in the repo root
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must run from the repository root directory (containing pyproject.toml)${NC}"
    exit 1
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${RED}Error: There are uncommitted changes. Please commit or stash them before releasing.${NC}"
    git status
    exit 1
fi

# Pull latest changes from main branch
echo -e "${YELLOW}Pulling latest changes from main branch...${NC}"
git checkout main
git pull origin main

# Update version in pyproject.toml if needed
echo -e "${YELLOW}Checking version in pyproject.toml...${NC}"
CURRENT_VERSION=$(grep -m 1 'version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
if [ "$CURRENT_VERSION" != "$VERSION" ]; then
    echo -e "${YELLOW}Updating version in pyproject.toml from $CURRENT_VERSION to $VERSION${NC}"
    sed -i'' -e "s/version = \"$CURRENT_VERSION\"/version = \"$VERSION\"/" pyproject.toml
    git add pyproject.toml
    git commit -m "chore: bump version to $VERSION"
fi

# Make sure CHANGELOG.md is up to date
echo -e "${YELLOW}Checking CHANGELOG.md...${NC}"
if ! grep -q "\[$VERSION\]" CHANGELOG.md; then
    echo -e "${RED}Warning: Version $VERSION not found in CHANGELOG.md"
    echo -e "Please update the CHANGELOG.md before continuing.${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create and push the tag
echo -e "${YELLOW}Creating tag ${VERSION_TAG}...${NC}"
git tag -a "$VERSION_TAG" -m "Release $VERSION_TAG"

echo -e "${YELLOW}Pushing commits to origin...${NC}"
git push origin main

echo -e "${YELLOW}Pushing tag to origin...${NC}"
git push origin "$VERSION_TAG"

echo -e "${GREEN}Release $VERSION_TAG complete!${NC}"
echo -e "${GREEN}The GitHub Actions workflow will now:${NC}"
echo -e "${GREEN}1. Run tests${NC}"
echo -e "${GREEN}2. Build the package${NC}"
echo -e "${GREEN}3. Publish to PyPI${NC}"
echo -e "${GREEN}4. Create a GitHub release${NC}"

echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Monitor the GitHub Actions workflow: https://github.com/aaryanguglani/fedzk/actions"
echo -e "2. Check the GitHub release page: https://github.com/aaryanguglani/fedzk/releases"
echo -e "3. Verify the package is available on PyPI: https://pypi.org/project/fedzk/" 