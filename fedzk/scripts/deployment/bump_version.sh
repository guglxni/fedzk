#!/bin/bash
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default bump type is patch
BUMP_TYPE=${1:-patch}

# Validate bump type
if [[ ! $BUMP_TYPE =~ ^(major|minor|patch)$ ]]; then
    echo -e "${RED}Invalid bump type. Use major, minor, or patch.${NC}"
    echo -e "${YELLOW}Example: $0 minor${NC}"
    exit 1
fi

# Ensure we're in the project root
cd "$(dirname "$0")/../../"
ROOT_DIR=$(pwd)
echo -e "${GREEN}Working from project root: ${ROOT_DIR}${NC}"

# Get the current version from pyproject.toml
CURRENT_VERSION=$(grep -E '^version\s*=\s*"[0-9]+\.[0-9]+\.[0-9]+"' pyproject.toml | grep -o '".*"' | tr -d '"')
echo -e "${YELLOW}Current version: ${CURRENT_VERSION}${NC}"

# Split the version into major, minor, and patch
IFS='.' read -r -a VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR=${VERSION_PARTS[0]}
MINOR=${VERSION_PARTS[1]}
PATCH=${VERSION_PARTS[2]}

# Bump the appropriate part
case $BUMP_TYPE in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
esac

# Construct the new version
NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"
echo -e "${GREEN}New version: ${NEW_VERSION}${NC}"

# Update version in pyproject.toml
echo -e "${YELLOW}Updating version in pyproject.toml...${NC}"
sed -i.bak "s/^version = \"${CURRENT_VERSION}\"/version = \"${NEW_VERSION}\"/" pyproject.toml
rm pyproject.toml.bak

echo -e "${GREEN}Updated version in pyproject.toml from ${CURRENT_VERSION} to ${NEW_VERSION}${NC}"

# Create a new git tag
echo -e "${YELLOW}Creating git tag v${NEW_VERSION}...${NC}"
git add pyproject.toml
git commit -m "chore: bump version to ${NEW_VERSION}"
git tag -a "v${NEW_VERSION}" -m "Version ${NEW_VERSION}"

echo -e "${GREEN}Version bump completed!${NC}"
echo -e "${YELLOW}To push the changes and trigger the release workflow, run:${NC}"
echo -e "${GREEN}git push && git push --tags${NC}" 