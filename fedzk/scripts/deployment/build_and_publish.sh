#!/bin/bash
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting FedZK package build and publish process...${NC}"

# Ensure we're in the project root
cd "$(dirname "$0")/../../"
ROOT_DIR=$(pwd)
echo -e "${GREEN}Working from project root: ${ROOT_DIR}${NC}"

# Ensure dependencies are installed
echo -e "${YELLOW}Checking build dependencies...${NC}"
python -m pip install --upgrade pip build twine

# Clean previous builds
echo -e "${YELLOW}Cleaning previous builds...${NC}"
rm -rf dist/ build/ *.egg-info/

# Validate pyproject.toml
echo -e "${YELLOW}Validating pyproject.toml...${NC}"
if ! python -c "import tomli; tomli.load(open('pyproject.toml', 'rb'))"; then
    echo -e "${RED}Error: pyproject.toml is not valid TOML${NC}"
    exit 1
fi
echo -e "${GREEN}pyproject.toml is valid${NC}"

# Check if the version has already been published
VERSION=$(grep -E '^version\s*=\s*"[0-9]+\.[0-9]+\.[0-9]+"' pyproject.toml | grep -o '".*"' | tr -d '"')
echo -e "${YELLOW}Checking if version ${VERSION} is already on TestPyPI...${NC}"
if pip index versions --index-url https://test.pypi.org/simple/ fedzk | grep -q "${VERSION}"; then
    echo -e "${RED}Error: Version ${VERSION} is already published on TestPyPI${NC}"
    echo -e "${YELLOW}Consider incrementing the version in pyproject.toml${NC}"
    exit 1
fi

# Build the package
echo -e "${YELLOW}Building package...${NC}"
python -m build

# Validate the built package
echo -e "${YELLOW}Validating package...${NC}"
twine check dist/*

# Upload to TestPyPI
echo -e "${YELLOW}Uploading to TestPyPI...${NC}"
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

echo -e "${GREEN}Package published to TestPyPI successfully!${NC}"
echo -e "${YELLOW}To install the package, run:${NC}"
echo -e "${GREEN}pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple fedzk==${VERSION}${NC}"

# Suggest verification steps
echo -e "${YELLOW}Verify the package installation and CLI by running:${NC}"
echo -e "${GREEN}pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple fedzk==${VERSION}${NC}"
echo -e "${GREEN}fedzk --help${NC}" 