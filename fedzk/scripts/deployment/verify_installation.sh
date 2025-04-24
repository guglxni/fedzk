#!/bin/bash
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Verifying FedZK package installation...${NC}"

# Create a temporary virtual environment
TEMP_DIR=$(mktemp -d)
echo -e "${YELLOW}Creating temporary environment in ${TEMP_DIR}${NC}"

python -m venv "${TEMP_DIR}/venv"
source "${TEMP_DIR}/venv/bin/activate"

# Get the latest version from TestPyPI
VERSION=$(curl -s https://test.pypi.org/pypi/fedzk/json | python -c "import sys, json; print(json.load(sys.stdin)['info']['version'])")
echo -e "${YELLOW}Latest version on TestPyPI: ${VERSION}${NC}"

# Install the package from TestPyPI
echo -e "${YELLOW}Installing fedzk ${VERSION} from TestPyPI...${NC}"
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "fedzk==${VERSION}"

# Verify basic CLI functionality
echo -e "${YELLOW}Verifying CLI functionality...${NC}"
echo -e "${GREEN}Running: fedzk --help${NC}"
fedzk --help

# Verify CLI commands
echo -e "${YELLOW}Verifying CLI commands...${NC}"
echo -e "${GREEN}Running: fedzk --version${NC}"
fedzk --version | grep "${VERSION}" && echo -e "${GREEN}Version matches: ${VERSION}${NC}" || (echo -e "${RED}Version mismatch!${NC}" && exit 1)

# Try running a few commands to verify functionality
echo -e "${YELLOW}Verifying CLI command functionality...${NC}"

# Test 'setup' command
echo -e "${GREEN}Testing 'setup' command help...${NC}"
fedzk setup --help

# Test 'benchmark' command
echo -e "${GREEN}Testing 'benchmark' command help...${NC}"
fedzk benchmark --help

# Clean up
deactivate
rm -rf "${TEMP_DIR}"

echo -e "${GREEN}Verification completed successfully!${NC}"
echo -e "${YELLOW}The FedZK package has been properly installed and the CLI is functional.${NC}" 