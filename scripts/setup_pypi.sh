#!/bin/bash

# Script to set up local environment for PyPI publishing
# This script helps developers configure their environment for manual PyPI publishing

set -e  # Exit on error

# Colors for output
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${GREEN}FEDzk PyPI Setup Script${RESET}"
echo -e "${YELLOW}This script helps you set up your environment for PyPI publishing${RESET}"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 first.${RESET}"
    exit 1
fi

# Check for pip
if ! command -v pip &> /dev/null; then
    echo -e "${RED}pip is not installed. Please install pip first.${RESET}"
    exit 1
fi

# Install required packages
echo -e "${YELLOW}Installing required packages...${RESET}"
pip install --upgrade pip build twine
echo -e "${GREEN}✓ Installed build and twine${RESET}"

# Check if ~/.pypirc exists
if [ ! -f ~/.pypirc ]; then
    echo -e "${YELLOW}Creating ~/.pypirc template...${RESET}"
    cat > ~/.pypirc << EOF
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-<your-api-token>

[testpypi]
username = __token__
password = pypi-<your-test-api-token>
repository = https://test.pypi.org/legacy/
EOF
    echo -e "${GREEN}✓ Created ~/.pypirc template${RESET}"
    echo -e "${YELLOW}Please edit ~/.pypirc and replace the token placeholders with your actual tokens${RESET}"
else
    echo -e "${YELLOW}~/.pypirc already exists. Please ensure it contains your PyPI tokens.${RESET}"
fi

# Create a helper script for publishing
echo -e "${YELLOW}Creating publish helper script...${RESET}"
cat > scripts/publish_manual.sh << EOF
#!/bin/bash

# Script to manually publish FEDzk to PyPI
# Usage: ./scripts/publish_manual.sh [test|prod]

set -e  # Exit on error

# Colors for output
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

TARGET="\$1"

if [ "\$TARGET" != "test" ] && [ "\$TARGET" != "prod" ]; then
    echo -e "\${RED}Usage: ./scripts/publish_manual.sh [test|prod]${RESET}"
    echo -e "\${YELLOW}Example: ./scripts/publish_manual.sh test${RESET}"
    exit 1
fi

# Clean up existing distributions
echo -e "\${YELLOW}Cleaning up old distribution files...${RESET}"
rm -rf dist/ build/ *.egg-info

# Build the package
echo -e "\${YELLOW}Building package...${RESET}"
python -m build
echo -e "\${GREEN}✓ Built package${RESET}"

# Check the built package
echo -e "\${YELLOW}Checking package...${RESET}"
twine check dist/*
echo -e "\${GREEN}✓ Package check passed${RESET}"

if [ "\$TARGET" == "test" ]; then
    # Upload to TestPyPI
    echo -e "\${YELLOW}Uploading to TestPyPI...${RESET}"
    twine upload --repository testpypi dist/*
    echo -e "\${GREEN}✓ Uploaded to TestPyPI${RESET}"
    echo -e "\${YELLOW}View at: https://test.pypi.org/project/fedzk/${RESET}"
else
    # Confirm before uploading to production PyPI
    echo -e "\${YELLOW}About to upload to PyPI (production)${RESET}"
    read -p "Are you sure you want to continue? (y/n) " -n 1 -r
    echo
    if [[ \$REPLY =~ ^[Yy]$ ]]; then
        echo -e "\${YELLOW}Uploading to PyPI...${RESET}"
        twine upload dist/*
        echo -e "\${GREEN}✓ Uploaded to PyPI${RESET}"
        echo -e "\${YELLOW}View at: https://pypi.org/project/fedzk/${RESET}"
    else
        echo -e "\${RED}Upload to PyPI canceled.${RESET}"
        exit 1
    fi
fi

echo -e "\${GREEN}Done!${RESET}"
EOF
chmod +x scripts/publish_manual.sh
echo -e "${GREEN}✓ Created scripts/publish_manual.sh${RESET}"

echo -e "\n${GREEN}PyPI setup complete!${RESET}"
echo -e "${YELLOW}Next steps:${RESET}"
echo -e "1. Edit ~/.pypirc with your actual PyPI tokens"
echo -e "2. Use scripts/publish_manual.sh to publish to TestPyPI or PyPI"
echo -e "3. Run: ./scripts/publish_manual.sh test (for TestPyPI)"
echo -e "4. Run: ./scripts/publish_manual.sh prod (for production PyPI)"
echo ""
echo -e "${YELLOW}Note: It's recommended to use the GitHub Actions workflow for publishing.${RESET}"
echo -e "${YELLOW}See docs/publishing.md for details on both manual and automated publishing.${RESET}"
echo "" 