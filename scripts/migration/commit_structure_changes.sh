#!/bin/bash

# Script to commit the directory structure changes

set -e  # Exit on error

# Colors for output
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${GREEN}FedZK Project Structure Commit Script${RESET}"
echo -e "${YELLOW}This script helps commit the directory structure changes${RESET}"
echo ""

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo -e "${RED}Error: Not a git repository${RESET}"
    exit 1
fi

# Check if there are changes to commit
if [ -z "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}No changes to commit${RESET}"
    exit 0
fi

# Show the changes
echo -e "${YELLOW}Changes to be committed:${RESET}"
git status --short

# Ask for confirmation
echo ""
read -p "Do you want to commit these changes? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo -e "${RED}Aborting${RESET}"
    exit 1
fi

# Add all changes
git add .

# Commit with a message about the structure change
git commit -m "chore: reorganize project structure to use src/fedzk/ instead of fedzk/

- Move code from fedzk/ to src/fedzk/
- Add DEPRECATED_STRUCTURE.md to document the transition
- Update CI workflow to use the new structure
- Update warning in README.md"

echo -e "${GREEN}Changes committed successfully!${RESET}"
echo -e "${YELLOW}You can now push the changes to the remote repository${RESET}"
echo -e "git push origin master" 