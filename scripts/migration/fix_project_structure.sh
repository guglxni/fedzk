#!/bin/bash

# Script to migrate code from deprecated fedzk/ directory to src/fedzk/

set -e  # Exit on error

# Colors for output
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${GREEN}FedZK Project Structure Migration Script${RESET}"
echo -e "${YELLOW}This script helps migrate code from the deprecated fedzk/ directory to src/fedzk/${RESET}"
echo ""

# Check if both directories exist
if [ ! -d "fedzk" ]; then
    echo -e "${RED}Error: fedzk/ directory not found${RESET}"
    exit 1
fi

if [ ! -d "src/fedzk" ]; then
    echo -e "${YELLOW}Creating src/fedzk/ directory structure...${RESET}"
    mkdir -p src/fedzk
fi

# Create __init__.py file if it doesn't exist
if [ ! -f "src/fedzk/__init__.py" ]; then
    echo -e "${YELLOW}Creating src/fedzk/__init__.py file...${RESET}"
    cat > src/fedzk/__init__.py << EOF
"""
FedZK: Secure Federated Learning with Zero-Knowledge Proofs
Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
"""

__version__ = "1.0.0"
EOF
fi

# Function to migrate a single file
migrate_file() {
    local src_file=$1
    local dest_file=$2
    
    # Check if destination file already exists
    if [ -f "$dest_file" ]; then
        # Compare the files
        if diff -q "$src_file" "$dest_file" >/dev/null; then
            echo -e "${GREEN}✓ Files are identical: $src_file${RESET}"
        else
            echo -e "${YELLOW}! Files differ: $src_file${RESET}"
            echo -e "   Options:"
            echo -e "   [1] Keep src/fedzk version"
            echo -e "   [2] Replace with fedzk version"
            echo -e "   [3] View diff"
            echo -e "   [4] Skip this file"
            
            read -p "   Select option [1-4]: " choice
            
            case $choice in
                1)
                    echo -e "${GREEN}Keeping src/fedzk version${RESET}"
                    ;;
                2)
                    echo -e "${YELLOW}Replacing with fedzk version${RESET}"
                    cp "$src_file" "$dest_file"
                    echo -e "${GREEN}✓ File migrated: $src_file -> $dest_file${RESET}"
                    ;;
                3)
                    echo -e "${YELLOW}Diff between files:${RESET}"
                    diff -u "$dest_file" "$src_file" | less
                    # Ask again after showing diff
                    migrate_file "$src_file" "$dest_file"
                    ;;
                4)
                    echo -e "${YELLOW}Skipping file${RESET}"
                    ;;
                *)
                    echo -e "${RED}Invalid option${RESET}"
                    migrate_file "$src_file" "$dest_file"
                    ;;
            esac
        fi
    else
        # Destination file doesn't exist, copy it
        mkdir -p "$(dirname "$dest_file")"
        cp "$src_file" "$dest_file"
        echo -e "${GREEN}✓ File migrated: $src_file -> $dest_file${RESET}"
    fi
}

# Function to migrate a directory
migrate_directory() {
    local src_dir=$1
    local dest_dir=$2
    
    echo -e "${YELLOW}Checking directory: $src_dir${RESET}"
    
    # Create destination directory if it doesn't exist
    mkdir -p "$dest_dir"
    
    # Create __init__.py file if it doesn't exist
    if [ ! -f "$dest_dir/__init__.py" ]; then
        echo -e "${YELLOW}Creating $dest_dir/__init__.py file...${RESET}"
        touch "$dest_dir/__init__.py"
    fi
    
    # Find all Python files in the source directory
    find "$src_dir" -type f -name "*.py" | while read src_file; do
        # Get the relative path
        rel_path="${src_file#$src_dir/}"
        dest_file="$dest_dir/$rel_path"
        
        migrate_file "$src_file" "$dest_file"
    done
    
    # Recursively migrate subdirectories
    find "$src_dir" -type d -not -path "$src_dir" | while read subdir; do
        # Get the relative path
        rel_path="${subdir#$src_dir/}"
        dest_subdir="$dest_dir/$rel_path"
        
        # Skip if destination directory already exists
        if [ ! -d "$dest_subdir" ]; then
            mkdir -p "$dest_subdir"
            echo -e "${GREEN}✓ Created directory: $dest_subdir${RESET}"
        fi
        
        # Create __init__.py file if it doesn't exist
        if [ ! -f "$dest_subdir/__init__.py" ]; then
            echo -e "${YELLOW}Creating $dest_subdir/__init__.py file...${RESET}"
            touch "$dest_subdir/__init__.py"
        fi
    done
}

# Migrate the main directories
echo -e "${YELLOW}Starting migration process...${RESET}"

# Get list of subdirectories in fedzk/
subdirs=$(find fedzk -maxdepth 1 -type d -not -path "fedzk")

for subdir in $subdirs; do
    # Get the base directory name
    base_name=$(basename "$subdir")
    dest_dir="src/fedzk/$base_name"
    
    migrate_directory "$subdir" "$dest_dir"
done

echo -e "\n${GREEN}Migration complete!${RESET}"
echo -e "${YELLOW}Important next steps:${RESET}"
echo -e "1. Check that all files were migrated correctly"
echo -e "2. Update import statements in your code"
echo -e "3. Run tests to ensure everything works properly"
echo -e "4. When ready, you can delete the deprecated fedzk/ directory"
echo -e "   (But keep it until you're sure everything works with src/fedzk/)"
echo -e ""
echo -e "${GREEN}Remember: Always use src/fedzk/ for new development!${RESET}" 