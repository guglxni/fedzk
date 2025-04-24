#!/bin/bash

echo "Starting project structure reorganization..."

# Create backup of current src directory
echo "Creating backup of current src directory..."
cp -r src src_backup

# Ensure the target directory exists
mkdir -p src/fedzk

# Copy all components from the comprehensive implementation
echo "Copying components from fedzk/src/fedzk to src/fedzk..."
cp -r fedzk/src/fedzk/* src/fedzk/

# Make sure the version numbers are consistent
VERSION_LINE=$(grep "__version__" src/fedzk/__init__.py)
echo "Updating version information..."
sed -i "" "s/__version__ = .*/$VERSION_LINE/" src/fedzk/__init__.py

# Remove duplicate directory structure
echo "Cleaning up duplicate directory structure..."
echo "NOTE: This will keep the fedzk/ directory for now, but mark it as deprecated."
echo "      To completely remove it, uncomment the rm -rf command below"
# rm -rf fedzk/

# Create a deprecation notice
echo "Creating deprecation notice..."
cat > DEPRECATED_STRUCTURE.md << 'EOL'
# Deprecated Directory Structure

The `fedzk/` directory at the project root is deprecated and should not be used.

The correct Python package structure is:
- `src/fedzk/` - This is the official Python package

The project will be built from the `src/` directory as specified in `pyproject.toml`.

This notice was created on April 24, 2025.
EOL

# Move pyproject.toml to the correct location if it doesn't exist
if [ ! -f pyproject.toml ]; then
  echo "Moving pyproject.toml to project root..."
  cp fedzk/build/packaging/pyproject.toml ./
fi

echo "Project reorganization complete!"
echo "Please review the changes and then run your tests to ensure everything works correctly."
echo "To completely remove the deprecated structure, uncomment the rm -rf line in this script and run it again." 