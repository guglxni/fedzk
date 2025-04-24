#!/bin/bash

echo "⚠️ DEPRECATED SCRIPT WARNING ⚠️"
echo "This script has been deprecated as it creates a non-standard project structure."
echo "Please use the fix_project_structure.sh script instead to maintain proper Python package structure."
echo ""
echo "The current recommended structure is:"
echo "- src/fedzk/ - Main Python package"
echo "- tests/ - Test suite"
echo "- docs/ - Documentation"
echo "- examples/ - Usage examples"
echo ""
echo "For more information, see docs/project_structure.md"
echo ""
echo "Do you want to proceed with this deprecated script anyway? (y/N)"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    # Original script content follows
    
    # Create directories if they don't exist
    mkdir -p fedzk/client
    mkdir -p fedzk/prover
    mkdir -p fedzk/mpc
    mkdir -p fedzk/coordinator
    mkdir -p fedzk/docs
    mkdir -p fedzk/tests
    mkdir -p fedzk/examples
    mkdir -p fedzk/scripts
    mkdir -p fedzk/benchmark
    mkdir -p fedzk/build/packaging
    mkdir -p fedzk/artifacts
    mkdir -p fedzk/src/fedzk
    mkdir -p fedzk/.github
    
    # Move files to appropriate directories, suppressing errors for files that don't exist
    mv client/* fedzk/client/ 2>/dev/null
    mv prover/* fedzk/prover/ 2>/dev/null
    mv mpc/* fedzk/mpc/ 2>/dev/null
    mv coordinator/* fedzk/coordinator/ 2>/dev/null
    mv docs/* fedzk/docs/ 2>/dev/null
    mv tests/* fedzk/tests/ 2>/dev/null
    mv examples/* fedzk/examples/ 2>/dev/null
    mv scripts/* fedzk/scripts/ 2>/dev/null
    mv benchmark/* fedzk/benchmark/ 2>/dev/null
    mv build/* fedzk/build/ 2>/dev/null
    mv artifacts/* fedzk/artifacts/ 2>/dev/null
    mv src/* fedzk/src/ 2>/dev/null
    mv .github/* fedzk/.github/ 2>/dev/null
    
    # Move root level configuration files
    cp .gitignore fedzk/ 2>/dev/null
    cp README.md fedzk/ 2>/dev/null
    
    # Clean up empty directories
    rmdir client/ 2>/dev/null
    rmdir prover/ 2>/dev/null
    rmdir mpc/ 2>/dev/null
    rmdir coordinator/ 2>/dev/null
    rmdir docs/ 2>/dev/null
    rmdir tests/ 2>/dev/null
    rmdir examples/ 2>/dev/null
    rmdir scripts/ 2>/dev/null
    rmdir benchmark/ 2>/dev/null
    rmdir build/ 2>/dev/null
    rmdir artifacts/ 2>/dev/null
    rmdir src/ 2>/dev/null
    rmdir .github/ 2>/dev/null
    
    echo "Project files organized successfully!"
    echo ""
    echo "⚠️ WARNING: This script has created a non-standard project structure."
    echo "Please see docs/project_structure.md for information on the recommended project structure."
else
    echo "Operation cancelled. Please use fix_project_structure.sh instead."
fi 