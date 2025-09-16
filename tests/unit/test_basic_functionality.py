"""
Basic functionality tests for FEDZK
These tests validate core functionality without requiring external dependencies.
"""

import unittest
import sys
import os
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestBasicFunctionality(unittest.TestCase):
    """Test basic FEDZK functionality."""

    def test_package_structure_exists(self):
        """Test that the package structure exists."""
        src_dir = Path(__file__).parent.parent.parent / 'src'
        fedzk_dir = src_dir / 'fedzk'
        
        self.assertTrue(src_dir.exists(), "src directory should exist")
        self.assertTrue(fedzk_dir.exists(), "fedzk package directory should exist")
        self.assertTrue((fedzk_dir / '__init__.py').exists(), "fedzk __init__.py should exist")

    def test_core_modules_exist(self):
        """Test that core modules exist."""
        src_dir = Path(__file__).parent.parent.parent / 'src'
        fedzk_dir = src_dir / 'fedzk'
        
        core_modules = [
            'coordinator',
            'client', 
            'prover',
            'zk',
            'validation',
            'monitoring',
            'compliance'
        ]
        
        for module in core_modules:
            module_path = fedzk_dir / module
            self.assertTrue(module_path.exists(), f"{module} module should exist")
            self.assertTrue((module_path / '__init__.py').exists(), f"{module} __init__.py should exist")

    def test_circuit_files_exist(self):
        """Test that circuit files exist."""
        circuits_dir = Path(__file__).parent.parent.parent / 'src' / 'fedzk' / 'zk' / 'circuits'
        
        self.assertTrue(circuits_dir.exists(), "circuits directory should exist")
        
        # Check for some circuit files
        circuit_files = list(circuits_dir.glob("*.circom"))
        self.assertGreater(len(circuit_files), 0, "Should have at least one .circom file")
        
        # Check for cryptographic keys
        key_files = list(circuits_dir.glob("*.zkey"))
        self.assertGreater(len(key_files), 0, "Should have at least one .zkey file")

    def test_configuration_files_exist(self):
        """Test that configuration files exist."""
        root_dir = Path(__file__).parent.parent.parent
        
        config_files = [
            'pyproject.toml',
            'README.md',
            'LICENSE',
            'CONTRIBUTING.md'
        ]
        
        for config_file in config_files:
            file_path = root_dir / config_file
            self.assertTrue(file_path.exists(), f"{config_file} should exist")

    def test_docker_files_exist(self):
        """Test that Docker files exist."""
        root_dir = Path(__file__).parent.parent.parent
        
        docker_files = [
            'Dockerfile',
            'docker/docker-compose.yml'
        ]
        
        for docker_file in docker_files:
            file_path = root_dir / docker_file
            self.assertTrue(file_path.exists(), f"{docker_file} should exist")

    def test_helm_charts_exist(self):
        """Test that Helm charts exist."""
        helm_dir = Path(__file__).parent.parent.parent / 'helm' / 'fedzk'
        
        self.assertTrue(helm_dir.exists(), "Helm chart directory should exist")
        self.assertTrue((helm_dir / 'Chart.yaml').exists(), "Chart.yaml should exist")
        self.assertTrue((helm_dir / 'values.yaml').exists(), "values.yaml should exist")
        
        templates_dir = helm_dir / 'templates'
        self.assertTrue(templates_dir.exists(), "templates directory should exist")

    def test_documentation_exists(self):
        """Test that documentation exists."""
        docs_dir = Path(__file__).parent.parent.parent / 'docs'
        
        self.assertTrue(docs_dir.exists(), "docs directory should exist")
        
        # Check for key documentation files
        doc_files = [
            'guides/getting-started.md',
            'guides/deployment.md',
            'api/openapi.yaml'
        ]
        
        for doc_file in doc_files:
            file_path = docs_dir / doc_file
            self.assertTrue(file_path.exists(), f"Documentation file {doc_file} should exist")

    def test_scripts_exist(self):
        """Test that CI/CD scripts exist."""
        scripts_dir = Path(__file__).parent.parent.parent / 'scripts'
        
        self.assertTrue(scripts_dir.exists(), "scripts directory should exist")
        
        # Check for CI scripts
        ci_scripts = [
            'ci/zk_toolchain_validator.py',
            'generate_performance_report.py'
        ]
        
        for script in ci_scripts:
            script_path = scripts_dir / script
            self.assertTrue(script_path.exists(), f"Script {script} should exist")

    def test_test_structure_exists(self):
        """Test that test structure is complete."""
        tests_dir = Path(__file__).parent.parent
        
        test_dirs = [
            'unit',
            'integration', 
            'e2e',
            'performance',
            'security',
            'compliance'
        ]
        
        for test_dir in test_dirs:
            dir_path = tests_dir / test_dir
            self.assertTrue(dir_path.exists(), f"Test directory {test_dir} should exist")


if __name__ == '__main__':
    unittest.main()
