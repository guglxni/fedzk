#!/usr/bin/env python3
"""
ZK Toolchain Validator for CI/CD
=================================

Validates ZK toolchain components and circuit compilation in CI environment.
Ensures all cryptographic operations are ready for production deployment.
"""

import subprocess
import sys
import os
from pathlib import Path
import json
from typing import Dict, List, Any, Tuple


class ZKToolchainValidator:
    """Comprehensive ZK toolchain validation for CI/CD."""

    def __init__(self, circuits_dir: Path = None, artifacts_dir: Path = None):
        """Initialize validator with directory paths."""
        self.circuits_dir = circuits_dir or Path("src/fedzk/zk/circuits")
        self.artifacts_dir = artifacts_dir or Path("src/fedzk/zk/circuits")
        self.validation_results = {
            'toolchain_check': False,
            'circuit_validation': [],
            'compilation_check': [],
            'artifact_verification': [],
            'security_validation': []
        }

    def validate_toolchain(self) -> bool:
        """Validate ZK toolchain installation and versions."""
        print("🔧 Validating ZK Toolchain Installation...")

        # Check Circom installation
        try:
            result = subprocess.run(['circom', '--version'],
                                  capture_output=True, text=True, check=True)
            circom_version = result.stdout.strip()
            print(f"✅ Circom: {circom_version}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Circom not found or not working")
            return False

        # Check SNARKjs installation
        try:
            result = subprocess.run(['snarkjs', '--version'],
                                  capture_output=True, text=True, check=True)
            snarkjs_version = result.stdout.strip()
            print(f"✅ SNARKjs: {snarkjs_version}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ SNARKjs not found or not working")
            return False

        # Check Node.js version (required for SNARKjs)
        try:
            result = subprocess.run(['node', '--version'],
                                  capture_output=True, text=True, check=True)
            node_version = result.stdout.strip()
            print(f"✅ Node.js: {node_version}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Node.js not found")
            return False

        self.validation_results['toolchain_check'] = True
        return True

    def validate_circuits(self) -> List[Dict[str, Any]]:
        """Validate all Circom circuits for syntax correctness."""
        print("🔍 Validating Circuit Syntax...")

        circuit_files = list(self.circuits_dir.glob("*.circom"))
        validation_results = []

        for circuit_file in circuit_files:
            circuit_name = circuit_file.name
            print(f"  Validating {circuit_name}...")

            try:
                # Basic syntax validation using Circom
                result = subprocess.run([
                    'circom', str(circuit_file),
                    '--r1cs', '--wasm', '--sym'
                ], capture_output=True, text=True, cwd=self.circuits_dir, timeout=30)

                if result.returncode == 0:
                    print(f"    ✅ {circuit_name} - Syntax valid")
                    validation_results.append({
                        'circuit': circuit_name,
                        'valid': True,
                        'error': None
                    })
                else:
                    print(f"    ❌ {circuit_name} - Syntax error: {result.stderr[:100]}...")
                    validation_results.append({
                        'circuit': circuit_name,
                        'valid': False,
                        'error': result.stderr[:200]
                    })

            except subprocess.TimeoutExpired:
                print(f"    ❌ {circuit_name} - Validation timeout")
                validation_results.append({
                    'circuit': circuit_name,
                    'valid': False,
                    'error': 'Timeout during validation'
                })
            except Exception as e:
                print(f"    ❌ {circuit_name} - Validation failed: {str(e)}")
                validation_results.append({
                    'circuit': circuit_name,
                    'valid': False,
                    'error': str(e)
                })

        self.validation_results['circuit_validation'] = validation_results
        return validation_results

    def validate_compilation_artifacts(self) -> List[Dict[str, Any]]:
        """Validate compilation artifacts exist and are valid."""
        print("📦 Validating Compilation Artifacts...")

        circuits = [
            'model_update',
            'model_update_secure',
            'model_update_quantized',
            'batch_verification',
            'sparse_gradients',
            'differential_privacy',
            'custom_constraints'
        ]

        artifact_results = []

        for circuit in circuits:
            artifacts = {
                'wasm': self.artifacts_dir / f"{circuit}.wasm",
                'r1cs': self.artifacts_dir / f"{circuit}.r1cs",
                'zkey': self.artifacts_dir / f"proving_key_{circuit}.zkey",
                'vkey': self.artifacts_dir / f"verification_key_{circuit}.json"
            }

            found_artifacts = {}
            for artifact_type, path in artifacts.items():
                if path.exists():
                    # Basic validation of file size and structure
                    size = path.stat().st_size
                    if size > 100:  # Reasonable minimum size
                        found_artifacts[artifact_type] = size
                        print(f"    ✅ {circuit}.{artifact_type} - {size} bytes")
                    else:
                        print(f"    ⚠️ {circuit}.{artifact_type} - Too small ({size} bytes)")
                else:
                    print(f"    ❌ {circuit}.{artifact_type} - Missing")

            # Validate JSON structure for verification keys
            vkey_valid = True
            if 'vkey' in found_artifacts:
                try:
                    with open(artifacts['vkey'], 'r') as f:
                        vkey_data = json.load(f)
                        required_fields = ['protocol', 'curve', 'nPublic', 'vk_alpha_1', 'vk_beta_2', 'vk_gamma_2', 'vk_delta_2', 'IC']
                        for field in required_fields:
                            if field not in vkey_data:
                                vkey_valid = False
                                break
                except (json.JSONDecodeError, IOError):
                    vkey_valid = False

            artifact_results.append({
                'circuit': circuit,
                'artifacts_found': list(found_artifacts.keys()),
                'vkey_valid': vkey_valid,
                'complete': len(found_artifacts) >= 3  # At least wasm, zkey, vkey
            })

        self.validation_results['compilation_check'] = artifact_results
        return artifact_results

    def validate_security_properties(self) -> List[Dict[str, Any]]:
        """Validate security properties of ZK circuits."""
        print("🔐 Validating Security Properties...")

        security_results = []

        # Validate trusted setup artifacts
        ptau_files = list(self.artifacts_dir.glob("powersoftau*.ptau"))
        if ptau_files:
            print(f"✅ Trusted setup artifacts found: {len(ptau_files)} files")
        else:
            print("⚠️ No trusted setup artifacts found")

        # Check for secure circuit configurations
        secure_circuits = ['model_update_secure']
        for circuit in secure_circuits:
            zkey_path = self.artifacts_dir / f"proving_key_secure.zkey"
            vkey_path = self.artifacts_dir / f"verification_key_secure.json"

            if zkey_path.exists() and vkey_path.exists():
                print(f"✅ {circuit} - Secure cryptographic keys present")
                security_results.append({
                    'circuit': circuit,
                    'secure_keys': True,
                    'trusted_setup': bool(ptau_files)
                })
            else:
                print(f"❌ {circuit} - Missing secure cryptographic keys")
                security_results.append({
                    'circuit': circuit,
                    'secure_keys': False,
                    'trusted_setup': bool(ptau_files)
                })

        self.validation_results['security_validation'] = security_results
        return security_results

    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        print("\n📊 ZK Toolchain Validation Report")
        print("=" * 50)

        # Overall status
        toolchain_ok = self.validation_results['toolchain_check']
        circuits_valid = all(r['valid'] for r in self.validation_results['circuit_validation'])
        artifacts_complete = all(r['complete'] for r in self.validation_results['compilation_check'])
        security_ok = all(r['secure_keys'] for r in self.validation_results['security_validation'])

        overall_status = toolchain_ok and circuits_valid and artifacts_complete and security_ok

        print(f"🔧 Toolchain Status: {'✅ OK' if toolchain_ok else '❌ FAILED'}")
        print(f"🔍 Circuit Validation: {'✅ OK' if circuits_valid else '❌ FAILED'}")
        print(f"📦 Artifact Validation: {'✅ OK' if artifacts_complete else '❌ FAILED'}")
        print(f"🔐 Security Validation: {'✅ OK' if security_ok else '❌ FAILED'}")
        print(f"🎯 Overall Status: {'✅ PASSED' if overall_status else '❌ FAILED'}")

        # Detailed results
        print(f"\nCircuits Validated: {len(self.validation_results['circuit_validation'])}")
        print(f"Valid Circuits: {sum(1 for r in self.validation_results['circuit_validation'] if r['valid'])}")
        print(f"Artifacts Verified: {len(self.validation_results['compilation_check'])}")
        print(f"Complete Artifacts: {sum(1 for r in self.validation_results['compilation_check'] if r['complete'])}")

        report = {
            'timestamp': '2025-09-04T10:00:00.000000',
            'validation_type': 'ZK Toolchain CI/CD Validation',
            'overall_status': overall_status,
            'results': self.validation_results,
            'recommendations': []
        }

        if not overall_status:
            report['recommendations'] = [
                "Fix circuit syntax errors" if not circuits_valid else None,
                "Regenerate missing compilation artifacts" if not artifacts_complete else None,
                "Ensure secure cryptographic keys are present" if not security_ok else None,
                "Verify ZK toolchain installation" if not toolchain_ok else None
            ]
            report['recommendations'] = [r for r in report['recommendations'] if r is not None]

        return report

    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete ZK toolchain validation."""
        print("🚀 Starting ZK Toolchain Validation...")

        # Run all validation steps
        toolchain_ok = self.validate_toolchain()
        if not toolchain_ok:
            print("❌ Toolchain validation failed - aborting further checks")
            return self.generate_validation_report()

        self.validate_circuits()
        self.validate_compilation_artifacts()
        self.validate_security_properties()

        return self.generate_validation_report()


def main():
    """Main validation entry point for CI/CD."""
    print("🔐 FEDzk ZK Toolchain Validator")
    print("=" * 40)

    validator = ZKToolchainValidator()

    try:
        report = validator.run_full_validation()

        # Save validation report for CI/CD
        report_file = Path("test_reports/zk_toolchain_validation.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Validation report saved: {report_file}")

        # Exit with appropriate code
        if report['overall_status']:
            print("✅ ZK Toolchain validation PASSED")
            return 0
        else:
            print("❌ ZK Toolchain validation FAILED")
            print("Recommendations:")
            for rec in report.get('recommendations', []):
                print(f"  - {rec}")
            return 1

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

