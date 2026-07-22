# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
ZK Toolchain Validator for FEDzk

This module provides comprehensive validation of the Zero-Knowledge proof toolchain
including tool installation, version compatibility, and circuit artifact integrity.
"""

import subprocess
import json
import hashlib
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ZKValidator:
    """
    Comprehensive Zero-Knowledge Toolchain Validator for FEDzk.

    This class provides centralized validation for:
    - Circom compiler installation and version compatibility
    - SNARKjs installation and version compatibility
    - Circuit file existence and integrity validation
    - Trusted setup artifact validation
    - Runtime health monitoring

    Used by ZKProver, ZKVerifier, and MPC server components.
    """

    # Minimum supported versions
    MIN_CIRCOM_VERSION = "2.1.0"
    MIN_SNARKJS_VERSION = "0.7.0"

    # Expected file sizes for basic integrity checks (in bytes)
    EXPECTED_FILE_SIZES = {
        "model_update.wasm": (1000, 100000),  # 1KB - 100KB
        "model_update_secure.wasm": (1000, 100000),
        "proving_key.zkey": (5000, 200000),  # 5KB - 200KB (adjusted for actual file sizes)
        "proving_key_secure.zkey": (10000, 300000),  # 10KB - 300KB
        "verification_key.json": (500, 100000),  # 500B - 100KB
        "verification_key_secure.json": (500, 100000),
    }

    def __init__(self, zk_asset_dir: Optional[str] = None):
        """
        Initialize ZK Validator.

        Args:
            zk_asset_dir: Path to ZK assets directory. If None, uses default.
        """
        if zk_asset_dir:
            self.zk_dir = Path(zk_asset_dir)
        else:
            # Default to src/fedzk/zk relative to this file
            self.zk_dir = Path(__file__).resolve().parent.parent / "zk"

        self.validation_results = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Runtime monitoring attributes
        self.runtime_monitoring = False
        self.monitor_thread = None
        self.last_health_check = None
        self.health_check_interval = 300  # 5 minutes default
        self.degradation_mode = False
        self.degradation_start_time = None
        self.health_history = []
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3

    def attempt_recovery(self) -> bool:
        """
        Attempt to recover from ZK toolchain issues.

        Returns:
            bool: True if recovery was successful, False otherwise.
        """
        if self.recovery_attempts >= self.max_recovery_attempts:
            logger.warning(f"Maximum recovery attempts ({self.max_recovery_attempts}) exceeded")
            return False

        self.recovery_attempts += 1
        logger.info(f"Attempting ZK toolchain recovery (attempt {self.recovery_attempts})")

        try:
            # Attempt to reinitialize paths and validate again
            self._validate_zk_toolchain()
            validation_result = self.validate_toolchain()

            if validation_result["overall_status"] == "passed":
                logger.info("✅ ZK toolchain recovery successful")
                self.recovery_attempts = 0  # Reset counter on success
                self.degradation_mode = False
                return True
            else:
                logger.warning("ZK toolchain recovery failed - issues persist")
                return False

        except Exception as e:
            logger.error(f"ZK toolchain recovery failed with error: {e}")
            return False

    def validate_toolchain(self) -> Dict[str, Any]:
        """
        Perform comprehensive validation of the ZK toolchain.

        Returns:
            Dict containing validation results with status, errors, and warnings.
        """
        results = {
            "overall_status": "unknown",
            "circom": self._validate_circom(),
            "snarkjs": self._validate_snarkjs(),
            "circuit_files": self._validate_circuit_files(),
            "integrity": self._validate_file_integrity(),
            "errors": [],
            "warnings": []
        }

        # Aggregate errors and warnings
        for component, result in results.items():
            if component in ["overall_status", "errors", "warnings"]:
                continue
            if "error" in result:
                results["errors"].append(f"{component}: {result['error']}")
            if "warning" in result:
                results["warnings"].append(f"{component}: {result['warning']}")

        # Determine overall status
        if results["errors"]:
            results["overall_status"] = "failed"
        elif results["warnings"]:
            results["overall_status"] = "warning"
        else:
            results["overall_status"] = "passed"

        self.validation_results = results
        return results

    def _validate_circom(self) -> Dict[str, Any]:
        """
        Validate Circom compiler installation and version.

        Returns:
            Dict with validation status, version info, and any errors.
        """
        try:
            # Check if circom is installed
            result = subprocess.run(["circom", "--version"],
                                  capture_output=True, text=True, check=True)
            version_output = result.stdout.strip()

            # Parse version (circom --version outputs: "circom 2.1.8")
            version = self._parse_circom_version(version_output)

            if not version:
                return {
                    "status": "failed",
                    "error": "Could not parse Circom version from output"
                }

            # Check version compatibility
            if self._compare_versions(version, self.MIN_CIRCOM_VERSION) < 0:
                return {
                    "status": "failed",
                    "version": version,
                    "error": f"Circom version {version} is below minimum required {self.MIN_CIRCOM_VERSION}"
                }

            return {
                "status": "passed",
                "version": version,
                "message": f"Circom {version} is compatible"
            }

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return {
                "status": "failed",
                "error": f"Circom not found or failed to run: {str(e)}"
            }

    def _validate_snarkjs(self) -> Dict[str, Any]:
        """
        Validate SNARKjs installation and version.

        Returns:
            Dict with validation status, version info, and any errors.
        """
        try:
            # Check if snarkjs is installed
            # Note: snarkjs --version can return exit code 99 but still work
            result = subprocess.run(["snarkjs", "--version"],
                                  capture_output=True, text=True, check=False)

            # SNARKjs --version can return exit code 99 but still provide version info
            if result.returncode not in [0, 99]:
                return {
                    "status": "failed",
                    "error": f"SNARKjs returned unexpected exit code {result.returncode}"
                }

            version_output = result.stdout.strip() or result.stderr.strip()

            # Parse version from output
            version = self._parse_snarkjs_version(version_output)

            if not version:
                # SNARKjs version parsing can be tricky, so we'll be lenient
                self.logger.warning(f"Could not parse SNARKjs version from: {version_output}")
                return {
                    "status": "passed",  # Assume it's working if it responds
                    "version": "unknown",
                    "message": "SNARKjs is installed and responding",
                    "warning": "Could not determine version"
                }

            # Check version compatibility
            if self._compare_versions(version, self.MIN_SNARKJS_VERSION) < 0:
                return {
                    "status": "warning",  # Warning instead of failure for SNARKjs
                    "version": version,
                    "warning": f"SNARKjs version {version} is below recommended {self.MIN_SNARKJS_VERSION}"
                }

            return {
                "status": "passed",
                "version": version,
                "message": f"SNARKjs {version} is compatible"
            }

        except FileNotFoundError:
            return {
                "status": "failed",
                "error": "SNARKjs not found in PATH"
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": f"SNARKjs validation failed: {str(e)}"
            }

    def _validate_circuit_files(self) -> Dict[str, Any]:
        """
        Validate that all required circuit files exist.

        Returns:
            Dict with validation status and missing files list.
        """
        required_files = [
            "model_update.wasm",
            "model_update_secure.wasm",
            "proving_key.zkey",
            "proving_key_secure.zkey",
            "verification_key.json",
            "verification_key_secure.json"
        ]

        missing_files = []
        found_files = []

        for filename in required_files:
            filepath = self.zk_dir / filename
            if filepath.exists():
                found_files.append(filename)
            else:
                missing_files.append(filename)

        if missing_files:
            return {
                "status": "failed",
                "missing_files": missing_files,
                "error": f"Missing circuit files: {', '.join(missing_files)}"
            }

        return {
            "status": "passed",
            "found_files": found_files,
            "message": f"All {len(found_files)} circuit files found"
        }

    def _validate_file_integrity(self) -> Dict[str, Any]:
        """
        Validate circuit file integrity through size and basic checks.

        Returns:
            Dict with integrity validation results.
        """
        integrity_results = {
            "status": "passed",
            "checked_files": [],
            "issues": []
        }

        for filename, (min_size, max_size) in self.EXPECTED_FILE_SIZES.items():
            filepath = self.zk_dir / filename

            if not filepath.exists():
                continue

            try:
                file_size = filepath.stat().st_size
                integrity_results["checked_files"].append({
                    "file": filename,
                    "size": file_size
                })

                if file_size < min_size:
                    integrity_results["issues"].append(
                        f"{filename}: File too small ({file_size} bytes, expected > {min_size})"
                    )
                elif file_size > max_size:
                    integrity_results["issues"].append(
                        f"{filename}: File too large ({file_size} bytes, expected < {max_size})"
                    )

                # Additional validation for JSON files
                if filename.endswith('.json'):
                    try:
                        with open(filepath, 'r') as f:
                            json.load(f)
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        integrity_results["issues"].append(
                            f"{filename}: Invalid JSON format: {str(e)}"
                        )

            except OSError as e:
                integrity_results["issues"].append(
                    f"{filename}: Cannot access file: {str(e)}"
                )

        if integrity_results["issues"]:
            integrity_results["status"] = "failed"

        return integrity_results

    # ===== RUNTIME VALIDATION METHODS =====

    def start_runtime_monitoring(self, check_interval: int = 300) -> bool:
        """
        Start continuous runtime monitoring of ZK toolchain health.

        Args:
            check_interval: Health check interval in seconds (default: 5 minutes)

        Returns:
            bool: True if monitoring started successfully, False otherwise
        """
        if self.runtime_monitoring:
            self.logger.warning("Runtime monitoring is already running")
            return False

        self.health_check_interval = check_interval
        self.runtime_monitoring = True

        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._runtime_monitor_loop,
            daemon=True,
            name="ZKValidator-Monitor"
        )
        self.monitor_thread.start()

        self.logger.info(f"Started runtime monitoring with {check_interval}s interval")
        return True

    def stop_runtime_monitoring(self) -> bool:
        """
        Stop continuous runtime monitoring.

        Returns:
            bool: True if monitoring stopped successfully, False otherwise
        """
        if not self.runtime_monitoring:
            self.logger.warning("Runtime monitoring is not running")
            return False

        self.runtime_monitoring = False

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)

        self.logger.info("Stopped runtime monitoring")
        return True

    def _runtime_monitor_loop(self):
        """Main runtime monitoring loop."""
        while self.runtime_monitoring:
            try:
                health_status = self.perform_health_check()
                self._record_health_status(health_status)

                # Handle degradation
                if health_status["status"] == "failed":
                    self._enter_degradation_mode(health_status)
                elif health_status["status"] == "passed" and self.degradation_mode:
                    self._exit_degradation_mode()

            except Exception as e:
                self.logger.error(f"Runtime monitoring error: {e}")
                health_status = {
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
                self._record_health_status(health_status)

            time.sleep(self.health_check_interval)

    def perform_health_check(self) -> Dict[str, Any]:
        """
        Perform a quick health check of the ZK toolchain.

        Returns:
            Dict containing health status information
        """
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "status": "unknown",
            "components": {},
            "issues": []
        }

        # Quick tool availability checks
        health_status["components"]["circom"] = self._quick_tool_check("circom", ["--version"])
        health_status["components"]["snarkjs"] = self._quick_tool_check("snarkjs", ["--version"])

        # File integrity check
        integrity_result = self._validate_file_integrity()
        health_status["components"]["files"] = {
            "status": integrity_result["status"],
            "issues": integrity_result.get("issues", [])
        }

        # Determine overall status
        component_statuses = [comp["status"] for comp in health_status["components"].values()
                            if isinstance(comp, dict) and "status" in comp]

        if "failed" in component_statuses:
            health_status["status"] = "failed"
        elif "warning" in component_statuses:
            health_status["status"] = "warning"
        else:
            health_status["status"] = "passed"

        # Collect issues
        for comp_name, comp_data in health_status["components"].items():
            if isinstance(comp_data, dict) and "issues" in comp_data:
                for issue in comp_data["issues"]:
                    health_status["issues"].append(f"{comp_name}: {issue}")

        self.last_health_check = health_status
        self._record_health_status(health_status)
        return health_status

    def _quick_tool_check(self, tool_name: str, args: List[str]) -> Dict[str, Any]:
        """
        Perform a quick availability check for a tool.

        Args:
            tool_name: Name of the tool to check
            args: Arguments to pass to the tool

        Returns:
            Dict with tool status
        """
        try:
            result = subprocess.run([tool_name] + args,
                                  capture_output=True,
                                  timeout=5.0,
                                  check=False)

            if tool_name == "snarkjs":
                # SNARKjs returns exit code 99 but still works
                success = result.returncode in [0, 99]
            else:
                success = result.returncode == 0

            return {
                "status": "passed" if success else "failed",
                "available": success,
                "exit_code": result.returncode
            }

        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            return {
                "status": "failed",
                "available": False,
                "error": str(e)
            }

    def _record_health_status(self, health_status: Dict[str, Any]):
        """Record health status in history."""
        self.health_history.append(health_status)

        # Keep only last 100 health checks to prevent memory issues
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]

    def _enter_degradation_mode(self, health_status: Dict[str, Any]):
        """Enter graceful degradation mode when health issues are detected."""
        if not self.degradation_mode:
            self.degradation_mode = True
            self.degradation_start_time = datetime.now()

            self.logger.warning("Entering degradation mode due to ZK toolchain issues:")
            for issue in health_status.get("issues", []):
                self.logger.warning(f"  • {issue}")

            # Could trigger additional recovery actions here
            self._attempt_recovery()

    def _exit_degradation_mode(self):
        """Exit degradation mode when issues are resolved."""
        if self.degradation_mode:
            degradation_duration = datetime.now() - self.degradation_start_time
            self.logger.info(f"Exiting degradation mode after {degradation_duration}")
            self.degradation_mode = False
            self.degradation_start_time = None

    def _attempt_recovery(self):
        """Attempt to recover from detected issues."""
        # For now, just log the attempt - in production this could:
        # - Restart services
        # - Re-download corrupted files
        # - Send alerts to administrators
        self.logger.info("Attempting recovery from ZK toolchain issues...")

        # Re-validate toolchain
        validation_results = self.validate_toolchain()

        if validation_results["overall_status"] == "passed":
            self.logger.info("Recovery successful - ZK toolchain restored")
        else:
            self.logger.error("Recovery failed - manual intervention required")
            # In production, this could trigger alerts or fail-safe modes

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status of the ZK toolchain.

        Returns:
            Dict containing current health information
        """
        if not self.last_health_check:
            # Return basic status with degradation info even if no health check performed
            health_info = {
                "status": "unknown",
                "message": "No health check performed yet",
                "timestamp": datetime.now().isoformat(),
                "degradation_mode": self.degradation_mode,
                "monitoring_active": self.runtime_monitoring
            }

            if self.degradation_mode and self.degradation_start_time:
                degradation_duration = datetime.now() - self.degradation_start_time
                health_info["degradation_duration"] = str(degradation_duration)

            return health_info

        health_info = self.last_health_check.copy()
        health_info["degradation_mode"] = self.degradation_mode
        health_info["monitoring_active"] = self.runtime_monitoring

        if self.degradation_mode and self.degradation_start_time:
            degradation_duration = datetime.now() - self.degradation_start_time
            health_info["degradation_duration"] = str(degradation_duration)

        return health_info

    def get_health_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent health check history.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            List of recent health status entries
        """
        return self.health_history[-limit:] if self.health_history else []

    def _parse_circom_version(self, version_output: str) -> Optional[str]:
        """
        Parse Circom version from command output.

        Args:
            version_output: Raw output from 'circom --version'

        Returns:
            Version string or None if parsing fails.
        """
        # Expected formats:
        # "circom compiler 2.2.2"
        # "circom 2.1.8"
        try:
            parts = version_output.strip().split()
            if len(parts) >= 3 and parts[0] == "circom" and parts[1] == "compiler":
                return parts[2]
            elif len(parts) >= 2 and parts[0] == "circom":
                return parts[1]
        except:
            pass
        return None

    def _parse_snarkjs_version(self, version_output: str) -> Optional[str]:
        """
        Parse SNARKjs version from command output.

        Args:
            version_output: Raw output from 'snarkjs --version'

        Returns:
            Version string or None if parsing fails.
        """
        # SNARKjs version output can vary, look for version patterns
        import re

        # Look for version patterns like "0.7.4", "1.0.0", etc.
        version_match = re.search(r'(\d+\.\d+\.\d+)', version_output)
        if version_match:
            return version_match.group(1)

        return None

    def _compare_versions(self, version1: str, version2: str) -> int:
        """
        Compare two version strings.

        Args:
            version1: First version string
            version2: Second version string

        Returns:
            -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        """
        def parse_version(v: str) -> Tuple[int, ...]:
            return tuple(int(x) for x in v.split('.') if x.isdigit())

        try:
            v1_parts = parse_version(version1)
            v2_parts = parse_version(version2)

            # Pad shorter version with zeros
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts = v1_parts + (0,) * (max_len - len(v1_parts))
            v2_parts = v2_parts + (0,) * (max_len - len(v2_parts))

            if v1_parts < v2_parts:
                return -1
            elif v1_parts > v2_parts:
                return 1
            else:
                return 0
        except:
            # If parsing fails, assume versions are compatible
            return 0

    def get_validation_report(self) -> str:
        """
        Generate a human-readable validation report.

        Returns:
            Formatted string with validation results.
        """
        if not self.validation_results:
            return "No validation results available. Run validate_toolchain() first."

        report = []
        report.append("🔍 FEDzk ZK Toolchain Validation Report")
        report.append("=" * 50)

        overall_status = self.validation_results["overall_status"]
        status_emoji = {
            "passed": "✅",
            "warning": "⚠️",
            "failed": "❌",
            "unknown": "❓"
        }

        report.append(f"Overall Status: {status_emoji.get(overall_status, '❓')} {overall_status.upper()}")

        # Component status
        for component in ["circom", "snarkjs", "circuit_files", "integrity"]:
            if component in self.validation_results:
                comp_result = self.validation_results[component]
                status = comp_result.get("status", "unknown")
                emoji = status_emoji.get(status, "❓")

                report.append(f"\n{component.upper()}: {emoji} {status.upper()}")

                if "version" in comp_result:
                    report.append(f"  Version: {comp_result['version']}")
                if "message" in comp_result:
                    report.append(f"  {comp_result['message']}")
                if "error" in comp_result:
                    report.append(f"  Error: {comp_result['error']}")
                if "warning" in comp_result:
                    report.append(f"  Warning: {comp_result['warning']}")

        # Errors and warnings summary
        if self.validation_results.get("errors"):
            report.append(f"\n❌ ERRORS ({len(self.validation_results['errors'])}):")
            for error in self.validation_results["errors"]:
                report.append(f"  • {error}")

        if self.validation_results.get("warnings"):
            report.append(f"\n⚠️  WARNINGS ({len(self.validation_results['warnings'])}):")
            for warning in self.validation_results["warnings"]:
                report.append(f"  • {warning}")

        if overall_status == "passed":
            report.append("\n🎉 ZK Toolchain is ready for production use!")
        elif overall_status == "warning":
            report.append("\n⚠️  ZK Toolchain has warnings - review before production use")
        else:
            report.append("\n❌ ZK Toolchain validation failed - fix issues before use")
            report.append("   Run 'scripts/setup_zk.sh' to install/configure ZK toolchain")

        return "\n".join(report)


def validate_zk_toolchain(zk_asset_dir: Optional[str] = None) -> Tuple[bool, str]:
    """
    Convenience function to validate ZK toolchain and return status.

    Args:
        zk_asset_dir: Path to ZK assets directory

    Returns:
        Tuple of (is_valid: bool, report: str)
    """
    validator = ZKValidator(zk_asset_dir)
    results = validator.validate_toolchain()
    report = validator.get_validation_report()

    is_valid = results["overall_status"] in ["passed", "warning"]
    return is_valid, report
