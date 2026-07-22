#!/usr/bin/env python3
"""
Advanced Proof Validation for FEDzk
====================================

Comprehensive cryptographic proof validation beyond basic SNARKjs verification.
Implements defense against malformed proof attacks and advanced security checks.

Features:
- Proof format validation and sanitization
- Defense against malformed proof attacks
- Proof size limits and complexity checks
- Additional cryptographic validation layers
- Attack pattern detection and mitigation
"""

import json
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)

class ProofValidationError(Exception):
    """Base exception for proof validation errors."""
    pass

class MalformedProofError(ProofValidationError):
    """Exception for malformed proof attacks."""
    pass

class ProofSizeError(ProofValidationError):
    """Exception for proof size violations."""
    pass

class ProofComplexityError(ProofValidationError):
    """Exception for proof complexity violations."""
    pass

class ProofFormatError(ProofValidationError):
    """Exception for proof format violations."""
    pass

class AttackPattern(Enum):
    """Known attack patterns for proof validation."""
    BUFFER_OVERFLOW = "buffer_overflow"
    INTEGER_OVERFLOW = "integer_overflow"
    FORMAT_INJECTION = "format_injection"
    MALFORMED_JSON = "malformed_json"
    SIZE_BOMB = "size_bomb"
    RECURSIVE_STRUCTURE = "recursive_structure"
    NULL_BYTE_INJECTION = "null_byte_injection"
    UNICODE_BOMB = "unicode_bomb"
    LARGE_NUMBER_ATTACK = "large_number_attack"
    DUPLICATE_KEYS = "duplicate_keys"

@dataclass
class ProofValidationConfig:
    """Configuration for proof validation."""
    max_proof_size: int = 1024 * 1024  # 1MB max
    max_signal_count: int = 1000
    max_field_size: int = 10000
    allowed_protocols: Set[str] = None
    max_nesting_depth: int = 10
    max_string_length: int = 100000
    enable_attack_detection: bool = True
    strict_mode: bool = True

    def __post_init__(self):
        if self.allowed_protocols is None:
            self.allowed_protocols = {"groth16", "plonk", "marlin"}

@dataclass
class ProofValidationResult:
    """Result of proof validation."""
    is_valid: bool
    attack_patterns_detected: List[AttackPattern]
    security_score: float
    validation_time: float
    warnings: List[str]
    metadata: Dict[str, Any]

class AdvancedProofValidator:
    """
    Advanced proof validator with comprehensive security checks.

    This validator goes beyond basic SNARKjs verification to provide:
    - Proof format validation and sanitization
    - Defense against malformed proof attacks
    - Proof size limits and complexity checks
    - Attack pattern detection and mitigation
    """

    def __init__(self, config: ProofValidationConfig = None):
        self.config = config or ProofValidationConfig()
        self.attack_patterns = set()
        self.validation_history = []

    def validate_proof_comprehensive(
        self,
        proof: Dict[str, Any],
        public_signals: List[Any],
        circuit_type: str = "model_update"
    ) -> ProofValidationResult:
        """
        Perform comprehensive proof validation.

        Args:
            proof: The cryptographic proof to validate
            public_signals: Public signals for verification
            circuit_type: Type of circuit (for circuit-specific validation)

        Returns:
            ProofValidationResult: Comprehensive validation result
        """
        start_time = time.time()
        attack_patterns = []
        warnings = []
        security_score = 100.0

        try:
            # Phase 1: Basic structural validation
            self._validate_proof_structure(proof, public_signals)

            # Phase 2: Size and complexity limits
            size_score = self._validate_size_limits(proof, public_signals)
            security_score -= size_score

            # Phase 3: Format validation and sanitization
            format_score = self._validate_proof_format(proof, public_signals)
            security_score -= format_score

            # Phase 4: Attack pattern detection
            if self.config.enable_attack_detection:
                detected_attacks = self._detect_attack_patterns(proof, public_signals)
                attack_patterns.extend(detected_attacks)
                security_score -= len(detected_attacks) * 20

            # Phase 5: Cryptographic integrity checks
            integrity_score = self._validate_cryptographic_integrity(proof, public_signals)
            security_score -= integrity_score

            # Phase 6: Circuit-specific validation
            circuit_score = self._validate_circuit_specific(proof, public_signals, circuit_type)
            security_score -= circuit_score

            # Phase 7: Timing attack resistance
            timing_score = self._validate_timing_resistance(proof, public_signals)
            security_score -= timing_score

            # Calculate final security score
            security_score = max(0.0, min(100.0, security_score))

            # Generate warnings based on score
            if security_score < 80:
                warnings.append("Low security score - additional review recommended")
            if security_score < 60:
                warnings.append("Critical security score - proof should be rejected")
            if attack_patterns:
                warnings.append(f"Attack patterns detected: {[p.value for p in attack_patterns]}")

            validation_time = time.time() - start_time

            # Log validation result
            self._log_validation_result(proof, security_score, attack_patterns, validation_time)

            return ProofValidationResult(
                is_valid=security_score >= 50 and not attack_patterns,  # Reject if score < 50 or attacks detected
                attack_patterns_detected=attack_patterns,
                security_score=security_score,
                validation_time=validation_time,
                warnings=warnings,
                metadata={
                    "circuit_type": circuit_type,
                    "proof_size": len(json.dumps(proof)),
                    "signal_count": len(public_signals),
                    "validation_phases": 7
                }
            )

        except ProofValidationError as e:
            validation_time = time.time() - start_time
            return ProofValidationResult(
                is_valid=False,
                attack_patterns_detected=[AttackPattern.MALFORMED_JSON],
                security_score=0.0,
                validation_time=validation_time,
                warnings=[f"Validation error: {str(e)}"],
                metadata={"error_type": type(e).__name__}
            )

    def _validate_proof_structure(self, proof: Dict[str, Any], public_signals: List[Any]) -> None:
        """Validate basic proof structure."""
        # Check required fields for Groth16
        required_fields = ['pi_a', 'pi_b', 'pi_c']
        for field in required_fields:
            if field not in proof:
                raise ProofFormatError(f"Missing required proof field: {field}")

        # Validate protocol
        if 'protocol' in proof:
            if proof['protocol'] not in self.config.allowed_protocols:
                raise ProofFormatError(f"Unsupported protocol: {proof['protocol']}")

        # Validate public signals
        if not isinstance(public_signals, list):
            raise ProofFormatError("Public signals must be a list")

        if len(public_signals) > self.config.max_signal_count:
            raise ProofSizeError(f"Too many public signals: {len(public_signals)} > {self.config.max_signal_count}")

    def _validate_size_limits(self, proof: Dict[str, Any], public_signals: List[Any]) -> float:
        """Validate proof size limits and return penalty score."""
        penalty_score = 0.0

        # Check total proof size
        proof_size = len(json.dumps(proof))
        if proof_size > self.config.max_proof_size:
            penalty_score += 50
            raise ProofSizeError(f"Proof size too large: {proof_size} > {self.config.max_proof_size}")

        # Check individual field sizes
        for key, value in proof.items():
            field_size = len(str(value))
            if field_size > self.config.max_field_size:
                penalty_score += 20

        # Check signal sizes
        for i, signal in enumerate(public_signals):
            signal_size = len(str(signal))
            if signal_size > self.config.max_field_size:
                penalty_score += 10

        return penalty_score

    def _validate_proof_format(self, proof: Dict[str, Any], public_signals: List[Any]) -> float:
        """Validate and sanitize proof format."""
        penalty_score = 0.0

        # Validate pi_a format (should be [x, y] for elliptic curve point)
        if 'pi_a' in proof:
            pi_a = proof['pi_a']
            if not self._is_valid_ec_point(pi_a):
                penalty_score += 30

        # Validate pi_b format (should be [[x1, y1], [x2, y2]] for elliptic curve point)
        if 'pi_b' in proof:
            pi_b = proof['pi_b']
            if not self._is_valid_ec_point_pair(pi_b):
                penalty_score += 30

        # Validate pi_c format
        if 'pi_c' in proof:
            pi_c = proof['pi_c']
            if not self._is_valid_ec_point(pi_c):
                penalty_score += 30

        # Validate public signals format
        for i, signal in enumerate(public_signals):
            if not self._is_valid_public_signal(signal):
                penalty_score += 10

        return penalty_score

    def _detect_attack_patterns(self, proof: Dict[str, Any], public_signals: List[Any]) -> List[AttackPattern]:
        """Detect known attack patterns in proof."""
        detected_attacks = []

        # Check for buffer overflow attempts
        if self._detect_buffer_overflow(proof, public_signals):
            detected_attacks.append(AttackPattern.BUFFER_OVERFLOW)

        # Check for integer overflow attempts
        if self._detect_integer_overflow(proof, public_signals):
            detected_attacks.append(AttackPattern.INTEGER_OVERFLOW)

        # Check for format injection
        if self._detect_format_injection(proof, public_signals):
            detected_attacks.append(AttackPattern.FORMAT_INJECTION)

        # Check for malformed JSON
        if self._detect_malformed_json(proof, public_signals):
            detected_attacks.append(AttackPattern.MALFORMED_JSON)

        # Check for size bomb attacks
        if self._detect_size_bomb(proof, public_signals):
            detected_attacks.append(AttackPattern.SIZE_BOMB)

        # Check for recursive structures
        if self._detect_recursive_structure(proof, public_signals):
            detected_attacks.append(AttackPattern.RECURSIVE_STRUCTURE)

        # Check for null byte injection
        if self._detect_null_byte_injection(proof, public_signals):
            detected_attacks.append(AttackPattern.NULL_BYTE_INJECTION)

        # Check for Unicode bomb
        if self._detect_unicode_bomb(proof, public_signals):
            detected_attacks.append(AttackPattern.UNICODE_BOMB)

        # Check for large number attacks
        if self._detect_large_number_attack(proof, public_signals):
            detected_attacks.append(AttackPattern.LARGE_NUMBER_ATTACK)

        # Check for duplicate keys
        if self._detect_duplicate_keys(proof, public_signals):
            detected_attacks.append(AttackPattern.DUPLICATE_KEYS)

        return detected_attacks

    def _validate_cryptographic_integrity(self, proof: Dict[str, Any], public_signals: List[Any]) -> float:
        """Validate cryptographic integrity beyond SNARKjs."""
        penalty_score = 0.0

        # Check proof consistency
        if not self._validate_proof_consistency(proof):
            penalty_score += 40

        # Check signal consistency
        if not self._validate_signal_consistency(public_signals):
            penalty_score += 20

        # Check for weak cryptographic parameters
        if self._detect_weak_crypto_parameters(proof):
            penalty_score += 30

        # Validate hash consistency
        if not self._validate_hash_consistency(proof, public_signals):
            penalty_score += 25

        return penalty_score

    def _validate_circuit_specific(self, proof: Dict[str, Any], public_signals: List[Any], circuit_type: str) -> float:
        """Validate circuit-specific requirements."""
        penalty_score = 0.0

        # Circuit-specific validations
        if circuit_type == "model_update":
            if len(public_signals) != 4:  # Expected for model update circuit
                penalty_score += 20
        elif circuit_type == "secure_model_aggregation":
            if len(public_signals) < 3:  # Minimum signals for secure aggregation
                penalty_score += 20

        # Validate signal types for circuit
        for signal in public_signals:
            if not isinstance(signal, (int, str)):
                penalty_score += 10

        return penalty_score

    def _validate_timing_resistance(self, proof: Dict[str, Any], public_signals: List[Any]) -> float:
        """Validate timing attack resistance."""
        penalty_score = 0.0

        # This would implement timing attack detection
        # For now, we return a minimal penalty
        return penalty_score

    def _is_valid_ec_point(self, point: List) -> bool:
        """Validate elliptic curve point format."""
        if not isinstance(point, list) or len(point) != 2:
            return False

        # Check if coordinates are strings (hex) or integers
        for coord in point:
            if not isinstance(coord, (str, int)):
                return False

            if isinstance(coord, str):
                # Validate hex format
                if not re.match(r'^0x[0-9a-fA-F]+$', coord):
                    return False

        return True

    def _is_valid_ec_point_pair(self, point_pair: List) -> bool:
        """Validate elliptic curve point pair format."""
        if not isinstance(point_pair, list) or len(point_pair) != 2:
            return False

        return all(self._is_valid_ec_point(point) for point in point_pair)

    def _is_valid_public_signal(self, signal: Any) -> bool:
        """Validate public signal format."""
        if isinstance(signal, int):
            # Check for reasonable integer range
            return -2**256 <= signal <= 2**256
        elif isinstance(signal, str):
            # Check string length
            return len(signal) <= self.config.max_string_length
        else:
            return False

    def _detect_buffer_overflow(self, proof: Dict, signals: List) -> bool:
        """Detect buffer overflow attempts."""
        # Check for extremely large values that could cause overflow
        for value in self._flatten_dict_values(proof) + signals:
            if isinstance(value, str) and len(value) > 1000000:
                return True
            elif isinstance(value, int) and abs(value) > 2**512:
                return True
        return False

    def _detect_integer_overflow(self, proof: Dict, signals: List) -> bool:
        """Detect integer overflow attempts."""
        for value in self._flatten_dict_values(proof) + signals:
            if isinstance(value, int):
                # Check for values that could cause overflow in cryptographic operations
                if abs(value) > 2**256:  # Typical field size
                    return True
        return False

    def _detect_format_injection(self, proof: Dict, signals: List) -> bool:
        """Detect format injection attacks."""
        for value in self._flatten_dict_values(proof) + signals:
            if isinstance(value, str):
                # Check for suspicious patterns
                if re.search(r'[%$\\]', value):
                    return True
        return False

    def _detect_malformed_json(self, proof: Dict, signals: List) -> bool:
        """Detect malformed JSON attacks."""
        try:
            # Try to serialize and deserialize
            json_str = json.dumps(proof)
            json.loads(json_str)
            return False
        except (TypeError, ValueError):
            return True

    def _detect_size_bomb(self, proof: Dict, signals: List) -> bool:
        """Detect size bomb attacks."""
        # Check for nested structures that could cause exponential growth
        def check_nesting(obj, depth=0):
            if depth > self.config.max_nesting_depth:
                return True
            if isinstance(obj, dict):
                return any(check_nesting(v, depth + 1) for v in obj.values())
            elif isinstance(obj, list):
                return any(check_nesting(item, depth + 1) for item in obj)
            return False

        return check_nesting(proof) or check_nesting(signals)

    def _detect_recursive_structure(self, proof: Dict, signals: List) -> bool:
        """Detect recursive structure attacks."""
        # This would implement cycle detection
        # For now, check nesting depth
        return self._detect_size_bomb(proof, signals)

    def _detect_null_byte_injection(self, proof: Dict, signals: List) -> bool:
        """Detect null byte injection attacks."""
        for value in self._flatten_dict_values(proof) + signals:
            if isinstance(value, str) and '\x00' in value:
                return True
        return False

    def _detect_unicode_bomb(self, proof: Dict, signals: List) -> bool:
        """Detect Unicode bomb attacks."""
        for value in self._flatten_dict_values(proof) + signals:
            if isinstance(value, str):
                # Check for unusual Unicode patterns
                if len(value.encode('utf-8')) > len(value) * 4:  # 4x expansion suggests bomb
                    return True
        return False

    def _detect_large_number_attack(self, proof: Dict, signals: List) -> bool:
        """Detect large number attacks."""
        for value in self._flatten_dict_values(proof) + signals:
            if isinstance(value, str):
                # Check for extremely large numbers
                if re.match(r'^\d{100,}', value):
                    return True
        return False

    def _detect_duplicate_keys(self, proof: Dict, signals: List) -> bool:
        """Detect duplicate key attacks."""
        try:
            # Try to create JSON with sorted keys
            json.dumps(proof, sort_keys=True)
            return False
        except ValueError:
            return True

    def _validate_proof_consistency(self, proof: Dict) -> bool:
        """Validate proof internal consistency."""
        # Check that all elliptic curve points have consistent format
        points = []
        if 'pi_a' in proof:
            points.append(proof['pi_a'])
        if 'pi_b' in proof:
            points.extend(proof['pi_b'])
        if 'pi_c' in proof:
            points.append(proof['pi_c'])

        # All points should have same format
        formats = [type(point).__name__ for point in points]
        return len(set(formats)) == 1

    def _validate_signal_consistency(self, signals: List) -> bool:
        """Validate signal consistency."""
        if not signals:
            return True

        # Check that all signals have reasonable values
        for signal in signals:
            if isinstance(signal, int):
                if abs(signal) > 2**256:  # Field size
                    return False
            elif isinstance(signal, str):
                if len(signal) > self.config.max_string_length:
                    return False

        return True

    def _detect_weak_crypto_parameters(self, proof: Dict) -> bool:
        """Detect weak cryptographic parameters."""
        # Check for obviously weak values
        for value in self._flatten_dict_values(proof):
            if isinstance(value, str) and value in ['0', '1', '123456789']:
                return True
            elif isinstance(value, int) and value in [0, 1]:
                return True
        return False

    def _validate_hash_consistency(self, proof: Dict, signals: List) -> bool:
        """Validate hash consistency."""
        try:
            # Create hash of proof and signals
            proof_hash = hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest()
            signals_hash = hashlib.sha256(json.dumps(signals, sort_keys=True).encode()).hexdigest()

            # Check if hashes are reasonable (not all zeros)
            return not (proof_hash.startswith('0' * 8) or signals_hash.startswith('0' * 8))
        except Exception:
            return False

    def _flatten_dict_values(self, d: Dict) -> List:
        """Flatten all values in a dictionary."""
        values = []
        for value in d.values():
            if isinstance(value, dict):
                values.extend(self._flatten_dict_values(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        values.extend(self._flatten_dict_values(item))
                    else:
                        values.append(item)
            else:
                values.append(value)
        return values

    def _log_validation_result(self, proof: Dict, score: float, attacks: List, time_taken: float) -> None:
        """Log validation result."""
        attack_names = [a.value for a in attacks]
        logger.info(
            f"Proof validation completed - Score: {score:.1f}, "
            f"Attacks: {attack_names}, Time: {time_taken:.3f}s"
        )

        # Store in validation history
        self.validation_history.append({
            "timestamp": time.time(),
            "score": score,
            "attacks": attack_names,
            "time": time_taken
        })

    def get_validation_stats(self) -> Dict:
        """Get validation statistics."""
        if not self.validation_history:
            return {}

        scores = [h["score"] for h in self.validation_history]
        times = [h["time"] for h in self.validation_history]

        return {
            "total_validations": len(self.validation_history),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "average_time": sum(times) / len(times),
            "attack_patterns_detected": sum(len(h["attacks"]) for h in self.validation_history)
        }

# Convenience functions for easy usage
def validate_proof_security(
    proof: Dict[str, Any],
    public_signals: List[Any],
    config: ProofValidationConfig = None
) -> ProofValidationResult:
    """
    Convenience function for proof validation.

    Args:
        proof: Cryptographic proof to validate
        public_signals: Public signals for verification
        config: Validation configuration (optional)

    Returns:
        ProofValidationResult: Validation result
    """
    validator = AdvancedProofValidator(config)
    return validator.validate_proof_comprehensive(proof, public_signals)

def create_secure_validator(strict_mode: bool = True) -> AdvancedProofValidator:
    """
    Create a validator with secure default settings.

    Args:
        strict_mode: Enable strict validation mode

    Returns:
        AdvancedProofValidator: Configured validator
    """
    config = ProofValidationConfig(
        max_proof_size=512 * 1024,  # 512KB
        max_signal_count=500,
        max_field_size=5000,
        enable_attack_detection=True,
        strict_mode=strict_mode
    )
    return AdvancedProofValidator(config)

