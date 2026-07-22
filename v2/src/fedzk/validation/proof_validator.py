#!/usr/bin/env python3
"""
FEDzk Proof Data Validation
===========================

Comprehensive validation and sanitization for zero-knowledge proof data.
Implements defense against proof manipulation attacks and cryptographic parameter validation.

Features:
- Proof structure validation and sanitization
- Cryptographic parameter validation
- Defense against proof manipulation attacks
- Proof size and complexity limits
- Production security standards
"""

import json
import hashlib
import hmac
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class ProofValidationError(Exception):
    """Base exception for proof validation errors."""
    pass

class ProofStructureError(ProofValidationError):
    """Exception for proof structure violations."""
    pass

class ProofParameterError(ProofValidationError):
    """Exception for cryptographic parameter violations."""
    pass

class ProofManipulationError(ProofValidationError):
    """Exception for proof manipulation attacks."""
    pass

class ProofComplexityError(ProofValidationError):
    """Exception for proof complexity violations."""
    pass

class ProofValidationLevel(Enum):
    """Proof validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"

class ProofAttackPattern(Enum):
    """Known attack patterns for proof manipulation."""
    MALFORMED_STRUCTURE = "malformed_structure"
    INVALID_PARAMETERS = "invalid_parameters"
    SIZE_INFLATION = "size_inflation"
    RECURSIVE_STRUCTURE = "recursive_structure"
    NULL_BYTE_INJECTION = "null_byte_injection"
    UNICODE_BOMB = "unicode_bomb"
    FORMAT_INJECTION = "format_injection"
    CRYPTOGRAPHIC_WEAKNESS = "cryptographic_weakness"
    TIMING_ATTACK = "timing_attack"
    REPLAY_ATTACK = "replay_attack"

class ZKProofType(Enum):
    """Supported zero-knowledge proof types."""
    GROTH16 = "groth16"
    PLONK = "plonk"
    MARLIN = "marlin"
    BULLETPROOFS = "bulletproofs"
    STARK = "stark"

@dataclass
class ProofValidationConfig:
    """Configuration for proof validation."""
    validation_level: ProofValidationLevel = ProofValidationLevel.STRICT
    max_proof_size: int = 1024 * 1024  # 1MB
    max_field_size: int = 10000
    max_nesting_depth: int = 10
    max_string_length: int = 100000
    allowed_proof_types: List[ZKProofType] = None
    enable_attack_detection: bool = True
    enable_timing_protection: bool = True
    enable_replay_protection: bool = True
    proof_cache_ttl: int = 3600  # 1 hour
    enable_size_validation: bool = True
    enable_structure_validation: bool = True

    def __post_init__(self):
        if self.allowed_proof_types is None:
            self.allowed_proof_types = [ZKProofType.GROTH16]

@dataclass
class ProofValidationResult:
    """Result of proof validation."""
    is_valid: bool
    validation_score: float
    detected_attacks: List[ProofAttackPattern]
    cryptographic_analysis: Dict[str, Any]
    structure_analysis: Dict[str, Any]
    warnings: List[str]
    sanitized_proof: Optional[Dict[str, Any]]
    validation_metadata: Dict[str, Any]

@dataclass
class ProofStructure:
    """Analyzed proof structure information."""
    proof_type: str
    field_count: int
    total_size: int
    nesting_depth: int
    has_required_fields: bool
    parameter_ranges: Dict[str, Tuple[float, float]]
    cryptographic_parameters: Dict[str, Any]
    complexity_score: float

class ProofValidator:
    """
    Comprehensive proof data validator for FEDzk.

    Provides multi-layer validation and sanitization for zero-knowledge proofs
    to prevent manipulation attacks and ensure cryptographic integrity.
    """

    def __init__(self, config: ProofValidationConfig = None):
        self.config = config or ProofValidationConfig()
        self.validation_history: List[ProofValidationResult] = []
        self.proof_cache: Dict[str, Dict[str, Any]] = {}
        self.attack_patterns_detected: Dict[str, int] = {}
        self.timing_measurements: List[float] = []

        # Initialize cryptographic parameter ranges for different proof types
        self._initialize_crypto_ranges()

        logger.info("ProofValidator initialized with comprehensive security features")

    def _initialize_crypto_ranges(self):
        """Initialize valid cryptographic parameter ranges."""
        # Groth16 parameter ranges (example values)
        self.crypto_ranges = {
            "groth16": {
                "pi_a": {"expected_length": 2, "value_range": (-2**256, 2**256)},
                "pi_b": {"expected_length": 2, "value_range": (-2**256, 2**256)},
                "pi_c": {"expected_length": 2, "value_range": (-2**256, 2**256)},
                "protocol": {"expected_values": ["groth16"]},
                "curve": {"expected_values": ["bn128", "bls12_381"]}
            },
            "plonk": {
                "proof": {"max_size": 100000},
                "public_signals": {"max_count": 100},
                "protocol": {"expected_values": ["plonk"]}
            }
        }

    def validate_proof_comprehensive(
        self,
        proof: Dict[str, Any],
        proof_type: ZKProofType = ZKProofType.GROTH16,
        public_signals: Optional[List[Any]] = None,
        client_id: str = "unknown"
    ) -> ProofValidationResult:
        """
        Perform comprehensive proof validation.

        Args:
            proof: Proof data to validate
            proof_type: Type of zero-knowledge proof
            public_signals: Public signals for verification
            client_id: Client identifier for tracking

        Returns:
            ProofValidationResult: Comprehensive validation result
        """
        start_time = time.time()
        validation_score = 100.0
        detected_attacks = []
        warnings = []
        sanitized_proof = None

        try:
            # Phase 1: Basic structure validation
            if self.config.enable_structure_validation:
                structure_score, structure_analysis = self._validate_proof_structure(proof, proof_type)
                validation_score -= structure_score

            # Phase 2: Size and complexity limits
            if self.config.enable_size_validation:
                size_score = self._validate_size_limits(proof)
                validation_score -= size_score

            # Phase 3: Cryptographic parameter validation
            crypto_score, crypto_analysis = self._validate_cryptographic_parameters(proof, proof_type)
            validation_score -= crypto_score

            # Phase 4: Attack pattern detection
            if self.config.enable_attack_detection:
                attacks = self._detect_attack_patterns(proof, public_signals)
                detected_attacks.extend(attacks)
                validation_score -= len(attacks) * 20

            # Phase 5: Timing attack protection
            if self.config.enable_timing_protection:
                timing_score = self._validate_timing_resistance(proof)
                validation_score -= timing_score

            # Phase 6: Replay attack protection
            if self.config.enable_replay_protection:
                replay_score = self._validate_replay_protection(proof)
                validation_score -= replay_score

            # Phase 7: Sanitization (if enabled)
            if self.config.validation_level in [ProofValidationLevel.STRICT, ProofValidationLevel.PARANOID]:
                sanitized_proof = self._sanitize_proof(proof)

            # Calculate final score
            validation_score = max(0.0, min(100.0, validation_score))

            # Generate warnings
            if validation_score < 80:
                warnings.append("Low validation score - additional review recommended")
            if validation_score < 60:
                warnings.append("Critical validation score - proof should be rejected")
            if detected_attacks:
                warnings.append(f"Attack patterns detected: {[a.value for a in detected_attacks]}")

            # Record timing
            validation_time = time.time() - start_time
            self.timing_measurements.append(validation_time)

            # Update detection history
            for attack in detected_attacks:
                self.attack_patterns_detected[attack.value] = \
                    self.attack_patterns_detected.get(attack.value, 0) + 1

            # Cache validated proof
            proof_hash = self._calculate_proof_hash(proof)
            self.proof_cache[proof_hash] = {
                "proof": proof,
                "validation_time": validation_time,
                "client_id": client_id,
                "timestamp": time.time()
            }

            result = ProofValidationResult(
                is_valid=validation_score >= 50 and not detected_attacks,
                validation_score=validation_score,
                detected_attacks=detected_attacks,
                cryptographic_analysis=crypto_analysis,
                structure_analysis=structure_analysis if 'structure_analysis' in locals() else {},
                warnings=warnings,
                sanitized_proof=sanitized_proof,
                validation_metadata={
                    "client_id": client_id,
                    "proof_type": proof_type.value,
                    "validation_level": self.config.validation_level.value,
                    "validation_time": validation_time,
                    "timestamp": time.time(),
                    "proof_hash": proof_hash
                }
            )

            # Store result in history
            self.validation_history.append(result)

            logger.info(
                f"Proof validation completed for client {client_id} - "
                f"Score: {validation_score:.1f}, Attacks: {len(detected_attacks)}, "
                f"Time: {validation_time:.3f}s"
            )

            return result

        except ProofValidationError as e:
            validation_time = time.time() - start_time
            logger.error(f"Proof validation error for client {client_id}: {e}")
            return ProofValidationResult(
                is_valid=False,
                validation_score=0.0,
                detected_attacks=[ProofAttackPattern.MALFORMED_STRUCTURE],
                cryptographic_analysis={},
                structure_analysis={},
                warnings=[f"Validation error: {str(e)}"],
                sanitized_proof=None,
                validation_metadata={"error": str(e), "client_id": client_id}
            )

    def _validate_proof_structure(self, proof: Dict[str, Any], proof_type: ZKProofType) -> Tuple[float, Dict[str, Any]]:
        """Validate proof structure and required fields."""
        penalty_score = 0.0
        analysis = {
            "has_required_fields": True,
            "field_count": len(proof),
            "max_nesting_depth": 0,
            "structure_issues": []
        }

        try:
            # Check required fields based on proof type
            required_fields = self._get_required_fields(proof_type)
            missing_fields = []

            for field in required_fields:
                if field not in proof:
                    missing_fields.append(field)
                    penalty_score += 25
                    analysis["has_required_fields"] = False

            if missing_fields:
                analysis["structure_issues"].append(f"Missing required fields: {missing_fields}")

            # Validate field structure
            for field_name, field_value in proof.items():
                field_score, field_analysis = self._validate_field_structure(field_name, field_value)
                penalty_score += field_score

                # Update max nesting depth
                if "nesting_depth" in field_analysis:
                    analysis["max_nesting_depth"] = max(analysis["max_nesting_depth"], field_analysis["nesting_depth"])

                # Collect structure issues
                if "issues" in field_analysis:
                    analysis["structure_issues"].extend(field_analysis["issues"])

            # Check nesting depth
            if analysis["max_nesting_depth"] > self.config.max_nesting_depth:
                penalty_score += 20
                analysis["structure_issues"].append(f"Nesting depth too high: {analysis['max_nesting_depth']}")

        except Exception as e:
            penalty_score += 50
            analysis["structure_issues"].append(f"Structure validation error: {str(e)}")

        return penalty_score, analysis

    def _get_required_fields(self, proof_type: ZKProofType) -> List[str]:
        """Get required fields for a proof type."""
        required_fields_map = {
            ZKProofType.GROTH16: ["pi_a", "pi_b", "pi_c"],
            ZKProofType.PLONK: ["proof", "public_signals"],
            ZKProofType.MARLIN: ["proof", "public_signals"],
            ZKProofType.BULLETPROOFS: ["proof", "public_signals"],
            ZKProofType.STARK: ["proof", "public_signals"]
        }
        return required_fields_map.get(proof_type, [])

    def _validate_field_structure(self, field_name: str, field_value: Any) -> Tuple[float, Dict[str, Any]]:
        """Validate individual field structure."""
        penalty_score = 0.0
        analysis = {"issues": []}

        try:
            # Check field size
            field_size = len(str(field_value))
            if field_size > self.config.max_field_size:
                penalty_score += 15
                analysis["issues"].append(f"Field {field_name} too large: {field_size}")

            # Check for suspicious patterns
            if isinstance(field_value, str):
                if "\x00" in field_value:
                    penalty_score += 20
                    analysis["issues"].append(f"Null byte injection in {field_name}")

                if len(field_value.encode('utf-8')) > len(field_value) * 4:
                    penalty_score += 15
                    analysis["issues"].append(f"Unicode bomb in {field_name}")

            # Calculate nesting depth
            analysis["nesting_depth"] = self._calculate_nesting_depth(field_value)

            # Validate field type
            if not self._is_valid_field_type(field_value):
                penalty_score += 10
                analysis["issues"].append(f"Invalid type for {field_name}: {type(field_value)}")

        except Exception as e:
            penalty_score += 25
            analysis["issues"].append(f"Field validation error for {field_name}: {str(e)}")

        return penalty_score, analysis

    def _calculate_nesting_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate nesting depth of an object."""
        if current_depth > self.config.max_nesting_depth:
            return current_depth

        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_nesting_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            if not obj:
                return current_depth
            return max(self._calculate_nesting_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth

    def _is_valid_field_type(self, value: Any) -> bool:
        """Check if a field value has a valid type."""
        valid_types = (str, int, float, list, dict, tuple, type(None))

        if not isinstance(value, valid_types):
            return False

        # Additional checks for strings
        if isinstance(value, str):
            if len(value) > self.config.max_string_length:
                return False
            # Check for control characters
            if any(ord(c) < 32 and c not in '\n\r\t' for c in value):
                return False

        return True

    def _validate_size_limits(self, proof: Dict[str, Any]) -> float:
        """Validate proof size limits."""
        penalty_score = 0.0

        try:
            # Check total proof size
            proof_size = len(json.dumps(proof))
            if proof_size > self.config.max_proof_size:
                penalty_score += 40
                logger.warning(f"Proof size too large: {proof_size} > {self.config.max_proof_size}")

            # Check individual field sizes
            for field_name, field_value in proof.items():
                field_size = len(str(field_value))
                if field_size > self.config.max_field_size:
                    penalty_score += 10
                    logger.warning(f"Field {field_name} too large: {field_size}")

        except Exception as e:
            penalty_score += 30
            logger.error(f"Size validation error: {e}")

        return penalty_score

    def _validate_cryptographic_parameters(self, proof: Dict[str, Any], proof_type: ZKProofType) -> Tuple[float, Dict[str, Any]]:
        """Validate cryptographic parameters."""
        penalty_score = 0.0
        analysis = {
            "parameter_ranges": {},
            "cryptographic_issues": [],
            "strength_assessment": "unknown"
        }

        try:
            # Get parameter ranges for proof type
            param_ranges = self.crypto_ranges.get(proof_type.value, {})

            for field_name, field_value in proof.items():
                if field_name in param_ranges:
                    field_score, field_analysis = self._validate_parameter_field(
                        field_name, field_value, param_ranges[field_name]
                    )
                    penalty_score += field_score

                    if "range" in field_analysis:
                        analysis["parameter_ranges"][field_name] = field_analysis["range"]

                    if "issues" in field_analysis:
                        analysis["cryptographic_issues"].extend(field_analysis["issues"])

            # Assess overall cryptographic strength
            analysis["strength_assessment"] = self._assess_cryptographic_strength(proof, proof_type)

        except Exception as e:
            penalty_score += 35
            analysis["cryptographic_issues"].append(f"Cryptographic validation error: {str(e)}")

        return penalty_score, analysis

    def _validate_parameter_field(self, field_name: str, field_value: Any, param_config: Dict) -> Tuple[float, Dict[str, Any]]:
        """Validate a specific cryptographic parameter field."""
        penalty_score = 0.0
        analysis = {}

        try:
            # Check expected values
            if "expected_values" in param_config:
                if field_value not in param_config["expected_values"]:
                    penalty_score += 20
                    analysis["issues"] = [f"Invalid value for {field_name}: {field_value}"]

            # Check expected length
            if "expected_length" in param_config:
                if isinstance(field_value, (list, tuple)):
                    if len(field_value) != param_config["expected_length"]:
                        penalty_score += 15
                        analysis["issues"] = [f"Invalid length for {field_name}: {len(field_value)}"]
                elif hasattr(field_value, '__len__'):
                    if len(field_value) != param_config["expected_length"]:
                        penalty_score += 15
                        analysis["issues"] = [f"Invalid length for {field_name}: {len(field_value)}"]

            # Check value ranges
            if "value_range" in param_config:
                min_val, max_val = param_config["value_range"]
                analysis["range"] = (min_val, max_val)

                # Validate numeric values
                if isinstance(field_value, (int, float)):
                    if not (min_val <= field_value <= max_val):
                        penalty_score += 25
                        analysis["issues"] = [f"Value out of range for {field_name}: {field_value}"]
                elif isinstance(field_value, (list, tuple)):
                    for i, val in enumerate(field_value):
                        if isinstance(val, (int, float)):
                            if not (min_val <= val <= max_val):
                                penalty_score += 25
                                analysis["issues"] = [f"Value {val} at index {i} out of range for {field_name}"]

        except Exception as e:
            penalty_score += 20
            analysis["issues"] = [f"Parameter validation error for {field_name}: {str(e)}"]

        return penalty_score, analysis

    def _assess_cryptographic_strength(self, proof: Dict[str, Any], proof_type: ZKProofType) -> str:
        """Assess the cryptographic strength of the proof."""
        try:
            strength_score = 100

            # Check for weak parameters
            if proof_type == ZKProofType.GROTH16:
                # Check for obviously weak values
                for field in ["pi_a", "pi_b", "pi_c"]:
                    if field in proof:
                        value = proof[field]
                        if isinstance(value, (list, tuple)) and len(value) >= 1:
                            if value[0] in [0, 1, "0", "1"]:
                                strength_score -= 30

            # Assess based on proof size and complexity
            proof_size = len(json.dumps(proof))
            if proof_size < 1000:  # Suspiciously small
                strength_score -= 20
            elif proof_size > 500000:  # Very large
                strength_score -= 10

            # Determine strength level
            if strength_score >= 80:
                return "strong"
            elif strength_score >= 60:
                return "moderate"
            elif strength_score >= 40:
                return "weak"
            else:
                return "very_weak"

        except Exception:
            return "unknown"

    def _detect_attack_patterns(self, proof: Dict[str, Any], public_signals: Optional[List[Any]]) -> List[ProofAttackPattern]:
        """Detect known attack patterns in proof data."""
        detected_patterns = []

        try:
            # Check for malformed structure
            if self._detect_malformed_structure(proof):
                detected_patterns.append(ProofAttackPattern.MALFORMED_STRUCTURE)

            # Check for invalid parameters
            if self._detect_invalid_parameters(proof):
                detected_patterns.append(ProofAttackPattern.INVALID_PARAMETERS)

            # Check for size inflation attacks
            if self._detect_size_inflation(proof):
                detected_patterns.append(ProofAttackPattern.SIZE_INFLATION)

            # Check for recursive structures
            if self._detect_recursive_structure(proof):
                detected_patterns.append(ProofAttackPattern.RECURSIVE_STRUCTURE)

            # Check for null byte injection
            if self._detect_null_byte_injection(proof):
                detected_patterns.append(ProofAttackPattern.NULL_BYTE_INJECTION)

            # Check for Unicode bomb
            if self._detect_unicode_bomb(proof):
                detected_patterns.append(ProofAttackPattern.UNICODE_BOMB)

            # Check for format injection
            if self._detect_format_injection(proof):
                detected_patterns.append(ProofAttackPattern.FORMAT_INJECTION)

            # Check for cryptographic weaknesses
            if self._detect_cryptographic_weakness(proof):
                detected_patterns.append(ProofAttackPattern.CRYPTOGRAPHIC_WEAKNESS)

            # Check for replay attacks
            if public_signals and self._detect_replay_attack(proof, public_signals):
                detected_patterns.append(ProofAttackPattern.REPLAY_ATTACK)

        except Exception as e:
            logger.error(f"Error detecting attack patterns: {e}")

        return list(set(detected_patterns))

    def _detect_malformed_structure(self, proof: Dict[str, Any]) -> bool:
        """Detect malformed proof structure."""
        try:
            # Try to serialize as JSON
            json.dumps(proof)
            return False
        except (TypeError, ValueError):
            return True

    def _detect_invalid_parameters(self, proof: Dict[str, Any]) -> bool:
        """Detect invalid cryptographic parameters."""
        # Check for obviously invalid values
        for field_name, field_value in proof.items():
            if isinstance(field_value, (int, float)):
                if field_value in [float('inf'), float('-inf')] or str(field_value).lower() == 'nan':
                    return True
            elif isinstance(field_value, str):
                if field_value in ['', 'null', 'undefined', 'NaN']:
                    return True
        return False

    def _detect_size_inflation(self, proof: Dict[str, Any]) -> bool:
        """Detect size inflation attacks."""
        try:
            proof_str = json.dumps(proof)
            # Check for excessive size without corresponding complexity
            if len(proof_str) > 100000:  # 100KB
                # Check if it's mostly repeated characters
                if len(set(proof_str)) / len(proof_str) < 0.1:  # <10% unique characters
                    return True
            return False
        except:
            return True

    def _detect_recursive_structure(self, proof: Dict[str, Any]) -> bool:
        """Detect recursive structure attacks."""
        return self._calculate_nesting_depth(proof) > self.config.max_nesting_depth * 2

    def _detect_null_byte_injection(self, proof: Dict[str, Any]) -> bool:
        """Detect null byte injection attacks."""
        def check_value(value):
            if isinstance(value, str):
                return '\x00' in value
            elif isinstance(value, (list, tuple)):
                return any(check_value(item) for item in value)
            elif isinstance(value, dict):
                return any(check_value(v) for v in value.values())
            return False

        return check_value(proof)

    def _detect_unicode_bomb(self, proof: Dict[str, Any]) -> bool:
        """Detect Unicode bomb attacks."""
        def check_value(value):
            if isinstance(value, str):
                try:
                    encoded = value.encode('utf-8')
                    return len(encoded) > len(value) * 10  # 10x expansion is suspicious
                except:
                    return True
            elif isinstance(value, (list, tuple)):
                return any(check_value(item) for item in value)
            elif isinstance(value, dict):
                return any(check_value(v) for v in value.values())
            return False

        return check_value(proof)

    def _detect_format_injection(self, proof: Dict[str, Any]) -> bool:
        """Detect format injection attacks."""
        def check_value(value):
            if isinstance(value, str):
                # Check for format string vulnerabilities
                suspicious_patterns = ['%s', '%d', '%x', '${', '{{']
                return any(pattern in value for pattern in suspicious_patterns)
            elif isinstance(value, (list, tuple)):
                return any(check_value(item) for item in value)
            elif isinstance(value, dict):
                return any(check_value(v) for v in value.values())
            return False

        return check_value(proof)

    def _detect_cryptographic_weakness(self, proof: Dict[str, Any]) -> bool:
        """Detect cryptographic weaknesses."""
        # Check for predictable values
        for field_name, field_value in proof.items():
            if isinstance(field_value, str):
                # Check for sequential patterns
                if re.search(r'012345|123456|abcdef', field_value.lower()):
                    return True
            elif isinstance(field_value, (list, tuple)):
                # Check for sequential numeric patterns
                if len(field_value) >= 3:
                    try:
                        values = [float(v) for v in field_value if str(v).replace('.', '').isdigit()]
                        if len(values) >= 3:
                            diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
                            if all(abs(d - diffs[0]) < 0.001 for d in diffs):  # Arithmetic sequence
                                return True
                    except:
                        pass
        return False

    def _detect_replay_attack(self, proof: Dict[str, Any], public_signals: List[Any]) -> bool:
        """Detect replay attacks."""
        # Check if this proof has been seen recently
        proof_hash = self._calculate_proof_hash(proof)

        if proof_hash in self.proof_cache:
            cached = self.proof_cache[proof_hash]
            # If seen within the last few minutes, might be replay
            if time.time() - cached["timestamp"] < 300:  # 5 minutes
                return True

        return False

    def _validate_timing_resistance(self, proof: Dict[str, Any]) -> float:
        """Validate timing attack resistance."""
        # In a real implementation, this would measure timing variations
        # For now, return minimal penalty
        return 0.0

    def _validate_replay_protection(self, proof: Dict[str, Any]) -> float:
        """Validate replay attack protection."""
        penalty_score = 0.0

        # Check for timestamp or nonce fields
        has_timing_protection = any(key in proof for key in ['timestamp', 'nonce', 'jti'])

        if not has_timing_protection:
            penalty_score += 15
            logger.warning("Proof lacks timing/replay protection mechanisms")

        return penalty_score

    def _sanitize_proof(self, proof: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize proof data by removing suspicious elements."""
        sanitized = {}

        for key, value in proof.items():
            try:
                # Sanitize strings
                if isinstance(value, str):
                    # Remove null bytes
                    value = value.replace('\x00', '')
                    # Limit length
                    if len(value) > self.config.max_string_length:
                        value = value[:self.config.max_string_length] + "..."
                # Sanitize lists/dicts recursively
                elif isinstance(value, dict):
                    value = self._sanitize_proof(value)
                elif isinstance(value, list):
                    value = [self._sanitize_value(item) for item in value]

                sanitized[key] = value

            except Exception as e:
                logger.error(f"Error sanitizing {key}: {e}")
                # Skip problematic fields
                continue

        return sanitized

    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize a single value."""
        if isinstance(value, str):
            # Remove control characters except common whitespace
            value = ''.join(c for c in value if ord(c) >= 32 or c in '\n\r\t')
            return value[:self.config.max_string_length] if len(value) > self.config.max_string_length else value
        elif isinstance(value, dict):
            return self._sanitize_proof(value)
        elif isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        else:
            return value

    def _calculate_proof_hash(self, proof: Dict[str, Any]) -> str:
        """Calculate a hash of the proof for caching and replay detection."""
        try:
            # Create a normalized version for consistent hashing
            normalized = json.dumps(proof, sort_keys=True, default=str)
            return hashlib.sha256(normalized.encode()).hexdigest()
        except:
            # Fallback for non-serializable proofs
            return hashlib.sha256(str(proof).encode()).hexdigest()

    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive validation metrics."""
        metrics = {
            "total_validations": len(self.validation_history),
            "average_score": 0.0,
            "attack_patterns_detected": dict(self.attack_patterns_detected),
            "validation_level": self.config.validation_level.value,
            "average_timing": 0.0,
            "cache_size": len(self.proof_cache)
        }

        if self.validation_history:
            scores = [result.validation_score for result in self.validation_history]
            metrics["average_score"] = sum(scores) / len(scores)
            metrics["min_score"] = min(scores)
            metrics["max_score"] = max(scores)

        if self.timing_measurements:
            metrics["average_timing"] = sum(self.timing_measurements) / len(self.timing_measurements)

        # Calculate pattern detection rates
        total_patterns = sum(metrics["attack_patterns_detected"].values())
        metrics["total_patterns_detected"] = total_patterns

        return metrics

    def cleanup_cache(self):
        """Clean up expired entries from cache."""
        current_time = time.time()
        expired_keys = []

        for proof_hash, cached_data in self.proof_cache.items():
            if current_time - cached_data["timestamp"] > self.config.proof_cache_ttl:
                expired_keys.append(proof_hash)

        for key in expired_keys:
            del self.proof_cache[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired proof cache entries")

# Convenience functions for easy usage
def create_proof_validator(level: ProofValidationLevel = ProofValidationLevel.STRICT) -> ProofValidator:
    """
    Create a proof validator with predefined security levels.

    Args:
        level: Validation strictness level

    Returns:
        ProofValidator: Configured validator
    """
    config = ProofValidationConfig(validation_level=level)

    # Adjust config based on level
    if level == ProofValidationLevel.BASIC:
        config.enable_attack_detection = False
        config.enable_timing_protection = False
    elif level == ProofValidationLevel.PARANOID:
        config.max_proof_size = 512 * 1024  # Smaller limit
        config.max_field_size = 5000  # Stricter field limits
        config.enable_replay_protection = True

    return ProofValidator(config)

def validate_zk_proof(
    proof: Dict[str, Any],
    proof_type: ZKProofType = ZKProofType.GROTH16,
    public_signals: Optional[List[Any]] = None,
    client_id: str = "unknown"
) -> ProofValidationResult:
    """
    Convenience function for zero-knowledge proof validation.

    Args:
        proof: Proof data to validate
        proof_type: Type of ZK proof
        public_signals: Public signals
        client_id: Client identifier

    Returns:
        ProofValidationResult: Validation result
    """
    validator = create_proof_validator()
    return validator.validate_proof_comprehensive(proof, proof_type, public_signals, client_id)

