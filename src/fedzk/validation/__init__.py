"""
FEDZK Validation Module

This module provides validation components for the FEDZK federated learning framework.
It includes gradient validation and proof validation functionality.
"""

from .gradient_validator import GradientValidator
from .proof_validator import ProofValidator

__all__ = [
    'GradientValidator',
    'ProofValidator'
]
