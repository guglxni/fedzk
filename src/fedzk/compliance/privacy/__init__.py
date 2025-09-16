"""
Privacy Framework

This module provides privacy impact assessment, data minimization,
and privacy protection capabilities for FEDZK.
"""

from .privacy_assessor import PrivacyImpactAssessor
from .data_minimization import DataMinimization

__all__ = [
    'PrivacyImpactAssessor',
    'DataMinimization'
]
