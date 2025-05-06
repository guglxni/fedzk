# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
Coordinator logic for FedZK.
Handles in-memory state for pending updates, proof verification, and aggregation.
"""

from typing import Dict, List, Optional, Tuple

from fedzk.prover.verifier import ZKVerifier


class ProofVerificationError(Exception):
    """Raised when ZK proof verification fails."""
    pass

# In-memory storage for pending updates and model version
pending_updates: List[Dict[str, List[float]]] = []
current_version: int = 1

# Initialize verifier (uses dummy key by default)
verifier = ZKVerifier("dummy_vk.json")

def submit_update(
    gradients: Dict[str, List[float]],
    proof: Dict,
    public_inputs: List
) -> Tuple[str, int, Optional[Dict[str, List[float]]]]:
    """
    Verify the provided ZK proof and either accept or aggregate updates.

    Returns:
        status: 'accepted' or 'aggregated'
        model_version: updated version number
        global_update: averaged gradients if aggregated, otherwise None
    Raises:
        ProofVerificationError: if verification fails
    """
    # Verify the proof
    if not verifier.verify_real_proof(proof, public_inputs):
        raise ProofVerificationError("Invalid ZK proof")

    # Store gradients
    pending_updates.append(gradients)

    # Aggregate when threshold reached
    if len(pending_updates) >= 2:
        keys = pending_updates[0].keys()
        avg_update = {
            k: [
                sum(update[k][i] for update in pending_updates) / len(pending_updates)
                for i in range(len(pending_updates[0][k]))
            ] for k in keys
        }
        # Reset state
        pending_updates.clear()
        global current_version
        current_version += 1
        return "aggregated", current_version, avg_update

    # Otherwise, accepted but not aggregated yet
    return "accepted", current_version, None

def get_status() -> Tuple[int, int]:
    """
    Get current pending update count and model version.

    Returns:
        pending_count, model_version
    """
    return len(pending_updates), current_version



