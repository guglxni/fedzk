# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
Federated Learning Aggregator for FEDzk.

This module provides a FastAPI service for the coordinator node in the FEDzk system,
which aggregates model updates from clients after verifying their ZK proofs.
"""

import os
import pathlib
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fedzk.prover.verifier import ZKVerifier

app = FastAPI(title="FEDzk Aggregator",
              description="Coordinator service for FEDzk federated learning with zero-knowledge proofs")

# Define base directory for ZK assets
ASSET_DIR = pathlib.Path(__file__).resolve().parent.parent / "zk"

# Initialize verifier with real verification key path
STD_VER_KEY = str(ASSET_DIR / "verification_key.json")
verifier = ZKVerifier(STD_VER_KEY)

# In-memory storage for pending updates
# In production, this would use a database
pending_updates: List[Dict[str, List[float]]] = []
current_version = 1


class UpdateSubmission(BaseModel):
    """Model for client update submissions."""
    gradients: Dict[str, List[float]]
    proof: str
    public_signals: List[Dict[str, Any]]


@app.post("/submit_update")
def submit_update(update: UpdateSubmission):
    """
    Submit a model update with zero-knowledge proof for verification and aggregation.
    
    Args:
        update: Contains gradients, proof, and public signals
        
    Returns:
        Dictionary with status and version info, and global update if aggregation occurred
    """
    # Verify the ZK proof
    if not verifier.verify_proof(update.proof, update.public_signals):
        raise HTTPException(status_code=400, detail="Invalid ZK proof")

    # Add gradients to pending updates
    pending_updates.append(update.gradients)

    # If we have enough updates, perform aggregation (FedAvg)
    if len(pending_updates) >= 2:
        # Simulate FedAvg (average values for each param)
        keys = pending_updates[0].keys()
        avg_update = {
            k: [(sum(grad[k][i] for grad in pending_updates) / len(pending_updates))
                for i in range(len(pending_updates[0][k]))]
            for k in keys
        }

        # Reset pending updates
        pending_updates.clear()

        # Update model version
        global current_version
        current_version += 1

        return {
            "status": "aggregated",
            "version": current_version,
            "global_update": avg_update
        }

    return {"status": "accepted", "version": current_version}


@app.get("/status")
def get_status():
    """
    Get the current status of the aggregator.
    
    Returns:
        Dictionary with current model version and number of pending updates
    """
    return {
        "current_model_version": current_version,
        "pending_updates": len(pending_updates)
    }
