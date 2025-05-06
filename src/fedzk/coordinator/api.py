# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
Coordinator API for FedZK.
Defines REST endpoints for submitting updates and checking status.
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fedzk.coordinator.logic import ProofVerificationError, get_status, submit_update

app = FastAPI(
    title="FedZK Coordinator API",
    description="REST API for submitting federated learning updates with zero-knowledge proofs",
    version="0.1.0"
)

class UpdateRequest(BaseModel):
    gradients: Dict[str, List[float]] = Field(..., description="Gradient updates by parameter name")
    proof: Dict[str, Any] = Field(..., description="Zero-knowledge proof object")
    public_inputs: List[Any] = Field(..., description="Public inputs/signals for proof verification")

class SubmitResponse(BaseModel):
    status: str = Field(..., description="accepted or aggregated")
    model_version: int = Field(..., description="Model version after submission")
    global_update: Optional[Dict[str, List[float]]] = Field(None, description="Averaged update if aggregation occurred")

class StatusResponse(BaseModel):
    pending_updates: int = Field(..., description="Number of pending updates")
    model_version: int = Field(..., description="Current model version")

@app.post("/submit_update", response_model=SubmitResponse)
def submit_update_endpoint(request: UpdateRequest):
    try:
        status, version, global_update = submit_update(request.gradients, request.proof, request.public_inputs)
        return SubmitResponse(status=status, model_version=version, global_update=global_update)
    except ProofVerificationError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/status", response_model=StatusResponse)
def get_status_endpoint():
    pending, version = get_status()
    return StatusResponse(pending_updates=pending, model_version=version)



