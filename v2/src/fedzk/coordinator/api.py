# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
Coordinator API for FEDzk.
Defines REST endpoints for submitting updates and checking status.
"""

import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fedzk.coordinator.logic import (
    ProofVerificationError,
    CryptographicIntegrityError,
    get_status,
    submit_update,
    get_security_stats,
    get_aggregation_history
)

app = FastAPI(
    title="FEDzk Coordinator API",
    description="REST API for submitting federated learning updates with zero-knowledge proofs",
    version="0.1.0"
)

class UpdateRequest(BaseModel):
    gradients: Dict[str, List[float]] = Field(..., description="Gradient updates by parameter name")
    proof: Dict[str, Any] = Field(..., description="Zero-knowledge proof object")
    public_inputs: List[Any] = Field(..., description="Public inputs/signals for proof verification")
    client_id: str = Field("unknown", description="Client identifier for security tracking")

class SubmitResponse(BaseModel):
    status: str = Field(..., description="accepted or aggregated")
    model_version: int = Field(..., description="Model version after submission")
    global_update: Optional[Dict[str, List[float]]] = Field(None, description="Averaged update if aggregation occurred")

class StatusResponse(BaseModel):
    pending_updates: int = Field(..., description="Number of pending updates")
    model_version: int = Field(..., description="Current model version")

@app.post("/submit_update", response_model=SubmitResponse)
def submit_update_endpoint(request: UpdateRequest):
    """
    Submit federated learning update with comprehensive cryptographic verification.

    This endpoint performs:
    - Rate limiting and security checks
    - Comprehensive ZK proof verification
    - Cryptographic integrity validation
    - Secure aggregation when threshold is met
    """
    try:
        status, version, global_update = submit_update(
            request.gradients,
            request.proof,
            request.public_inputs,
            request.client_id
        )
        return SubmitResponse(status=status, model_version=version, global_update=global_update)
    except ProofVerificationError as e:
        # Map different error types to appropriate HTTP status codes
        status_code = 400
        if hasattr(e, 'error_type'):
            if e.error_type == "client_blocked":
                status_code = 429
            elif e.error_type == "rate_limited":
                status_code = 429
            elif e.error_type == "verification_failed":
                status_code = 400

        raise HTTPException(status_code=status_code, detail=str(e))
    except CryptographicIntegrityError as e:
        raise HTTPException(status_code=422, detail=f"Cryptographic integrity error: {str(e)}")

@app.get("/status", response_model=StatusResponse)
def get_status_endpoint():
    """Get current coordinator status including pending updates and model version."""
    pending, version = get_status()
    return StatusResponse(pending_updates=pending, model_version=version)


@app.get("/security")
def get_security_stats_endpoint():
    """
    Get security statistics and performance metrics.

    Returns detailed information about:
    - Blocked clients and failed verifications
    - Performance metrics for proof verification
    - System status and pending operations
    """
    return get_security_stats()


@app.get("/aggregation/history")
def get_aggregation_history_endpoint(limit: int = 10):
    """
    Get history of completed aggregation batches.

    Args:
        limit: Maximum number of batches to return (default: 10)

    Returns:
        List of aggregation batch summaries with metadata
    """
    return get_aggregation_history(limit)


@app.get("/health")
def health_check_endpoint():
    """
    Comprehensive health check for the coordinator.

    Returns:
        status: Overall health status
        cryptographic_verification: ZK verification system status
        security: Security system status
        aggregation: Aggregation system status
        performance: Performance metrics
    """
    try:
        security_stats = get_security_stats()

        # Check if coordinator can perform basic operations
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "cryptographic_verification": {
                "status": "operational",
                "verifier_initialized": True
            },
            "security": security_stats["security"],
            "aggregation": {
                "pending_updates": security_stats["system"]["total_pending_updates"],
                "completed_batches": security_stats["system"]["total_aggregation_batches"],
                "current_version": security_stats["system"]["current_model_version"]
            },
            "performance": security_stats["performance"]
        }

        # Check for any critical issues
        if security_stats["security"]["blocked_clients"] > 10:
            health_status["status"] = "degraded"
            health_status["warnings"] = ["High number of blocked clients"]

        if security_stats["performance"].get("success_rate", 1.0) < 0.95:
            health_status["status"] = "degraded"
            health_status["warnings"] = health_status.get("warnings", []) + ["Low verification success rate"]

        return health_status

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }



