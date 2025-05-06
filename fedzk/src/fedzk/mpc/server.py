# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
MPC Server module for FedZK Proof generation and verification.
Exposes /generate_proof and /verify_proof endpoints.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from fedzk.prover.verifier import ZKVerifier
from fedzk.prover.zkgenerator import ZKProver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mpc_server")

# Load circuit and key paths from environment or defaults
STD_WASM = os.getenv("ZK_WASM", "zk/model_update.wasm")
STD_ZKEY = os.getenv("ZK_ZKEY", "zk/proving_key.zkey")
SEC_WASM = os.getenv("ZK_SEC_WASM", "zk/model_update_secure.wasm")
SEC_ZKEY = os.getenv("ZK_SEC_ZKEY", "zk/proving_key_secure.zkey")
STD_VER_KEY = os.getenv("ZK_VER_KEY", "zk/verification_key.json")
SEC_VER_KEY = os.getenv("ZK_SEC_VER_KEY", "zk/verification_key_secure.json")

app = FastAPI(
    title="FedZK MPC Proof Server",
    description="Service to generate and verify zero-knowledge proofs via HTTP",
    version="0.1.0"
)

# Load allowed API keys from environment variable (comma-separated)
raw_keys = os.getenv("MPC_API_KEYS", "")
ALLOWED_API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

class GenerateRequest(BaseModel):
    gradients: Optional[List[float]] = Field(None, description="Flattened gradient vector")
    batch: bool = Field(False, description="Enable batch processing of multiple gradient sets")
    gradient_batches: Optional[List[List[float]]] = Field(None, description="List of gradient lists for batch processing")
    secure: bool = Field(False, description="Use secure circuit with constraints")
    max_norm: float = Field(100.0, description="Maximum allowed L2 norm for secure circuit")
    min_active: int = Field(3, description="Minimum required non-zero elements for secure circuit")

class GenerateResponse(BaseModel):
    proof: Any = Field(..., description="Generated proof object")
    public_inputs: Any = Field(..., description="Public signals for proof verification")

class VerifyRequest(BaseModel):
    proof: Any = Field(..., description="Proof object to verify")
    public_inputs: Any = Field(..., description="Public signals for verification")
    secure: bool = Field(False, description="Use secure circuit verification")

class VerifyResponse(BaseModel):
    valid: bool = Field(..., description="Whether the proof is valid")

@app.post("/generate_proof")
def generate_proof_endpoint(req: GenerateRequest, request: Request):
    logger.info(f"Generate request received (secure={req.secure})")
    # Authenticate API key
    api_key = request.headers.get("x-api-key")
    if api_key not in ALLOWED_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    # Validate request: gradients or gradient_batches must be provided
    if not req.batch and req.gradients is None:
        raise HTTPException(status_code=422, detail="Missing 'gradients' for proof generation")
    try:
        # Handle batch requests
        if req.batch:
            if not req.gradient_batches:
                raise HTTPException(status_code=422, detail="Missing gradient_batches for batch processing")
            results = []
            for grads in req.gradient_batches:
                # Convert float gradients to integers for the circuit
                integer_grads = [int(g) if isinstance(g, int) else int(g * 10) for g in grads]
                gradient_tensor = torch.tensor(integer_grads)
                gradient_dict: Dict[str, torch.Tensor] = {"gradients": gradient_tensor}
                # Generate proof per batch item
                if req.secure:
                    proof, public_inputs = ZKProver(SEC_WASM, SEC_ZKEY).generate_real_proof_secure(gradient_dict)
                else:
                    proof, public_inputs = ZKProver(STD_WASM, STD_ZKEY).generate_real_proof(gradient_dict)
                results.append({"proof": proof, "public_inputs": public_inputs})
            return {"batch_proofs": results}
        # Choose circuit and key based on secure flag
        secure = req.secure
        circuit_path = SEC_WASM if secure else STD_WASM
        key_path = SEC_ZKEY if secure else STD_ZKEY

        # Validate existence of files
        if not os.path.exists(circuit_path) or not os.path.exists(key_path):
            msg = f"Circuit or key not found: {circuit_path}, {key_path}"
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

        # Convert float gradients to integers for the circuit
        integer_gradients = [int(g) if isinstance(g, int) else int(g * 10) for g in req.gradients]

        # Prepare gradient dict for prover
        gradient_tensor = torch.tensor(integer_gradients)
        gradient_dict: Dict[str, torch.Tensor] = {"gradients": gradient_tensor}

        prover = ZKProver(circuit_path, key_path)
        if secure:
            # For secure circuits, ensure we have integer values for maxNorm and minNonZero
            max_norm = getattr(req, "max_norm", 100)
            min_active = getattr(req, "min_active", 3)

            # Convert to integers if they're floats
            if isinstance(max_norm, float):
                max_norm = int(max_norm * 100)  # Scale appropriately
            if isinstance(min_active, float):
                min_active = int(min_active)

            proof, public_inputs = prover.generate_real_proof_secure(
                gradient_dict,
                max_norm=max_norm,
                min_active=min_active
            )
        else:
            proof, public_inputs = prover.generate_real_proof(gradient_dict)

        # Single-proof response
        return {"proof": proof, "public_inputs": public_inputs}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in generate_proof")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify_proof", response_model=VerifyResponse)
def verify_proof_endpoint(req: VerifyRequest, request: Request):
    logger.info(f"Verify request received (secure={req.secure})")
    # Authenticate API key
    api_key = request.headers.get("x-api-key")
    if api_key not in ALLOWED_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    try:
        # Choose verification key
        secure = req.secure
        ver_key = SEC_VER_KEY if secure else STD_VER_KEY

        if not os.path.exists(ver_key):
            msg = f"Verification key not found: {ver_key}"
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

        verifier = ZKVerifier(ver_key)
        if secure:
            valid = verifier.verify_real_proof_secure(req.proof, req.public_inputs)
        else:
            valid = verifier.verify_real_proof(req.proof, req.public_inputs)

        return VerifyResponse(valid=valid)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in verify_proof")
        raise HTTPException(status_code=500, detail=str(e))



