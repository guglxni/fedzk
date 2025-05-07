# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
MPC Server module for FedZK Proof generation and verification.
Exposes /generate_proof and /verify_proof endpoints.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from pydantic import BaseModel, Field

from fedzk.prover.verifier import ZKVerifier
from fedzk.prover.zkgenerator import ZKProver
from fedzk.prover.batch_zkgenerator import BatchZKProver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mpc_server")

# Attempt to get asset paths from environment or use defaults
# These defaults assume the server is run from the project root
# and assets are in src/fedzk/zk/
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent # Assuming src/fedzk/mpc/server.py
STD_WASM_DEFAULT = str(PROJ_ROOT / "src" / "fedzk" / "zk" / "model_update.wasm")
STD_ZKEY_DEFAULT = str(PROJ_ROOT / "src" / "fedzk" / "zk" / "proving_key.zkey")
SEC_WASM_DEFAULT = str(PROJ_ROOT / "src" / "fedzk" / "zk" / "model_update_secure.wasm")
SEC_ZKEY_DEFAULT = str(PROJ_ROOT / "src" / "fedzk" / "zk" / "proving_key_secure.zkey")
STD_VER_KEY_DEFAULT = str(PROJ_ROOT / "src" / "fedzk" / "zk" / "verification_key.json")
SEC_VER_KEY_DEFAULT = str(PROJ_ROOT / "src" / "fedzk" / "zk" / "verification_key_secure.json")

STD_WASM = os.getenv("MPC_STD_WASM_PATH", STD_WASM_DEFAULT)
STD_ZKEY = os.getenv("MPC_STD_ZKEY_PATH", STD_ZKEY_DEFAULT)
SEC_WASM = os.getenv("MPC_SEC_WASM_PATH", SEC_WASM_DEFAULT)
SEC_ZKEY = os.getenv("MPC_SEC_ZKEY_PATH", SEC_ZKEY_DEFAULT)
STD_VER_KEY = os.getenv("MPC_STD_VER_KEY_PATH", STD_VER_KEY_DEFAULT)
SEC_VER_KEY = os.getenv("MPC_SEC_VER_KEY_PATH", SEC_VER_KEY_DEFAULT)

app = FastAPI(
    title="FedZK MPC Proof Server",
    description="Service to generate and verify zero-knowledge proofs via HTTP",
    version="0.1.0"
)

# Load allowed API keys from environment variable (comma-separated)
raw_keys = os.getenv("MPC_API_KEYS", "")
ALLOWED_API_KEYS = set(os.getenv("MPC_ALLOWED_API_KEYS", "testkey,anotherkey").split(","))

# Stub for API key verification dependency
async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    if x_api_key not in ALLOWED_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

class GenerateRequest(BaseModel):
    gradients: Dict[str, List[float]]
    batch: bool = Field(False, description="Enable batch processing of multiple gradient sets")
    secure: bool = Field(False, description="Use secure circuit with constraints")
    max_norm_squared: Optional[float] = Field(None, alias="maxNorm")
    min_active: Optional[int] = Field(None, alias="minNonZero")
    chunk_size: Optional[int] = Field(None, description="Chunk size for batch processing")

class GenerateResponse(BaseModel):
    proof: Any = Field(..., description="Generated proof object")
    public_inputs: Any = Field(..., description="Public signals for proof verification")

class VerifyRequest(BaseModel):
    proof: Dict[str, Any]
    public_inputs: List[Any]
    secure: bool = False

class VerifyResponse(BaseModel):
    valid: bool

@app.post("/generate_proof", summary="Generate ZK Proof Remotely")
async def generate_proof_endpoint(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    try:
        gradient_dict_tensors = {k: torch.tensor(v) for k, v in req.gradients.items()}
        
        if req.batch:
            logger.info(f"Batch proof request, chunk_size={req.chunk_size}")
            prover = BatchZKProver(
                chunk_size=req.chunk_size,
                secure=req.secure,
                max_norm_squared=req.max_norm_squared,
                min_active=req.min_active
            )
            proof, public_inputs = prover.generate_proof(gradient_dict_tensors)
        else:
            logger.info(f"Single proof request: secure={req.secure}")
            prover = ZKProver(
                secure=req.secure, 
                max_norm_squared=req.max_norm_squared, 
                min_active=req.min_active
            )
            proof, public_inputs = prover.generate_proof(gradient_dict_tensors)

        return {"proof": proof, "public_inputs": public_inputs, "status": "success"}
    except Exception as e:
        logger.exception("Error in generate_proof")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify_proof", response_model=VerifyResponse, summary="Verify ZK Proof Remotely")
async def verify_proof_endpoint(req: VerifyRequest, api_key: str = Depends(verify_api_key)):
    try:
        logger.info(f"Verify request received (secure={req.secure})")
        # Use correct verification key path based on secure parameter
        vkey_path = SEC_VER_KEY if req.secure else STD_VER_KEY
        verifier = ZKVerifier(verification_key_path=vkey_path)
        is_valid = verifier.verify_real_proof(req.proof, req.public_inputs)
        return VerifyResponse(valid=is_valid)
    except Exception as e:
        logger.exception("Error in verify_proof")
        raise HTTPException(status_code=500, detail=str(e))



