# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
MPC Server module for FEDzk Proof generation and verification.
Exposes /generate_proof and /verify_proof endpoints.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from pydantic import BaseModel, Field

from fedzk.prover.verifier import ZKVerifier
from fedzk.prover.zkgenerator import ZKProver
from fedzk.prover.batch_zkgenerator import BatchZKProver
from fedzk.prover.zk_validator import ZKValidator

# Global ZK validator instance for runtime monitoring
zk_validator = None

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mpc_server")

# Performance monitoring
import time
from collections import defaultdict

# Metrics storage (in production, use proper metrics system like Prometheus)
metrics = {
    "requests_total": defaultdict(int),
    "request_duration": defaultdict(list),
    "errors_total": defaultdict(int),
    "active_connections": 0
}

def record_request_start():
    """Record request start time."""
    metrics["active_connections"] += 1
    return time.time()

def record_request_end(start_time: float, endpoint: str, status: str):
    """Record request completion."""
    duration = time.time() - start_time
    metrics["active_connections"] -= 1
    metrics["requests_total"][f"{endpoint}_{status}"] += 1
    metrics["request_duration"][endpoint].append(duration)
    
    # Log slow requests
    if duration > 5.0:  # 5 second threshold
        logger.warning(f"Slow request: {endpoint} took {duration:.2f}s")

# Attempt to get asset paths from environment or use defaults
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent.parent # Assuming src/fedzk/mpc/server.py
STD_WASM_DEFAULT = str(PROJ_ROOT / "src" / "fedzk" / "zk" / "circuits" / "build" / "model_update.wasm")
STD_ZKEY_DEFAULT = str(PROJ_ROOT / "src" / "fedzk" / "zk" / "circuits" / "proving_key.zkey")
SEC_WASM_DEFAULT = str(PROJ_ROOT / "src" / "fedzk" / "zk" / "circuits" / "build" / "model_update_secure.wasm")
SEC_ZKEY_DEFAULT = str(PROJ_ROOT / "src" / "fedzk" / "zk" / "circuits" / "proving_key_secure.zkey")
STD_VER_KEY_DEFAULT = str(PROJ_ROOT / "src" / "fedzk" / "zk" / "circuits" / "verification_key.json")
SEC_VER_KEY_DEFAULT = str(PROJ_ROOT / "src" / "fedzk" / "zk" / "circuits" / "verification_key_secure.json")

STD_WASM = os.getenv("MPC_STD_WASM_PATH", STD_WASM_DEFAULT)
STD_ZKEY = os.getenv("MPC_STD_ZKEY_PATH", STD_ZKEY_DEFAULT)
SEC_WASM = os.getenv("MPC_SEC_WASM_PATH", SEC_WASM_DEFAULT)
SEC_ZKEY = os.getenv("MPC_SEC_ZKEY_PATH", SEC_ZKEY_DEFAULT)
STD_VER_KEY = os.getenv("MPC_STD_VER_KEY_PATH", STD_VER_KEY_DEFAULT)
SEC_VER_KEY = os.getenv("MPC_SEC_VER_KEY_PATH", SEC_VER_KEY_DEFAULT)

# Lifespan event handling (modern FastAPI approach)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("FEDzk MPC Proof Server starting up...")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'production')}")
    logger.info("ZK enforcement: strict (no test mode bypasses)")
    
    # Ensure ZK tools are in PATH
    current_path = os.environ.get('PATH', '')
    homebrew_bin = '/opt/homebrew/bin'
    if homebrew_bin not in current_path:
        os.environ['PATH'] = f"{homebrew_bin}:{current_path}"
        logger.info(f"Added {homebrew_bin} to PATH")
    
    # Comprehensive ZK toolchain validation using centralized validator
    zk_asset_dir = Path(__file__).resolve().parent.parent / "zk"
    validator = ZKValidator(str(zk_asset_dir))
    validation_results = validator.validate_toolchain()

    if validation_results["overall_status"] == "failed":
        logger.error("ZK toolchain validation failed:")
        for error in validation_results["errors"]:
            logger.error(f"  • {error}")
        raise RuntimeError(
            "ZK toolchain validation failed. Please run 'scripts/setup_zk.sh' to install/configure ZK toolchain."
        )

    if validation_results["overall_status"] == "warning":
        logger.warning("ZK toolchain validation passed with warnings:")
        for warning in validation_results["warnings"]:
            logger.warning(f"  • {warning}")

    logger.info("✅ ZK toolchain validation passed - all components ready")

    # Initialize ZK validator for runtime monitoring
    global zk_validator
    zk_asset_dir = Path(__file__).resolve().parent.parent / "zk"
    zk_validator = ZKValidator(str(zk_asset_dir))

    # Start runtime monitoring (5 minute intervals)
    if zk_validator.start_runtime_monitoring(check_interval=300):
        logger.info("✅ ZK runtime monitoring started (5-minute intervals)")
    else:
        logger.warning("⚠️  Failed to start ZK runtime monitoring")

    yield  # Application runs here

    # Shutdown
    logger.info("FEDzk MPC Proof Server shutting down...")

    # Stop runtime monitoring
    if zk_validator:
        zk_validator.stop_runtime_monitoring()
        logger.info("✅ ZK runtime monitoring stopped")

    logger.info(f"Final metrics: {dict(metrics['requests_total'])}")

# Production-grade FastAPI configuration
app = FastAPI(
    title="FEDzk MPC Proof Server",
    description="Production-grade service for generating and verifying zero-knowledge proofs in federated learning",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT", "development") == "development" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT", "development") == "development" else None,
    openapi_url="/openapi.json" if os.getenv("ENVIRONMENT", "development") == "development" else None,
    lifespan=lifespan
)

# Add security middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Configure CORS for production
allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add trusted host middleware
trusted_hosts = os.getenv("TRUSTED_HOSTS", "").split(",") if os.getenv("TRUSTED_HOSTS") else ["*"]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

# Production-grade API key verification with rate limiting and security
import hashlib
import time
from collections import defaultdict
from datetime import datetime, timedelta

# Rate limiting storage (in production, use Redis)
_rate_limit_store = defaultdict(list)
_failed_attempts = defaultdict(int)

# Load allowed API keys from environment with secure handling
def load_api_keys():
    """Load and validate API keys from environment."""
    raw_keys = os.getenv("MPC_API_KEYS", "")
    if not raw_keys:
        # In production, this should fail hard
        import warnings
        warnings.warn("No API keys configured! Using default keys for development only.")
        # Hash the default keys for consistency (skip length validation for defaults)
        default_keys = {"testkey", "anotherkey"}
        hashed_keys = set()
        for key in default_keys:
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            hashed_keys.add(key_hash)
        return hashed_keys
    
    # Hash keys for storage (don't store plain text)
    keys = set()
    for key in raw_keys.split(","):
        key = key.strip()
        if len(key) < 32:  # Enforce minimum key length
            raise ValueError(f"API key too short: minimum 32 characters required, got {len(key)}")
        # Store SHA-256 hash of the key
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        keys.add(key_hash)
    
    return keys

ALLOWED_API_KEY_HASHES = load_api_keys()

def hash_api_key(key: str) -> str:
    """Hash an API key for secure comparison."""
    return hashlib.sha256(key.encode()).hexdigest()

def check_rate_limit(client_ip: str, max_requests: int = 100, window_minutes: int = 60) -> bool:
    """Check if client has exceeded rate limit."""
    now = time.time()
    window_start = now - (window_minutes * 60)
    
    # Clean old entries
    _rate_limit_store[client_ip] = [
        req_time for req_time in _rate_limit_store[client_ip] 
        if req_time > window_start
    ]
    
    # Check limit
    if len(_rate_limit_store[client_ip]) >= max_requests:
        return False
    
    # Record this request
    _rate_limit_store[client_ip].append(now)
    return True

def check_failed_attempts(client_ip: str, max_failures: int = 10) -> bool:
    """Check if client has too many failed authentication attempts."""
    return _failed_attempts[client_ip] < max_failures

async def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="x-api-key")
):
    """Production-grade API key verification with security features."""
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limiting
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Check failed attempts
    if not check_failed_attempts(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many failed authentication attempts. IP temporarily blocked."
        )
    
    # Validate API key presence
    if not x_api_key:
        _failed_attempts[client_ip] += 1
        raise HTTPException(
            status_code=401, 
            detail="Missing API key. Include 'x-api-key' header."
        )
    
    # Validate API key format (skip for known default keys)
    if len(x_api_key) < 32 and x_api_key not in {"testkey", "anotherkey"}:
        _failed_attempts[client_ip] += 1
        raise HTTPException(
            status_code=401,
            detail="Invalid API key format."
        )
    
    # Hash and verify key
    key_hash = hash_api_key(x_api_key)
    if key_hash not in ALLOWED_API_KEY_HASHES:
        _failed_attempts[client_ip] += 1
        raise HTTPException(
            status_code=401,
            detail="Invalid API key."
        )
    
    # Reset failed attempts on successful auth
    _failed_attempts[client_ip] = 0
    
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

# Main production endpoints with comprehensive error handling
@app.post("/generate_proof", summary="Generate ZK Proof Remotely")
async def generate_proof_endpoint(
    req: GenerateRequest, 
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Generate zero-knowledge proof for gradient updates with production-grade error handling."""
    start_time = record_request_start()
    client_ip = request.client.host if request.client else "unknown"
    
    try:
        # Input validation
        if not req.gradients:
            metrics["errors_total"]["generate_proof_validation"] += 1
            record_request_end(start_time, "generate_proof", "error")
            raise HTTPException(status_code=422, detail="Empty gradients dictionary")
        
        # Validate gradient data
        for param_name, gradient_values in req.gradients.items():
            if not isinstance(gradient_values, list) or not gradient_values:
                metrics["errors_total"]["generate_proof_validation"] += 1
                record_request_end(start_time, "generate_proof", "error")
                raise HTTPException(
                    status_code=422, 
                    detail=f"Invalid gradient data for parameter '{param_name}'"
                )
            
            # Check for reasonable bounds
            if len(gradient_values) > 10000:  # Prevent DoS
                metrics["errors_total"]["generate_proof_validation"] += 1
                record_request_end(start_time, "generate_proof", "error")
                raise HTTPException(
                    status_code=422, 
                    detail=f"Gradient size too large for parameter '{param_name}'"
                )
        
        logger.info(f"Processing proof request from {client_ip}, secure={req.secure}, batch={req.batch}")
        
        # Convert to tensors with error handling
        try:
            gradient_dict_tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in req.gradients.items()}
        except Exception as e:
            metrics["errors_total"]["generate_proof_conversion"] += 1
            record_request_end(start_time, "generate_proof", "error")
            logger.error(f"Tensor conversion failed: {e}")
            raise HTTPException(status_code=422, detail=f"Failed to convert gradients to tensors: {str(e)}")
        
        # Generate proof based on configuration
        try:
            if req.batch and req.chunk_size and req.chunk_size > 1:
                logger.info(f"Using batch ZK prover with chunk_size={req.chunk_size}")
                prover = BatchZKProver(
                    secure=req.secure,
                    chunk_size=req.chunk_size or 2,
                    max_norm_squared=req.max_norm_squared,
                    min_active=req.min_active
                )
                proof_result = prover.generate_proof(gradient_dict_tensors)
                
                if isinstance(proof_result, dict) and "proof" in proof_result:
                    result = proof_result
                else:
                    result = {"proof": proof_result, "batch": True}
            else:
                logger.info("Using standard ZK prover")
                prover = ZKProver(
                    secure=req.secure,
                    max_norm_squared=req.max_norm_squared,
                    min_active=req.min_active
                )
                proof, public_signals = prover.generate_proof(gradient_dict_tensors)
                result = {
                    "proof": proof,
                    "public_signals": public_signals,
                    "batch": False
                }
            
            # Add metadata
            result.update({
                "timestamp": time.time(),
                "secure": req.secure,
                "client_ip": client_ip,
                "proof_id": hashlib.sha256(f"{client_ip}_{time.time()}".encode()).hexdigest()[:16]
            })
            
            logger.info(f"Proof generated successfully for {client_ip}")
            record_request_end(start_time, "generate_proof", "success")
            return result
            
        except Exception as e:
            metrics["errors_total"]["generate_proof_generation"] += 1
            record_request_end(start_time, "generate_proof", "error")
            logger.error(f"Proof generation failed for {client_ip}: {e}")
            
            # Strict error handling - no fallback or mock implementations allowed
            if "snarkjs" in str(e).lower() or "circom" in str(e).lower() or "witness" in str(e).lower() or "circuit" in str(e).lower():
                raise HTTPException(
                    status_code=503,
                    detail="ZK toolchain error. Please ensure Circom and SNARKjs are properly installed and all circuit files are present."
                )
            else:
                raise HTTPException(status_code=500, detail=f"Proof generation failed: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        metrics["errors_total"]["generate_proof_unexpected"] += 1
        record_request_end(start_time, "generate_proof", "error")
        logger.error(f"Unexpected error in generate_proof: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/verify_proof", response_model=VerifyResponse, summary="Verify ZK Proof Remotely")
async def verify_proof_endpoint(
    req: VerifyRequest, 
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Verify zero-knowledge proof with production-grade error handling."""
    start_time = record_request_start()
    client_ip = request.client.host if request.client else "unknown"
    
    try:
        # Input validation
        if not req.proof:
            metrics["errors_total"]["verify_proof_validation"] += 1
            record_request_end(start_time, "verify_proof", "error")
            raise HTTPException(status_code=422, detail="Empty proof data")
        
        if not isinstance(req.public_inputs, list):
            metrics["errors_total"]["verify_proof_validation"] += 1
            record_request_end(start_time, "verify_proof", "error")
            raise HTTPException(status_code=422, detail="Public inputs must be a list")
        
        logger.info(f"Processing verification request from {client_ip}, secure={req.secure}")
        
        try:
            # Initialize verifier
            vkey_path = SEC_VER_KEY if req.secure else STD_VER_KEY
            verifier = ZKVerifier(vkey_path)
            
            # Perform verification
            is_valid = verifier.verify_proof(req.proof, req.public_inputs)
            
            logger.info(f"Verification completed for {client_ip}: valid={is_valid}")
            record_request_end(start_time, "verify_proof", "success")
            
            return VerifyResponse(valid=is_valid)
            
        except Exception as e:
            metrics["errors_total"]["verify_proof_verification"] += 1
            record_request_end(start_time, "verify_proof", "error")
            logger.error(f"Verification failed for {client_ip}: {e}")
            
            # Strict error handling for verification
            if "No such file" in str(e) or "FileNotFoundError" in str(e) or "verification key" in str(e).lower():
                raise HTTPException(
                    status_code=503,
                    detail="Verification key not found. ZK setup is incomplete. Please run 'scripts/setup_zk.sh' to generate circuit artifacts."
                )
            elif "snarkjs" in str(e).lower() or "verify" in str(e).lower():
                raise HTTPException(
                    status_code=503,
                    detail="ZK verification toolchain error. Please ensure SNARKjs is properly installed."
                )
            else:
                raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        metrics["errors_total"]["verify_proof_unexpected"] += 1
        record_request_end(start_time, "verify_proof", "error")
        logger.error(f"Unexpected error in verify_proof: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ZK Health monitoring functions
async def get_zk_health_status() -> Dict[str, Any]:
    """
    Get current ZK toolchain health status.

    Returns:
        Dict containing ZK health information
    """
    global zk_validator

    if not zk_validator:
        return {
            "status": "unknown",
            "message": "ZK validator not initialized",
            "monitoring_active": False,
            "degradation_mode": False
        }

    try:
        health_status = zk_validator.get_health_status()
        return health_status
    except Exception as e:
        logger.error(f"Failed to get ZK health status: {e}")
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}",
            "monitoring_active": zk_validator.runtime_monitoring if zk_validator else False,
            "degradation_mode": False
        }

# Health check and monitoring endpoints
@app.get("/health", summary="Health Check")
async def health_check():
    """Enhanced health check endpoint with ZK toolchain monitoring."""
    start_time = record_request_start()

    try:
        # Check system health
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development")
        }

        # Enhanced ZK toolchain health check
        zk_health = await get_zk_health_status()

        # Update overall status based on ZK health
        if zk_health.get("status") == "failed":
            health_status["status"] = "unhealthy"
            health_status["zk_issues"] = zk_health.get("issues", [])
        elif zk_health.get("status") == "warning":
            health_status["status"] = "degraded"
            health_status["zk_warnings"] = zk_health.get("issues", [])

        # Include ZK health information
        health_status["zk_toolchain"] = {
            "status": zk_health.get("status"),
            "monitoring_active": zk_health.get("monitoring_active", False),
            "degradation_mode": zk_health.get("degradation_mode", False),
            "last_check": zk_health.get("timestamp")
        }

        # Check ZK circuit files exist (legacy check)
        circuit_files = [STD_WASM, STD_ZKEY, STD_VER_KEY, SEC_WASM, SEC_ZKEY, SEC_VER_KEY]
        missing_files = [f for f in circuit_files if not os.path.exists(f)]

        if missing_files:
            health_status["status"] = "unhealthy" if health_status["status"] == "healthy" else health_status["status"]
            health_status["warnings"] = health_status.get("warnings", []) + [f"Missing circuit files: {missing_files}"]

        # Add comprehensive metrics
        health_status["metrics"] = {
            "active_connections": metrics["active_connections"],
            "total_requests": sum(metrics["requests_total"].values()),
            "total_errors": sum(metrics["errors_total"].values()),
            "zk_monitoring": {
                "active": zk_health.get("monitoring_active", False),
                "degradation_mode": zk_health.get("degradation_mode", False)
            }
        }

        record_request_end(start_time, "health", "success")
        return health_status

    except Exception as e:
        record_request_end(start_time, "health", "error")
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/metrics", summary="Service Metrics")
async def get_metrics(api_key: str = Depends(verify_api_key)):
    """Get detailed service metrics (requires authentication)."""
    try:
        # Calculate average response times
        avg_durations = {}
        for endpoint, durations in metrics["request_duration"].items():
            if durations:
                avg_durations[endpoint] = sum(durations) / len(durations)
        
        return {
            "requests_by_endpoint": dict(metrics["requests_total"]),
            "average_response_times": avg_durations,
            "errors_by_endpoint": dict(metrics["errors_total"]),
            "active_connections": metrics["active_connections"],
            "system_info": {
                "environment": os.getenv("ENVIRONMENT", "production"),
                "zk_enforced": True,
                "timestamp": time.time()
            }
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to collect metrics")

@app.get("/ready", summary="Readiness Check")
async def readiness_check():
    """Readiness check for Kubernetes deployments."""
    try:
        # More thorough checks for readiness - strict ZK validation
        # Check that we can initialize provers (this should work since startup validation passed)
        try:
            ZKProver(secure=False)
            ZKVerifier(STD_VER_KEY)
        except Exception as e:
            logger.error(f"ZK components not ready: {e}")
            raise HTTPException(status_code=503, detail="ZK components not ready")
        
        return {"status": "ready", "timestamp": time.time()}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")

@app.get("/zk/health", summary="ZK Toolchain Health Status")
async def zk_health_check(api_key: str = Depends(verify_api_key)):
    """
    Dedicated ZK toolchain health check endpoint.

    Provides detailed information about ZK toolchain status,
    runtime monitoring, and health history.
    """
    try:
        global zk_validator

        if not zk_validator:
            return {
                "status": "error",
                "message": "ZK validator not initialized",
                "timestamp": time.time()
            }

        # Get current health status
        health_status = zk_validator.get_health_status()

        # Add additional runtime information
        health_info = {
            "status": health_status.get("status"),
            "timestamp": health_status.get("timestamp"),
            "monitoring_active": health_status.get("monitoring_active", False),
            "degradation_mode": health_status.get("degradation_mode", False),
            "degradation_duration": health_status.get("degradation_duration"),
            "components": health_status.get("components", {}),
            "issues": health_status.get("issues", []),
            "last_check_timestamp": health_status.get("timestamp"),
            "health_checks_performed": len(zk_validator.health_history) if zk_validator else 0
        }

        # Add recent health history (last 5 checks)
        if zk_validator:
            recent_history = zk_validator.get_health_history(limit=5)
            health_info["recent_history"] = recent_history

        return health_info

    except Exception as e:
        logger.error(f"ZK health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"ZK health check failed: {str(e)}")

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the MPC server with production configuration."""
    import uvicorn
    
    logger.info(f"Starting FEDzk MPC Proof Server on {host}:{port}")
    
    # Production server configuration
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True,
        workers=1,  # Single worker for now - can be scaled in production
        loop="asyncio",  # Use asyncio (uvloop not required)
        http="auto"  # Auto-detect best HTTP implementation
    )

if __name__ == "__main__":
    run_server(
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000))
    )



