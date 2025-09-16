# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

# src/fedzk/mpc/client.py
import requests
import time
import logging
from typing import Dict, Any, Optional, Tuple, List

from fedzk.prover.zkgenerator import ZKProver
from fedzk.prover.batch_zkgenerator import BatchZKProver

logger = logging.getLogger(__name__)

class MPCClient:
    def __init__(
        self,
        server_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        logger.info(f"MPCClient initialized with server: {server_url}")
        logger.info("Real cryptographic operations only - no fallbacks or mocks")

    def generate_proof(
        self,
        gradient_dict: Dict[str, Any],
        secure: bool = False,
        batch: bool = False,
        chunk_size: Optional[int] = None,
        max_norm_squared: Optional[float] = None,
        min_active: Optional[int] = None,
    ) -> Tuple[Dict, List]:
        """
        Generate zero-knowledge proof via MPC server.

        Args:
            gradient_dict: Dictionary of gradient tensors/parameters
            secure: Whether to use secure circuit with constraints
            batch: Enable batch processing of multiple gradient sets
            chunk_size: Chunk size for batch processing
            max_norm_squared: Maximum allowed squared L2 norm for gradients
            min_active: Minimum number of non-zero gradient elements required

        Returns:
            Tuple of (proof_dict, public_signals_list)

        Raises:
            ConnectionError: If MPC server is unreachable
            RuntimeError: If proof generation fails
        """
        # Validate input
        if not gradient_dict:
            raise ValueError("Empty gradient dictionary provided")

        # Prepare request payload - convert tensors to lists for JSON serialization
        processed_gradients = {}
        for key, value in gradient_dict.items():
            if hasattr(value, 'tolist'):  # PyTorch tensor
                processed_gradients[key] = value.tolist()
            else:
                processed_gradients[key] = value

        payload = {
            "gradients": processed_gradients,
            "secure": secure,
            "batch": batch,
        }

        if chunk_size is not None:
            payload["chunk_size"] = chunk_size
        if max_norm_squared is not None:
            payload["maxNorm"] = max_norm_squared
        if min_active is not None:
            payload["minNonZero"] = min_active

        # Make MPC server call - no fallbacks allowed
        logger.info(f"Attempting MPC proof generation via server: {self.server_url}")
        response = self._call_mpc_server(payload)

        if response and "proof" in response:
            logger.info("MPC server proof generation successful")
            return response["proof"], response.get("public_signals", [])
        else:
            raise RuntimeError("MPC server returned invalid response")

    def _call_mpc_server(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make HTTP call to MPC server.

        Args:
            payload: Request payload for proof generation

        Returns:
            Server response or None if failed
        """
        url = f"{self.server_url}/generate_proof"

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"MPC server call attempt {attempt + 1}/{self.max_retries}")

                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )

                response.raise_for_status()
                return response.json()

            except requests.RequestException as e:
                logger.warning(f"MPC server call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All MPC server attempts failed: {e}")
                    raise

        return None



    def get_server_health(self) -> Dict[str, Any]:
        """
        Check MPC server health status.

        Returns:
            Dict containing server health information
        """
        try:
            url = f"{self.server_url}/health"
            headers = {}
            if self.api_key:
                headers["x-api-key"] = self.api_key

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            health_data = response.json()
            health_data["server_reachable"] = True
            return health_data

        except requests.RequestException as e:
            return {
                "server_reachable": False,
                "error": str(e),
                "status": "unhealthy"
            }
        except Exception as e:
            return {
                "server_reachable": False,
                "error": str(e),
                "status": "error"
            } 