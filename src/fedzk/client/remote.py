# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
HTTP client for FedZK Coordinator API.
Provides methods to submit gradient updates with proofs and fetch coordinator status.
"""

from typing import Any, Dict, List, Optional

import requests


class ClientAPI:
    """
    HTTP client for interacting with the FedZK coordinator service.
    """
    def __init__(self, base_url: str, timeout: Optional[float] = None):
        """
        Args:
            base_url: Coordinator base URL (e.g. http://localhost:8000)
            timeout: Optional request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def submit_update(
        self,
        gradients: Dict[str, List[float]],
        proof: Dict[str, Any],
        public_inputs: List[Any]
    ) -> Dict[str, Any]:
        """
        Submit a model update and ZK proof to the coordinator.

        Returns the JSON response with status, model_version, and optional global_update.
        Raises HTTPError on failure.
        """
        url = f"{self.base_url}/submit_update"
        payload = {
            "gradients": gradients,
            "proof": proof,
            "public_inputs": public_inputs
        }
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_status(self) -> Dict[str, Any]:
        """
        Get current coordinator status: pending_updates and model_version.

        Returns the JSON response.
        Raises HTTPError on failure.
        """
        url = f"{self.base_url}/status"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()



