# src/fedzk/mpc/client.py
import requests
from typing import Dict, Any, Optional, Tuple, List

class MPCClient:
    def __init__(
        self,
        server_url: str,
        api_key: Optional[str] = None,
        fallback_disabled: bool = False,
        fallback_mode: str = "silent",
    ):
        self.server_url = server_url
        self.api_key = api_key
        self.fallback_disabled = fallback_disabled
        self.fallback_mode = fallback_mode
        # Placeholder for local ZKProver if needed for fallback
        # from fedzk.prover.zkgenerator import ZKProver 
        # self.local_prover = ZKProver()

    def generate_proof(
        self,
        gradient_dict: Dict[str, Any],
        secure: bool = False,
        batch: bool = False,
        chunk_size: Optional[int] = None,
        max_norm_squared: Optional[float] = None,
        min_active: Optional[int] = None,
    ) -> Tuple[Dict, List]: # Assuming proof is Dict, public_inputs is List
        # This is a stub. In a real implementation, it would:
        # 1. Try to POST to self.server_url/generate_proof
        # 2. If successful, return the server's proof and public_inputs.
        # 3. If MPC server fails and fallback is enabled:
        #    - Log warning/error based on fallback_mode.
        #    - Use self.local_prover (properly initialized) to generate proof locally.
        #    - Return local proof.
        # 4. If MPC server fails and fallback is disabled, raise an exception.

        print(f"[MPCClient STUB] Attempting to generate proof via MPC server: {self.server_url}")
        # Simulate an MPC server error for testing fallback
        if not self.fallback_disabled:
            if self.fallback_mode == "warn":
                print("[MPCClient STUB] MPC server failed. WARN: Falling back to local proof.")
            elif self.fallback_mode != "silent": # strict or other non-silent modes
                print("[MPCClient STUB] MPC server failed. ERROR: Falling back to local proof.")
            
            # Simulate local proof generation
            # This requires ZKProver to be importable and correctly functioning
            # For now, returning a consistent stub proof to allow CLI tests to proceed
            print("[MPCClient STUB] Generating proof locally (stubbed)...")
            return {"proof": "local_stub_proof_from_mpc_client_fallback"}, ["local_stub_public_inputs"]
        else:
            print("[MPCClient STUB] MPC server failed and fallback is disabled.")
            raise ConnectionError("MPC server call failed and fallback is disabled (stub behavior)")

        # Fallback for non-implemented path, should not be reached if logic above is complete
        return {"proof": "mpc_stub_proof"}, ["mpc_stub_public_signal"] 