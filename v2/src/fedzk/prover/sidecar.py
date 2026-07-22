# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""HTTP client for fedzk-zk Axum sidecar (optional verify path)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class SidecarUnavailable(RuntimeError):
    pass


def sidecar_base() -> str:
    return os.environ.get("FEDZK_ZK_SIDECAR", "http://127.0.0.1:8787").rstrip("/")


def healthz(base: Optional[str] = None, timeout: float = 2.0) -> Dict[str, Any]:
    url = f"{base or sidecar_base()}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise SidecarUnavailable(f"healthz failed: {e}") from e


def verify_via_sidecar(
    proof: Dict[str, Any],
    public_inputs: List[Any],
    vkey: Union[Dict[str, Any], str, Path],
    *,
    base: Optional[str] = None,
    timeout: float = 30.0,
) -> bool:
    """POST /verify with inline vkey JSON (or load from path)."""
    if isinstance(vkey, (str, Path)):
        vkey_obj = json.loads(Path(vkey).read_text())
    else:
        vkey_obj = vkey
    body = json.dumps(
        {"vkey": vkey_obj, "proof": proof, "public": public_inputs}
    ).encode()
    req = urllib.request.Request(
        f"{base or sidecar_base()}/verify",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
            return bool(payload.get("ok"))
    except urllib.error.HTTPError as e:
        if e.code in (400, 422):
            return False
        raise SidecarUnavailable(f"verify HTTP {e.code}: {e.read()[:200]!r}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise SidecarUnavailable(f"verify failed: {e}") from e
