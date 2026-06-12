#!/usr/bin/env python3
"""Runtime device probing helpers for bulk multimodal training.

The ROCm stack on some hosts reports a visible GPU while hanging on the first
real tensor transfer or compute op. These helpers gate CUDA/HIP usage on a
small subprocess smoke test instead of `torch.cuda.is_available()` alone.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict, Tuple

import torch


def _clean_text(text: str, limit: int = 600) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def probe_cuda_runtime(timeout_sec: int = 12) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "ok": False,
            "reason": "torch.cuda.is_available() returned False",
            "device_type": "cpu",
        }

    code = """
import json, time, torch
payload = {
    "torch_version": torch.__version__,
    "hip_version": getattr(torch.version, "hip", None),
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if not payload["cuda_available"] or payload["device_count"] < 1:
    payload["ok"] = False
    payload["reason"] = "no CUDA/HIP device"
    print(json.dumps(payload))
    raise SystemExit(0)

payload["device_name"] = torch.cuda.get_device_name(0)
t0 = time.time()
x = torch.ones((32, 32), device="cuda")
y = (x @ x).sum()
torch.cuda.synchronize()
payload["ok"] = True
payload["elapsed_sec"] = time.time() - t0
payload["result"] = float(y.item())
print(json.dumps(payload))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=os.environ.copy(),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "reason": f"CUDA/HIP smoke test timed out after {timeout_sec}s",
            "device_type": "cpu",
            "stdout": _clean_text(exc.stdout or ""),
            "stderr": _clean_text(exc.stderr or ""),
        }

    stdout = _clean_text(proc.stdout or "")
    stderr = _clean_text(proc.stderr or "")
    if proc.returncode != 0:
        return {
            "ok": False,
            "reason": f"CUDA/HIP smoke test failed with exit code {proc.returncode}",
            "device_type": "cpu",
            "stdout": stdout,
            "stderr": stderr,
        }

    try:
        payload = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except Exception:
        return {
            "ok": False,
            "reason": "CUDA/HIP smoke test returned unparsable output",
            "device_type": "cpu",
            "stdout": stdout,
            "stderr": stderr,
        }

    if not payload.get("ok"):
        payload["device_type"] = "cpu"
        return payload
    payload["device_type"] = "cuda"
    return payload


def select_torch_device() -> Tuple[torch.device, Dict[str, Any]]:
    force_cpu = os.getenv("BMM_FORCE_CPU")
    force_cuda = os.getenv("BMM_FORCE_CUDA")

    if force_cpu == "1":
        return torch.device("cpu"), {
            "ok": True,
            "device_type": "cpu",
            "reason": "forced by BMM_FORCE_CPU=1",
        }

    if force_cuda == "1":
        probe = probe_cuda_runtime()
        if probe.get("ok"):
            return torch.device("cuda"), probe
        return torch.device("cpu"), {
            **probe,
            "reason": "BMM_FORCE_CUDA=1 was set, but runtime smoke test failed",
        }

    if torch.cuda.is_available():
        probe = probe_cuda_runtime()
        if probe.get("ok"):
            return torch.device("cuda"), probe
        return torch.device("cpu"), probe

    if torch.backends.mps.is_available():
        return torch.device("mps"), {
            "ok": True,
            "device_type": "mps",
            "reason": "MPS available",
        }

    return torch.device("cpu"), {
        "ok": True,
        "device_type": "cpu",
        "reason": "no GPU runtime available",
    }
