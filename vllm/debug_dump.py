# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import time
from pathlib import Path
from typing import Any

import torch


def _dump_dir() -> Path | None:
    dump_dir = os.environ.get("VLLM_DEBUG_DUMP_DIR")
    if not dump_dir:
        return None
    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_enabled() -> bool:
    return _dump_dir() is not None


def _prefix() -> str:
    return os.environ.get("VLLM_DEBUG_DUMP_PREFIX", "vllm")


def _full_tensors() -> bool:
    return os.environ.get("VLLM_DEBUG_DUMP_FULL_TENSORS", "0") == "1"


def _slice_rows() -> int:
    try:
        return max(1, int(os.environ.get("VLLM_DEBUG_DUMP_SLICE_ROWS", "8")))
    except Exception:
        return 8


def _path_for(tag: str, suffix: str) -> Path | None:
    dump_dir = _dump_dir()
    if dump_dir is None:
        return None
    safe_tag = tag.replace("/", "_").replace(" ", "_")
    return dump_dir / f"{_prefix()}_{time.time_ns()}_{os.getpid()}_{safe_tag}{suffix}"


def append_log(tag: str, payload: dict[str, Any]) -> Path | None:
    path = _path_for(tag, ".log")
    if path is None:
        return None
    record = {
        "tag": tag,
        "ts_ns": time.time_ns(),
        **payload,
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    summary: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }
    try:
        view = tensor.detach().float()
        summary["min"] = float(view.min().item())
        summary["max"] = float(view.max().item())
        summary["mean"] = float(view.mean().item())
        summary["std"] = float(view.std().item()) if view.numel() > 1 else 0.0
    except Exception:
        pass
    return summary


def _tensor_payload(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    cpu = tensor.detach().to("cpu")
    if _full_tensors() or cpu.ndim == 0:
        return cpu
    rows = _slice_rows()
    if cpu.ndim >= 1 and cpu.shape[0] > rows:
        return cpu[-rows:]
    return cpu


def save_tensors(
    tag: str,
    tensors: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> Path | None:
    path = _path_for(tag, ".pt")
    if path is None:
        return None

    payload: dict[str, Any] = {
        "tag": tag,
        "ts_ns": time.time_ns(),
        "meta": meta or {},
        "tensor_summaries": {},
        "tensors": {},
    }
    for name, value in tensors.items():
        if isinstance(value, torch.Tensor):
            payload["tensor_summaries"][name] = _tensor_summary(value)
            payload["tensors"][name] = _tensor_payload(value)
        else:
            payload["tensors"][name] = value

    torch.save(payload, path)
    return path
