from __future__ import annotations

import csv
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_utf8_preserving_newlines(path: Path) -> tuple[str, str, bool]:
    raw = path.read_bytes()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Input is not valid UTF-8: {path}: {exc}") from exc
    return text, "UTF-8-BOM" if has_bom else "UTF-8", has_bom


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def run_id(config: dict[str, Any], input_paths: list[Path]) -> str:
    payload = {
        "project_version": config.get("project_version"),
        "model": config["model"],
        "inference": config["inference"],
        "inputs": [
            {"name": path.name, "sha256": sha256_file(path)} for path in input_paths
        ],
        "versions": {
            "vieneu": package_version("vieneu"),
            "sea-g2p": package_version("sea-g2p"),
            "onnxruntime": package_version("onnxruntime"),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()[:12]


def stable_seed(base_seed: int, chunk_id: str) -> int:
    suffix = int(hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:8], 16)
    return (base_seed + suffix) % (2**32)


def environment_snapshot() -> dict[str, Any]:
    try:
        nvidia = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        nvidia = "unavailable"

    ram_gb: float | str = "unknown"
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            ram_gb = round(status.total_physical / (1024**3), 2)

    return {
        "python": platform.python_version(),
        "python_executable": os.path.abspath(os.sys.executable),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpu": os.cpu_count(),
        "ram_gb": ram_gb,
        "gpu": nvidia or "unavailable",
        "cuda": "not-used (ONNX CPU baseline)",
        "packages": {
            name: package_version(name)
            for name in (
                "vieneu",
                "sea-g2p",
                "onnxruntime",
                "onnxruntime-gpu",
                "numpy",
                "soundfile",
                "PyYAML",
            )
        },
    }

