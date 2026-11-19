"""
BentoML serving definitions for the retrained CIFAR-100 ResNet model.

Exposes two comparable service classes sharing an identical TorchScript
inference engine:
  - CIFAR100ServiceNoBatch: batchable=False (per-request inference)
  - CIFAR100ServiceAdaptiveBatch: batchable=True, batch_dim=0,
    max_batch_size=32, max_latency_ms=1000 (adaptive dynamic batching)
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import bentoml
import numpy as np
import torch
import torch.nn.functional as F

SERVICE_DIR = Path(__file__).resolve().parent
MODELS_DIR = SERVICE_DIR / "models"
DEFAULT_TS_PATH = MODELS_DIR / "cifar100_resnet_scripted.pt"
DEFAULT_META_PATH = MODELS_DIR / "model_metadata.json"

MAX_BATCH_SIZE = int(os.getenv("BENTO_MAX_BATCH_SIZE", "32"))
MAX_LATENCY_MS = int(os.getenv("BENTO_MAX_LATENCY_MS", "1000"))
CPU_THREADS = int(os.getenv("TORCH_NUM_THREADS", "4"))


class _CIFAR100InferenceEngine:
    """Shared TorchScript CPU inference engine and batch-size telemetry recorder."""

    def __init__(self, batching_mode: str) -> None:
        self.batching_mode = batching_mode
        if not torch.cuda.is_available():
            torch.set_num_threads(CPU_THREADS)

        ts_path, meta_path = self._resolve_artifact_paths()
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata: dict[str, Any] = json.load(f)

        model = torch.jit.load(str(ts_path), map_location="cpu").eval()
        self.model = torch.jit.optimize_for_inference(model)

        # Warm up JIT / oneDNN kernels across typical batch sizes
        with torch.inference_mode():
            for b in (1, 8, 32):
                _ = self.model(torch.zeros(b, 3, 32, 32, dtype=torch.float32))

        self.batch_sizes: list[int] = []
        self.total_batches: int = 0
        self.total_samples: int = 0

    @staticmethod
    def _resolve_artifact_paths() -> tuple[Path, Path]:
        if DEFAULT_TS_PATH.exists() and DEFAULT_META_PATH.exists():
            return DEFAULT_TS_PATH, DEFAULT_META_PATH

        bento_model = bentoml.models.get("cifar100_resnet:latest")
        return (
            Path(bento_model.path_of("cifar100_resnet_scripted.pt")),
            Path(bento_model.path_of("model_metadata.json")),
        )

    def predict_probs(self, images: np.ndarray) -> np.ndarray:
        """Runs batched inference and returns softmax probabilities of shape (B, 100)."""
        arr = np.asarray(images, dtype=np.float32)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)
        if arr.ndim != 4 or arr.shape[1:] != (3, 32, 32):
            raise ValueError(
                f"Expected input shape (N, 3, 32, 32) or (3, 32, 32), got {arr.shape}"
            )

        batch_size = int(arr.shape[0])
        self.batch_sizes.append(batch_size)
        self.total_batches += 1
        self.total_samples += batch_size

        tensor = torch.from_numpy(np.ascontiguousarray(arr))
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    def get_metadata(self) -> dict[str, Any]:
        return {
            "batching_mode": self.batching_mode,
            "cpu_threads": CPU_THREADS,
            "max_batch_size": MAX_BATCH_SIZE if self.batching_mode == "adaptive" else 1,
            "max_latency_ms": MAX_LATENCY_MS if self.batching_mode == "adaptive" else 0,
            **self.metadata,
        }

    def get_batch_stats(self) -> dict[str, Any]:
        if not self.batch_sizes:
            return {
                "batching_mode": self.batching_mode,
                "total_batches": 0,
                "total_samples": 0,
                "mean_batch_size": 0.0,
                "max_batch_size_observed": 0,
                "min_batch_size_observed": 0,
                "histogram": {},
            }
        hist = {str(k): int(v) for k, v in sorted(Counter(self.batch_sizes).items())}
        return {
            "batching_mode": self.batching_mode,
            "total_batches": self.total_batches,
            "total_samples": self.total_samples,
            "mean_batch_size": round(sum(self.batch_sizes) / len(self.batch_sizes), 4),
            "max_batch_size_observed": max(self.batch_sizes),
            "min_batch_size_observed": min(self.batch_sizes),
            "histogram": hist,
        }

    def reset_stats(self) -> dict[str, str]:
        self.batch_sizes.clear()
        self.total_batches = 0
        self.total_samples = 0
        return {"status": "reset", "batching_mode": self.batching_mode}


@bentoml.service(
    name="cifar100_resnet_no_batch",
    workers=1,
    resources={"cpu": "4"},
    traffic={"timeout": 60},
)
class CIFAR100ServiceNoBatch:
    """BentoML service with dynamic batching disabled (batchable=False)."""

    def __init__(self) -> None:
        self.engine = _CIFAR100InferenceEngine(batching_mode="disabled")

    @bentoml.api(batchable=False)
    def predict(self, images: np.ndarray) -> np.ndarray:
        return self.engine.predict_probs(images)

    @bentoml.api
    def metadata(self) -> dict[str, Any]:
        return self.engine.get_metadata()

    @bentoml.api
    def batch_stats(self) -> dict[str, Any]:
        return self.engine.get_batch_stats()

    @bentoml.api
    def reset_stats(self) -> dict[str, str]:
        return self.engine.reset_stats()


@bentoml.service(
    name="cifar100_resnet_adaptive_batch",
    workers=1,
    resources={"cpu": "4"},
    traffic={"timeout": 60},
)
class CIFAR100ServiceAdaptiveBatch:
    """BentoML service with adaptive dynamic batching enabled."""

    def __init__(self) -> None:
        self.engine = _CIFAR100InferenceEngine(batching_mode="adaptive")

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=MAX_BATCH_SIZE,
        max_latency_ms=MAX_LATENCY_MS,
    )
    def predict(self, images: np.ndarray) -> np.ndarray:
        return self.engine.predict_probs(images)

    @bentoml.api
    def metadata(self) -> dict[str, Any]:
        return self.engine.get_metadata()

    @bentoml.api
    def batch_stats(self) -> dict[str, Any]:
        return self.engine.get_batch_stats()

    @bentoml.api
    def reset_stats(self) -> dict[str, str]:
        return self.engine.reset_stats()
