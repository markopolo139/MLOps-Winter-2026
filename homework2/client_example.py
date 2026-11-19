"""
Client request example for the CIFAR-100 BentoML inference service.

Loads a real CIFAR-100 test image, applies the exact evaluation normalization,
saves a reusable JSON payload (`sample_request.json` for cURL), sends a POST
request to `/predict`, and prints the top-5 predicted classes and latency.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests
import torchvision.transforms as T
from PIL import Image
from torchvision.datasets import CIFAR100

HW2_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HW2_DIR.parent.parent / "homework1" / "156042_156053" / "data"

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def prepare_sample(
    data_dir: Path, sample_idx: int
) -> tuple[np.ndarray, int, str, list[str]]:
    """Loads and preprocesses a single CIFAR-100 test image."""
    dataset = CIFAR100(root=str(data_dir), train=False, download=False)
    img_arr = dataset.data[sample_idx]
    true_label = int(dataset.targets[sample_idx])
    class_names = list(dataset.classes)

    eval_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
        ]
    )
    tensor = eval_transform(Image.fromarray(img_arr)).unsqueeze(0).numpy().astype(np.float32)
    return tensor, true_label, class_names[true_label], class_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Send an inference request to BentoML service")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:3000", help="Base BentoML URL")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="CIFAR-100 data dir")
    parser.add_argument("--sample-idx", type=int, default=7, help="Test dataset sample index")
    parser.add_argument(
        "--save-json",
        type=Path,
        default=HW2_DIR / "sample_request.json",
        help="Path to save JSON payload for cURL",
    )
    args = parser.parse_args()

    image_batch, true_id, true_name, class_names = prepare_sample(args.data_dir, args.sample_idx)
    payload = {"images": image_batch.tolist()}

    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    meta_resp = requests.post(f"{args.url}/metadata", json={}, timeout=10.0)
    meta_resp.raise_for_status()
    service_meta = meta_resp.json()

    t0 = time.perf_counter()
    pred_resp = requests.post(f"{args.url}/predict", json=payload, timeout=10.0)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    pred_resp.raise_for_status()

    probs = np.asarray(pred_resp.json(), dtype=np.float32)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    pred_id = int(top5_idx[0])

    print(f"Service URL      : {args.url}")
    print(f"Batching Mode    : {service_meta.get('batching_mode')}")
    print(f"Input Shape      : {list(image_batch.shape)}")
    print(f"Saved cURL JSON  : {args.save_json}")
    print(f"Sample Index     : {args.sample_idx}")
    print(f"Ground Truth     : [{true_id}] {true_name}")
    print(f"Top-1 Prediction : [{pred_id}] {class_names[pred_id]} ({probs[pred_id] * 100:.2f}%)")
    print(f"Round-trip Time  : {latency_ms:.2f} ms")
    print("\nTop-5 Predictions:")
    for rank, idx in enumerate(top5_idx, start=1):
        print(f"  {rank}. [{int(idx):2d}] {class_names[int(idx)]:20s} : {probs[int(idx)] * 100:6.2f}%")


if __name__ == "__main__":
    main()
