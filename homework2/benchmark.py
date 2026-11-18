"""
Dynamic (adaptive) batching benchmark for BentoML CIFAR-100 inference services.

Follows the HOMEWORK2.md benchmark methodology:
  - Fixed normalized CIFAR-100 sample payload (1 request = 1 sample, shape (1, 3, 32, 32))
  - Warm-up phase (default 10s) discarded before measurements
  - Fixed 60-second measurement window per run
  - Concurrency sweep across [1, 8, 32] concurrent async clients
  - 3 repeats per (mode, concurrency), reporting mean +- std
  - Records throughput (samples/s), p50 / p95 / p99 latencies (ms), and
    BentoML adaptive batch size histograms from /batch_stats
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import bentoml
import httpx
import matplotlib.pyplot as plt
import numpy as np
import torch

from client_example import DEFAULT_DATA_DIR, prepare_sample

HW2_DIR = Path(__file__).resolve().parent
PLOTS_DIR = HW2_DIR / "plots"

SERVICE_TARGETS = {
    "no_batch": "service:CIFAR100ServiceNoBatch",
    "adaptive_batch": "service:CIFAR100ServiceAdaptiveBatch",
}


def collect_hardware_info() -> dict[str, Any]:
    """Collects CPU model, core counts, and library versions for fixed-condition reporting."""
    cpu_model = platform.processor() or "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU-only)",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "bentoml_version": bentoml.__version__,
        "os": platform.platform(),
    }


def start_bentoml_server(
    service_target: str,
    port: int,
    max_batch_size: int,
    max_latency_ms: int,
    cpu_threads: int,
) -> tuple[subprocess.Popen, Path]:
    """Starts a BentoML HTTP server subprocess and waits until /readyz returns 200."""
    bentoml_bin = Path(sys.executable).parent / "bentoml"
    env = os.environ.copy()
    env.setdefault("BENTOML_HOME", str(HW2_DIR / ".bentoml"))
    env["BENTO_MAX_BATCH_SIZE"] = str(max_batch_size)
    env["BENTO_MAX_LATENCY_MS"] = str(max_latency_ms)
    env["TORCH_NUM_THREADS"] = str(cpu_threads)

    safe_name = service_target.split(":")[-1]
    log_path = Path(f"/tmp/bentoml_{safe_name}_{port}.log")
    log_fp = open(log_path, "w", encoding="utf-8")

    cmd = [
        str(bentoml_bin),
        "serve",
        service_target,
        "--working-dir",
        str(HW2_DIR),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Parent process no longer needs this fd; child inherited it.
    log_fp.close()

    ready_url = f"http://127.0.0.1:{port}/readyz"
    deadline = time.perf_counter() + 45.0
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            output = log_path.read_text(encoding="utf-8", errors="ignore")[-4000:]
            raise RuntimeError(f"BentoML server exited prematurely ({proc.returncode}):\n{output}")
        try:
            resp = httpx.get(ready_url, timeout=1.0)
            if resp.status_code == 200:
                return proc, log_path
        except httpx.HTTPError:
            pass
        time.sleep(0.5)

    stop_bentoml_server(proc)
    raise TimeoutError(f"Timed out waiting for BentoML server {service_target} on port {port}")


def stop_bentoml_server(proc: subprocess.Popen) -> None:
    """Gracefully terminates a BentoML subprocess."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5.0)


async def run_load_window(
    base_url: str,
    payload_bytes: bytes,
    concurrency: int,
    duration_sec: float,
    record_stats: bool = True,
) -> dict[str, Any]:
    """Runs `concurrency` async workers posting single-sample requests for `duration_sec`."""
    predict_url = f"{base_url}/predict"
    headers = {"Content-Type": "application/json"}
    limits = httpx.Limits(
        max_connections=max(64, concurrency * 2),
        max_keepalive_connections=max(64, concurrency * 2),
    )

    latencies_ms: list[float] = []
    errors = 0

    async with httpx.AsyncClient(limits=limits, timeout=60.0) as client:
        if record_stats:
            await client.post(f"{base_url}/reset_stats", json={})

        window_start = time.perf_counter()
        deadline = window_start + duration_sec

        async def worker() -> None:
            nonlocal errors
            while True:
                now = time.perf_counter()
                if now >= deadline:
                    break
                t0 = now
                try:
                    resp = await client.post(predict_url, content=payload_bytes, headers=headers)
                    t1 = time.perf_counter()
                    if resp.status_code == 200:
                        latencies_ms.append((t1 - t0) * 1000.0)
                    else:
                        errors += 1
                except Exception:
                    errors += 1

        await asyncio.gather(*(worker() for _ in range(concurrency)))
        actual_elapsed = time.perf_counter() - window_start

        batch_stats: dict[str, Any] = {}
        if record_stats:
            stats_resp = await client.post(f"{base_url}/batch_stats", json={})
            if stats_resp.status_code == 200:
                batch_stats = stats_resp.json()

    completed = len(latencies_ms)
    throughput = completed / actual_elapsed if actual_elapsed > 0 else 0.0
    if latencies_ms:
        arr = np.asarray(latencies_ms, dtype=np.float64)
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        mean_lat = float(np.mean(arr))
    else:
        p50 = p95 = p99 = mean_lat = float("nan")

    return {
        "concurrency": concurrency,
        "duration_requested_sec": duration_sec,
        "actual_elapsed_sec": round(actual_elapsed, 4),
        "completed_samples": completed,
        "errors": errors,
        "throughput_samples_per_sec": round(throughput, 4),
        "latency_mean_ms": round(mean_lat, 4),
        "latency_p50_ms": round(p50, 4),
        "latency_p95_ms": round(p95, 4),
        "latency_p99_ms": round(p99, 4),
        "batch_stats": batch_stats,
    }


def summarize_repeats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Computes mean and sample std across repeat measurements."""

    def stats_for(key: str) -> dict[str, float]:
        vals = [float(r[key]) for r in runs]
        return {
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
        }

    merged_hist: Counter[int] = Counter()
    total_batches = 0
    total_samples = 0
    for r in runs:
        bs = r.get("batch_stats") or {}
        total_batches += int(bs.get("total_batches", 0))
        total_samples += int(bs.get("total_samples", 0))
        for k, v in (bs.get("histogram") or {}).items():
            merged_hist[int(k)] += int(v)

    mean_batch_size = (total_samples / total_batches) if total_batches > 0 else 0.0
    max_batch_observed = max(merged_hist.keys()) if merged_hist else 0

    return {
        "concurrency": runs[0]["concurrency"],
        "repeats": len(runs),
        "throughput_samples_per_sec": stats_for("throughput_samples_per_sec"),
        "latency_mean_ms": stats_for("latency_mean_ms"),
        "latency_p50_ms": stats_for("latency_p50_ms"),
        "latency_p95_ms": stats_for("latency_p95_ms"),
        "latency_p99_ms": stats_for("latency_p99_ms"),
        "total_errors": sum(int(r["errors"]) for r in runs),
        "batch_stats_aggregated": {
            "total_batches": total_batches,
            "total_samples": total_samples,
            "mean_batch_size": round(mean_batch_size, 4),
            "max_batch_size_observed": max_batch_observed,
            "histogram": {str(k): int(v) for k, v in sorted(merged_hist.items())},
        },
        "runs": runs,
    }


async def benchmark_running_service(
    mode: str,
    base_url: str,
    payload_bytes: bytes,
    concurrencies: list[int],
    warmup_sec: float,
    duration_sec: float,
    repeats: int,
) -> dict[str, Any]:
    """Benchmarks a single running service across all concurrency levels."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        meta_resp = await client.post(f"{base_url}/metadata", json={})
        meta_resp.raise_for_status()
        service_meta = meta_resp.json()

    print(f"\n{'=' * 68}", flush=True)
    print(
        f"Benchmarking mode={mode} | service_batching={service_meta.get('batching_mode')} "
        f"| max_batch_size={service_meta.get('max_batch_size')} "
        f"| max_latency_ms={service_meta.get('max_latency_ms')}",
        flush=True,
    )
    print(f"{'=' * 68}", flush=True)

    results_by_concurrency: dict[str, Any] = {}
    for conc in concurrencies:
        if warmup_sec > 0:
            print(f"[mode={mode} C={conc}] Warm-up for {warmup_sec:.0f}s...", flush=True)
            await run_load_window(
                base_url=base_url,
                payload_bytes=payload_bytes,
                concurrency=conc,
                duration_sec=warmup_sec,
                record_stats=False,
            )

        repeat_runs: list[dict[str, Any]] = []
        for rep in range(1, repeats + 1):
            print(
                f"[mode={mode} C={conc}] Repeat {rep}/{repeats} ({duration_sec:.0f}s window)...",
                flush=True,
            )
            run_res = await run_load_window(
                base_url=base_url,
                payload_bytes=payload_bytes,
                concurrency=conc,
                duration_sec=duration_sec,
                record_stats=True,
            )
            repeat_runs.append(run_res)
            bs = run_res["batch_stats"]
            print(
                f"  -> throughput={run_res['throughput_samples_per_sec']:.2f} samples/s | "
                f"p50={run_res['latency_p50_ms']:.2f} ms | "
                f"p95={run_res['latency_p95_ms']:.2f} ms | "
                f"p99={run_res['latency_p99_ms']:.2f} ms | "
                f"mean_batch={bs.get('mean_batch_size', 1.0):.2f} | "
                f"errors={run_res['errors']}",
                flush=True,
            )

        summary = summarize_repeats(repeat_runs)
        results_by_concurrency[str(conc)] = summary
        print(
            f"[mode={mode} C={conc} SUMMARY] "
            f"throughput={summary['throughput_samples_per_sec']['mean']:.2f} +- "
            f"{summary['throughput_samples_per_sec']['std']:.2f} samples/s | "
            f"p95={summary['latency_p95_ms']['mean']:.2f} +- "
            f"{summary['latency_p95_ms']['std']:.2f} ms",
            flush=True,
        )

    return {
        "mode": mode,
        "service_metadata": {
            k: v for k, v in service_meta.items() if k != "class_names"
        },
        "by_concurrency": results_by_concurrency,
    }


def generate_plots(all_results: dict[str, Any], concurrencies: list[int]) -> list[str]:
    """Generates throughput, latency, and adaptive batch size distribution plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    modes = all_results["modes"]
    has_both = "no_batch" in modes and "adaptive_batch" in modes
    saved_paths: list[str] = []

    if has_both:
        nb = modes["no_batch"]["by_concurrency"]
        ab = modes["adaptive_batch"]["by_concurrency"]

        nb_tput = [nb[str(c)]["throughput_samples_per_sec"]["mean"] for c in concurrencies]
        nb_tput_err = [nb[str(c)]["throughput_samples_per_sec"]["std"] for c in concurrencies]
        ab_tput = [ab[str(c)]["throughput_samples_per_sec"]["mean"] for c in concurrencies]
        ab_tput_err = [ab[str(c)]["throughput_samples_per_sec"]["std"] for c in concurrencies]

        x = np.arange(len(concurrencies))
        width = 0.35

        # 1. Throughput vs Concurrency
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        bars1 = ax.bar(
            x - width / 2, nb_tput, width, yerr=nb_tput_err, capsize=5,
            label="No Batching (batchable=False)", color="#5C7AEA", alpha=0.9
        )
        bars2 = ax.bar(
            x + width / 2, ab_tput, width, yerr=ab_tput_err, capsize=5,
            label=(
                f"Adaptive Batching (max_batch={all_results['config']['max_batch_size']}, "
                f"max_lat={all_results['config']['max_latency_ms']}ms)"
            ),
            color="#1FBF75", alpha=0.9
        )
        ax.set_title("Inference Throughput vs Concurrency (60s window x 3 repeats)", fontsize=13)
        ax.set_xlabel("Concurrent Clients", fontsize=11)
        ax.set_ylabel("Throughput [samples/s]", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in concurrencies])
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.legend()
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.tight_layout()
        tput_path = PLOTS_DIR / "throughput_vs_concurrency.png"
        plt.savefig(tput_path, dpi=200)
        plt.close()
        saved_paths.append(str(tput_path))

        # 2. Latency p50 & p95 vs Concurrency
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for ax, metric, title in [
            (axes[0], "latency_p50_ms", "Median Latency (p50)"),
            (axes[1], "latency_p95_ms", "Tail Latency (p95)"),
        ]:
            nb_lat = [nb[str(c)][metric]["mean"] for c in concurrencies]
            nb_err = [nb[str(c)][metric]["std"] for c in concurrencies]
            ab_lat = [ab[str(c)][metric]["mean"] for c in concurrencies]
            ab_err = [ab[str(c)][metric]["std"] for c in concurrencies]

            ax.errorbar(
                concurrencies, nb_lat, yerr=nb_err, marker="o", capsize=4,
                linewidth=2, label="No Batching", color="#5C7AEA"
            )
            ax.errorbar(
                concurrencies, ab_lat, yerr=ab_err, marker="s", capsize=4,
                linewidth=2, label="Adaptive Batching", color="#1FBF75"
            )
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Concurrent Clients", fontsize=11)
            ax.set_ylabel("Latency [ms]", fontsize=11)
            ax.set_xticks(concurrencies)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend()
        plt.tight_layout()
        lat_path = PLOTS_DIR / "latency_vs_concurrency.png"
        plt.savefig(lat_path, dpi=200)
        plt.close()
        saved_paths.append(str(lat_path))

    # 3. Adaptive Batch Size Distribution
    if "adaptive_batch" in modes:
        ab = modes["adaptive_batch"]["by_concurrency"]
        fig, axes = plt.subplots(1, len(concurrencies), figsize=(4.5 * len(concurrencies), 4.2))
        if len(concurrencies) == 1:
            axes = [axes]
        for ax, c in zip(axes, concurrencies):
            stats = ab[str(c)]["batch_stats_aggregated"]
            hist = stats.get("histogram", {})
            sizes = [int(k) for k in hist.keys()]
            counts = [int(v) for v in hist.values()]
            if sizes:
                ax.bar([str(s) for s in sizes], counts, color="#1FBF75", alpha=0.88)
            ax.set_title(
                f"Concurrency = {c}\n(mean batch={stats['mean_batch_size']:.2f}, "
                f"max={stats['max_batch_size_observed']})",
                fontsize=10.5,
            )
            ax.set_xlabel("Batch Size Formed", fontsize=10)
            ax.set_ylabel("Number of Batches", fontsize=10)
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        dist_path = PLOTS_DIR / "batch_size_distribution.png"
        plt.savefig(dist_path, dpi=200)
        plt.close()
        saved_paths.append(str(dist_path))

    return saved_paths


def print_comparison_table(all_results: dict[str, Any], concurrencies: list[int]) -> None:
    """Prints the HOMEWORK2.md required comparison table."""
    modes = all_results["modes"]
    if "no_batch" not in modes or "adaptive_batch" not in modes:
        return
    nb = modes["no_batch"]["by_concurrency"]
    ab = modes["adaptive_batch"]["by_concurrency"]

    print("\n" + "=" * 78)
    print("HOMEWORK 2 BENCHMARK COMPARISON TABLE")
    print("=" * 78)
    print(
        "| Concurrency | Throughput no batching [samples/s] | "
        "Throughput with batching [samples/s] | "
        "p95 latency no batching [ms] | p95 latency with batching [ms] |"
    )
    print("|---|---|---|---|---|")
    for c in concurrencies:
        nb_t = nb[str(c)]["throughput_samples_per_sec"]
        ab_t = ab[str(c)]["throughput_samples_per_sec"]
        nb_p95 = nb[str(c)]["latency_p95_ms"]
        ab_p95 = ab[str(c)]["latency_p95_ms"]
        print(
            f"| {c} | "
            f"{nb_t['mean']:.2f} ± {nb_t['std']:.2f} | "
            f"{ab_t['mean']:.2f} ± {ab_t['std']:.2f} | "
            f"{nb_p95['mean']:.2f} ± {nb_p95['std']:.2f} | "
            f"{ab_p95['mean']:.2f} ± {ab_p95['std']:.2f} |"
        )
    print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BentoML dynamic batching")
    parser.add_argument(
        "--mode",
        choices=["all", "no_batch", "adaptive_batch"],
        default="all",
        help="Which service mode(s) to benchmark",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional URL of an already running service (disables auto-spawn)",
    )
    parser.add_argument("--port", type=int, default=3000, help="Port for auto-spawned servers")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--warmup", type=float, default=10.0, help="Warm-up seconds per concurrency")
    parser.add_argument("--duration", type=float, default=60.0, help="Measurement window seconds")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per concurrency level")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-latency-ms", type=int, default=1000)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=HW2_DIR / "benchmark_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    if args.url and args.mode == "all":
        raise ValueError("Specify --mode no_batch or --mode adaptive_batch when passing --url.")

    sample_arr, true_id, true_name, _ = prepare_sample(args.data_dir, sample_idx=7)
    payload_bytes = json.dumps({"images": sample_arr.tolist()}).encode("utf-8")
    print(
        f"Prepared fixed payload: shape={list(sample_arr.shape)}, "
        f"true_label=[{true_id}] {true_name}, json_bytes={len(payload_bytes)}"
    )

    modes_to_run = ["no_batch", "adaptive_batch"] if args.mode == "all" else [args.mode]
    all_results: dict[str, Any] = {
        "hardware": collect_hardware_info(),
        "config": {
            "concurrencies": args.concurrencies,
            "warmup_sec": args.warmup,
            "duration_sec": args.duration,
            "repeats": args.repeats,
            "max_batch_size": args.max_batch_size,
            "max_latency_ms": args.max_latency_ms,
            "cpu_threads": args.cpu_threads,
            "payload_shape": list(sample_arr.shape),
            "payload_json_bytes": len(payload_bytes),
        },
        "modes": {},
    }

    for mode in modes_to_run:
        if args.url:
            mode_result = asyncio.run(
                benchmark_running_service(
                    mode=mode,
                    base_url=args.url.rstrip("/"),
                    payload_bytes=payload_bytes,
                    concurrencies=args.concurrencies,
                    warmup_sec=args.warmup,
                    duration_sec=args.duration,
                    repeats=args.repeats,
                )
            )
            all_results["modes"][mode] = mode_result
        else:
            proc, _log_path = start_bentoml_server(
                service_target=SERVICE_TARGETS[mode],
                port=args.port,
                max_batch_size=args.max_batch_size,
                max_latency_ms=args.max_latency_ms,
                cpu_threads=args.cpu_threads,
            )
            try:
                mode_result = asyncio.run(
                    benchmark_running_service(
                        mode=mode,
                        base_url=f"http://127.0.0.1:{args.port}",
                        payload_bytes=payload_bytes,
                        concurrencies=args.concurrencies,
                        warmup_sec=args.warmup,
                        duration_sec=args.duration,
                        repeats=args.repeats,
                    )
                )
                all_results["modes"][mode] = mode_result
            finally:
                stop_bentoml_server(proc)
                time.sleep(1.0)

    plot_paths = generate_plots(all_results, args.concurrencies)
    all_results["plots"] = plot_paths

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved benchmark results to {args.output}")
    print(f"Saved plots: {plot_paths}")
    print_comparison_table(all_results, args.concurrencies)


if __name__ == "__main__":
    main()
