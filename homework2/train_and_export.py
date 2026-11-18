"""
Retrains the Homework 1 CIFAR-100 ResNet model using the preserved Optuna-optimized
hyperparameters and exports artifacts for BentoML serving:
  - PyTorch Lightning checkpoint (.ckpt)
  - Pure PyTorch state_dict + metadata (.pt / .json)
  - TorchScript traced model (.pt) verified across batch sizes [1, 8, 32]
  - BentoML model store entry (`cifar100_resnet`)
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

HW2_DIR = Path(__file__).resolve().parent
HW1_DIR = HW2_DIR.parent.parent / "homework1" / "156042_156053"
os.environ.setdefault("BENTOML_HOME", str(HW2_DIR / ".bentoml"))

sys.path.insert(0, str(HW2_DIR))

import bentoml
import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from torchvision.datasets import CIFAR100

from src.data_module import CIFAR100DataModule
from src.model import CIFAR100LightningModule
from src.utils import MetricsHistoryCallback, plot_and_save_curves, setup_mlflow_logger


class EpochSummaryCallback(Callback):
    """Prints concise per-epoch timing and metric progress for long CPU runs."""

    def __init__(self, max_epochs: int) -> None:
        super().__init__()
        self.max_epochs = max_epochs
        self.epoch_start = 0.0

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.epoch_start = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        total_batches = trainer.num_training_batches
        if (batch_idx + 1) % 150 == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.perf_counter() - self.epoch_start
            print(
                f"  [Epoch {trainer.current_epoch + 1}/{self.max_epochs}] "
                f"batch {batch_idx + 1}/{total_batches} ({elapsed:.1f}s elapsed)",
                flush=True,
            )

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        elapsed = time.perf_counter() - self.epoch_start
        m = trainer.callback_metrics
        tr_loss = float(m["train/loss"]) if "train/loss" in m else float("nan")
        val_loss = float(m["val/loss"]) if "val/loss" in m else float("nan")
        val_acc = float(m["val/acc"]) * 100.0 if "val/acc" in m else float("nan")
        val_f1 = float(m["val/f1_macro"]) * 100.0 if "val/f1_macro" in m else float("nan")
        print(
            f"[Epoch {trainer.current_epoch + 1}/{self.max_epochs} done in {elapsed:.1f}s] "
            f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.2f}% | val_f1_macro={val_f1:.2f}%",
            flush=True,
        )


def load_optimized_params(hpo_summary_path: Path) -> dict:
    """Loads best hyperparameters saved by Homework 1 Optuna study."""
    with open(hpo_summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return summary["best_params"]


def verify_torchscript_equivalence(
    eager_model: torch.nn.Module,
    ts_model: torch.jit.ScriptModule,
    batch_sizes: tuple[int, ...] = (1, 8, 32),
    atol: float = 1e-5,
) -> dict[int, float]:
    """Verifies TorchScript logits match eager PyTorch logits across batch sizes."""
    eager_model.eval()
    ts_model.eval()
    max_diffs: dict[int, float] = {}
    with torch.inference_mode():
        for b in batch_sizes:
            x = torch.randn(b, 3, 32, 32)
            eager_out = eager_model(x)
            ts_out = ts_model(x)
            diff = float((eager_out - ts_out).abs().max().item())
            max_diffs[b] = diff
            if not torch.allclose(eager_out, ts_out, atol=atol, rtol=1e-5):
                raise AssertionError(
                    f"TorchScript mismatch at batch_size={b}: max_diff={diff:.2e} > {atol}"
                )
    return max_diffs


def register_bentoml_model(
    ts_path: Path,
    metadata_path: Path,
    model_metadata: dict,
) -> str:
    """Registers the exported TorchScript model into the BentoML local model store."""
    with bentoml.models.create(
        name="cifar100_resnet",
        module="torch.jit",
        signatures={"__call__": {"batchable": True, "batch_dim": 0}},
        labels={
            "dataset": "cifar100",
            "framework": "pytorch-lightning",
            "architecture": "CIFAR100ResNet",
        },
        metadata=model_metadata,
    ) as bento_model:
        shutil.copy2(ts_path, bento_model.path_of("cifar100_resnet_scripted.pt"))
        shutil.copy2(metadata_path, bento_model.path_of("model_metadata.json"))
        return str(bento_model.tag)


def train_and_export(
    hpo_summary_path: Path,
    data_dir: Path,
    epochs: int = 8,
    seed: int = 42,
    num_workers: int = 4,
    fast_dev_run: bool = False,
) -> dict:
    """Retrains the tuned CIFAR-100 model and exports all serving artifacts."""
    best_params = load_optimized_params(hpo_summary_path)
    print(f"Loaded optimized hyperparameters from {hpo_summary_path}:")
    print(json.dumps(best_params, indent=2), flush=True)

    L.seed_everything(seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    else:
        torch.set_num_threads(min(4, os.cpu_count() or 4))

    pin_memory = torch.cuda.is_available()
    data_module = CIFAR100DataModule(
        data_dir=str(data_dir),
        val_split=0.2,
        batch_size=int(best_params["batch_size"]),
        num_workers=num_workers,
        pin_memory=pin_memory,
        augment=True,
        seed=seed,
    )

    model = CIFAR100LightningModule(
        num_classes=100,
        base_channels=64,
        lr=float(best_params["lr"]),
        optimizer_name=str(best_params["optimizer_name"]),
        weight_decay=float(best_params["weight_decay"]),
        dropout_rate=float(best_params["dropout_rate"]),
        use_scheduler=True,
        max_epochs=epochs,
    )

    models_dir = HW2_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    hw1_ckpt_dir = HW1_DIR / "checkpoints"
    hw1_ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics_cb = MetricsHistoryCallback()
    epoch_cb = EpochSummaryCallback(max_epochs=epochs)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(models_dir),
        monitor="val/f1_macro",
        mode="max",
        filename="best-epoch={epoch:02d}",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    mlflow_db = HW2_DIR / "mlflow.db"
    mlflow_logger = setup_mlflow_logger(
        experiment_name="cifar100_hw2_retrain",
        tracking_uri=f"sqlite:///{mlflow_db}",
        run_name="tuned_model_retrain",
        tags={"model": "CIFAR100ResNet", "source": "hw1_hpo_summary"},
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=mlflow_logger,
        callbacks=[checkpoint_cb, lr_monitor, metrics_cb, epoch_cb],
        log_every_n_steps=50,
        enable_progress_bar=False,
        default_root_dir=str(HW2_DIR),
        fast_dev_run=fast_dev_run,
    )

    t0 = time.perf_counter()
    trainer.fit(model, datamodule=data_module)
    training_time_sec = time.perf_counter() - t0
    print(f"\nTraining completed in {training_time_sec:.1f}s ({training_time_sec / 60:.2f} min)", flush=True)

    if fast_dev_run:
        test_results = trainer.test(model, datamodule=data_module)
        best_ckpt_path = models_dir / "fast_dev_run.ckpt"
        trainer.save_checkpoint(str(best_ckpt_path))
    else:
        test_results = trainer.test(model, datamodule=data_module, ckpt_path="best")
        best_ckpt_path = Path(checkpoint_cb.best_model_path)
    test_metrics = test_results[0] if test_results else {}
    best_val_f1 = float(checkpoint_cb.best_model_score) if checkpoint_cb.best_model_score else 0.0
    best_val_acc = max(metrics_cb.history["val/acc"]) if metrics_cb.history["val/acc"] else 0.0

    # Copy Lightning checkpoint to canonical HW1 and HW2 locations
    hw2_ckpt_dest = models_dir / "best_tuned_model.ckpt"
    hw1_ckpt_dest = hw1_ckpt_dir / "best_tuned_model.ckpt"
    if best_ckpt_path.resolve() != hw2_ckpt_dest.resolve():
        shutil.copy2(best_ckpt_path, hw2_ckpt_dest)
        best_ckpt_path.unlink(missing_ok=True)
    shutil.copy2(hw2_ckpt_dest, hw1_ckpt_dest)
    print(f"Saved Lightning checkpoints to:\n  - {hw2_ckpt_dest}\n  - {hw1_ckpt_dest}", flush=True)

    # Plot training/validation curves
    plots_dir = HW2_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if metrics_cb.history["train/loss"] and metrics_cb.history["val/loss"]:
        plot_and_save_curves(
            train_losses=metrics_cb.history["train/loss"],
            val_losses=metrics_cb.history["val/loss"],
            train_accs=metrics_cb.history["train/acc"],
            val_accs=metrics_cb.history["val/acc"],
            train_f1s=metrics_cb.history["train/f1_macro"],
            val_f1s=metrics_cb.history["val/f1_macro"],
            output_path=str(plots_dir / "retrained_model_curves.png"),
            title_suffix="(HW2 Retrained Tuned Model)",
        )

    # Load best checkpoint on CPU for clean export
    best_lightning_model = CIFAR100LightningModule.load_from_checkpoint(
        str(hw2_ckpt_dest),
        map_location="cpu",
    )
    eager_resnet = best_lightning_model.model.eval().cpu().to(memory_format=torch.contiguous_format)

    # Load CIFAR-100 class names from existing dataset metadata
    cifar_raw = CIFAR100(root=str(data_dir), train=False, download=False)
    class_names = list(cifar_raw.classes)

    model_metadata = {
        "architecture": "CIFAR100ResNet",
        "num_classes": 100,
        "base_channels": 64,
        "input_shape": [1, 3, 32, 32],
        "normalization": {
            "mean": [0.5071, 0.4867, 0.4408],
            "std": [0.2675, 0.2565, 0.2761],
        },
        "hyperparameters": best_params,
        "epochs": epochs,
        "seed": seed,
        "training_time_sec": round(training_time_sec, 2),
        "val_f1_macro": round(best_val_f1, 6),
        "val_acc": round(best_val_acc, 6),
        "test_f1_macro": round(float(test_metrics.get("test/f1_macro", 0.0)), 6),
        "test_acc": round(float(test_metrics.get("test/acc", 0.0)), 6),
        "test_loss": round(float(test_metrics.get("test/loss", 0.0)), 6),
        "class_names": class_names,
    }

    state_path = models_dir / "cifar100_resnet_state.pt"
    torch.save(
        {
            "state_dict": eager_resnet.state_dict(),
            "metadata": model_metadata,
        },
        state_path,
    )

    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    # Export TorchScript traced model and validate numerical equivalence
    ts_path = models_dir / "cifar100_resnet_scripted.pt"
    dummy_input = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        traced_model = torch.jit.trace(eager_resnet, dummy_input)
    torch.jit.save(traced_model, str(ts_path))

    reloaded_ts = torch.jit.load(str(ts_path), map_location="cpu").eval()
    max_diffs = verify_torchscript_equivalence(eager_resnet, reloaded_ts)
    print(f"Verified TorchScript equivalence across batch sizes: {max_diffs}", flush=True)

    # Register in BentoML local store (store concise metadata without huge list in BentoML metadata)
    bento_meta = {k: v for k, v in model_metadata.items() if k != "class_names"}
    bento_tag = register_bentoml_model(ts_path, metadata_path, bento_meta)
    print(f"Registered BentoML model: {bento_tag}", flush=True)

    return {
        "best_ckpt": str(hw2_ckpt_dest),
        "state_path": str(state_path),
        "torchscript_path": str(ts_path),
        "metadata_path": str(metadata_path),
        "bento_tag": bento_tag,
        "metrics": {
            "val_f1_macro": best_val_f1,
            "val_acc": best_val_acc,
            "test_f1_macro": float(test_metrics.get("test/f1_macro", 0.0)),
            "test_acc": float(test_metrics.get("test/acc", 0.0)),
            "test_loss": float(test_metrics.get("test/loss", 0.0)),
            "training_time_sec": training_time_sec,
        },
        "torchscript_max_diffs": max_diffs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain tuned HW1 model and export for BentoML")
    parser.add_argument(
        "--hpo-summary",
        type=Path,
        default=HW1_DIR / "hpo_summary.json",
        help="Path to HW1 hpo_summary.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=HW1_DIR / "data",
        help="CIFAR-100 dataset directory",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--fast-dev-run", action="store_true", help="Smoke-test 1 batch end-to-end")
    args = parser.parse_args()

    result = train_and_export(
        hpo_summary_path=args.hpo_summary,
        data_dir=args.data_dir,
        epochs=args.epochs,
        seed=args.seed,
        num_workers=args.num_workers,
        fast_dev_run=args.fast_dev_run,
    )
    print("\nExport Summary:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
