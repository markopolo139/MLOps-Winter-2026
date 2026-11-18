import os
from typing import Optional, List, Dict
import matplotlib.pyplot as plt
import mlflow
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger


class MetricsHistoryCallback(Callback):
    """
    Collects epoch-level metrics for plotting and offline reporting.
    """

    def __init__(self):
        super().__init__()
        self.history = {
            "train/loss": [],
            "train/acc": [],
            "train/f1_macro": [],
            "val/loss": [],
            "val/acc": [],
            "val/f1_macro": [],
        }

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        logged = trainer.callback_metrics
        for k in ["train/loss", "train/acc", "train/f1_macro"]:
            if k in logged:
                self.history[k].append(float(logged[k].detach().cpu().item()))

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        logged = trainer.callback_metrics
        for k in ["val/loss", "val/acc", "val/f1_macro"]:
            if k in logged:
                self.history[k].append(float(logged[k].detach().cpu().item()))


def setup_mlflow_logger(
    experiment_name: str = "cifar100_hpo",
    tracking_uri: str = "sqlite:///mlflow.db",
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> MLFlowLogger:
    """
    Sets up an MLflow logger for PyTorch Lightning.
    Uses SQLite database backend (sqlite:///mlflow.db) as recommended by modern MLflow.
    """
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name=run_name,
        tags=tags,
    )
    return logger


def plot_and_save_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    train_f1s: List[float],
    val_f1s: List[float],
    output_path: str = "training_curves.png",
    title_suffix: str = "",
) -> str:
    """
    Plots training and validation curves side-by-side (Loss, Accuracy, Macro-F1).
    Saves figure to output_path and returns the path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, train_losses, "b-o", label="Train Loss")
    axes[0].plot(epochs, val_losses, "r--s", label="Val Loss")
    axes[0].set_title(f"Loss Curves {title_suffix}".strip())
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, [a * 100 for a in train_accs], "b-o", label="Train Acc")
    axes[1].plot(epochs, [a * 100 for a in val_accs], "r--s", label="Val Acc")
    axes[1].set_title(f"Accuracy Curves (%) {title_suffix}".strip())
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # F1-Score
    axes[2].plot(epochs, [f * 100 for f in train_f1s], "b-o", label="Train Macro-F1")
    axes[2].plot(epochs, [f * 100 for f in val_f1s], "r--s", label="Val Macro-F1")
    axes[2].set_title(f"Macro F1-Score (%) {title_suffix}".strip())
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score (%)")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path

