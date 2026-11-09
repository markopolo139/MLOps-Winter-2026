import argparse
import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.data_module import CIFAR100DataModule
from src.model import CIFAR100LightningModule
from src.utils import setup_mlflow_logger, plot_and_save_curves, MetricsHistoryCallback


def train_model(
    lr: float = 1e-3,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-4,
    dropout_rate: float = 0.1,
    batch_size: int = 64,
    max_epochs: int = 10,
    experiment_name: str = "cifar100_hpo",
    run_name: str = "baseline",
    data_dir: str = "./data",
    seed: int = 42,
) -> dict:
    """
    Trains CIFAR-100 LightningModule with MLflow tracking and saves metrics/plots.
    """
    L.seed_everything(seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # 1. DataModule
    data_module = CIFAR100DataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=0.2,
        seed=seed,
    )

    # 2. LightningModule
    model = CIFAR100LightningModule(
        num_classes=100,
        base_channels=64,
        lr=lr,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate,
        use_scheduler=True,
        max_epochs=max_epochs,
    )

    # 3. Callbacks & Logger
    metrics_cb = MetricsHistoryCallback()
    checkpoint_cb = ModelCheckpoint(
        monitor="val/f1_macro",
        mode="max",
        filename="best-epoch={epoch:02d}",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    mlflow_logger = setup_mlflow_logger(
        experiment_name=experiment_name,
        tracking_uri="sqlite:///mlflow.db",
        run_name=run_name,
        tags={"model": "CIFAR100ResNet", "type": "single_run"},
    )

    # 4. Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=mlflow_logger,
        callbacks=[checkpoint_cb, lr_monitor, metrics_cb],
        log_every_n_steps=20,
    )

    # 5. Fit & Validate
    trainer.fit(model, datamodule=data_module)

    # 6. Test on test set with best checkpoint
    test_results = trainer.test(model, datamodule=data_module, ckpt_path="best")

    # 7. Generate and save training/val curve plot
    os.makedirs("./plots", exist_ok=True)
    plot_file = f"./plots/{run_name}_curves.png"
    if metrics_cb.history["train/loss"] and metrics_cb.history["val/loss"]:
        plot_and_save_curves(
            train_losses=metrics_cb.history["train/loss"],
            val_losses=metrics_cb.history["val/loss"],
            train_accs=metrics_cb.history["train/acc"],
            val_accs=metrics_cb.history["val/acc"],
            train_f1s=metrics_cb.history["train/f1_macro"],
            val_f1s=metrics_cb.history["val/f1_macro"],
            output_path=plot_file,
            title_suffix=f"({run_name})",
        )
        # Log plot to MLflow
        mlflow_logger.experiment.log_artifact(
            run_id=mlflow_logger.run_id,
            local_path=plot_file,
        )

    best_val_f1 = float(checkpoint_cb.best_model_score) if checkpoint_cb.best_model_score else None
    print(f"\n[{run_name}] Finished! Best Val Macro-F1: {best_val_f1:.4f}")
    if test_results:
        print(f"[{run_name}] Test Results: {test_results[0]}")

    return {
        "best_val_f1": best_val_f1,
        "best_model_path": checkpoint_cb.best_model_path,
        "test_results": test_results[0] if test_results else {},
        "metrics_history": metrics_cb.history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-100 Lightning Model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Max training epochs")
    parser.add_argument("--run_name", type=str, default="baseline", help="MLflow run name")
    parser.add_argument("--experiment_name", type=str, default="cifar100_hpo", help="MLflow experiment name")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for dataset")
    args = parser.parse_args()

    train_model(
        lr=args.lr,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        run_name=args.run_name,
        experiment_name=args.experiment_name,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
