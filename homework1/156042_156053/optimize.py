import argparse
import json
import os
import torch
import optuna
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.data_module import CIFAR100DataModule
from src.model import CIFAR100LightningModule
from src.utils import setup_mlflow_logger


class OptunaPruningCallback(L.Callback):
    """
    Callback that reports epoch validation metric to Optuna and raises TrialPruned if needed.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "val/f1_macro"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return

        epoch = trainer.current_epoch
        score_val = float(current_score.detach().cpu().item())
        self.trial.report(score_val, step=epoch)

        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial {self.trial.number} pruned at epoch {epoch} with score {score_val:.4f}")


def objective(
    trial: optuna.Trial,
    epochs_per_trial: int = 8,
    data_dir: str = "./data",
    experiment_name: str = "cifar100_hpo",
    seed: int = 42,
) -> float:
    """
    Optuna objective function for tuning CIFAR-100 hyperparameters.
    """
    # Hyperparameter search space
    lr = trial.suggest_float("lr", 3e-4, 3e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer_name", ["adamw", "sgd"])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.35)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])

    L.seed_everything(seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    data_module = CIFAR100DataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=0.2,
        seed=seed,
    )

    model = CIFAR100LightningModule(
        num_classes=100,
        base_channels=64,
        lr=lr,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate,
        use_scheduler=True,
        max_epochs=epochs_per_trial,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val/f1_macro",
        mode="max",
        save_top_k=1,
    )
    pruning_cb = OptunaPruningCallback(trial, monitor="val/f1_macro")

    mlflow_logger = setup_mlflow_logger(
        experiment_name=experiment_name,
        tracking_uri="sqlite:///mlflow.db",
        run_name=f"trial_{trial.number}",
        tags={"optuna_trial": str(trial.number), "type": "hpo_trial"},
    )

    trainer = L.Trainer(
        max_epochs=epochs_per_trial,
        accelerator="auto",
        devices=1,
        logger=mlflow_logger,
        callbacks=[checkpoint_cb, pruning_cb],
        log_every_n_steps=25,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=data_module)

    best_score = float(checkpoint_cb.best_model_score) if checkpoint_cb.best_model_score else 0.0
    return best_score


def run_hpo(
    n_trials: int = 10,
    epochs_per_trial: int = 8,
    data_dir: str = "./data",
    experiment_name: str = "cifar100_hpo",
    study_name: str = "cifar100_optimization",
    storage: str = "sqlite:///optuna_study.db",
    seed: int = 42,
) -> optuna.Study:
    """
    Executes the Optuna study with a median pruner.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Seed with strong candidate trial
    study.enqueue_trial({
        "lr": 0.00168,
        "optimizer_name": "adamw",
        "weight_decay": 0.00041,
        "dropout_rate": 0.10,
        "batch_size": 64,
    })

    print(f"\n==========================================")
    print(f"Starting Optuna HPO ({n_trials} trials, {epochs_per_trial} epochs/trial)")
    print(f"Tracking MLflow: sqlite:///mlflow.db")
    print(f"Storage: {storage}")
    print(f"==========================================\n")

    study.optimize(
        lambda trial: objective(
            trial,
            epochs_per_trial=epochs_per_trial,
            data_dir=data_dir,
            experiment_name=experiment_name,
            seed=seed,
        ),
        n_trials=n_trials,
    )

    print("\n==========================================")
    print("Optimization Completed!")
    print(f"Best Trial #{study.best_trial.number}")
    print(f"Best Validation Macro-F1: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")
    print("==========================================\n")

    # Save summary to JSON
    results = {
        "best_trial_number": study.best_trial.number,
        "best_val_f1_macro": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    with open("hpo_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate and save Optuna analysis figures
    os.makedirs("./plots", exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        import optuna.visualization.matplotlib as optuna_plt

        # Optimization history
        ax1 = optuna_plt.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig("./plots/optuna_optimization_history.png", dpi=200)
        plt.close()

        # Parameter importances
        ax2 = optuna_plt.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig("./plots/optuna_param_importances.png", dpi=200)
        plt.close()
        print("Optuna visualization plots saved in ./plots/")
    except Exception as e:
        print(f"Notice: Could not generate some Optuna plots ({e})")

    return study


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for CIFAR-100")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per trial")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--experiment_name", type=str, default="cifar100_hpo", help="MLflow experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_hpo(
        n_trials=args.trials,
        epochs_per_trial=args.epochs,
        data_dir=args.data_dir,
        experiment_name=args.experiment_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
