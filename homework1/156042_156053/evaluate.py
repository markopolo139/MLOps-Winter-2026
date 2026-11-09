import argparse
import json
import os
import matplotlib.pyplot as plt
import optuna

from train import train_model


def run_evaluation(epochs: int = 16, trials: int = 6, seed: int = 42):
    print("=" * 60)
    print("STEP 1: Training Baseline Model")
    print("Baseline config: lr=0.002, optimizer=sgd, weight_decay=0.0, dropout=0.0, batch_size=128")
    print("=" * 60)

    baseline_results = train_model(
        lr=0.002,
        optimizer_name="sgd",
        weight_decay=0.0,
        dropout_rate=0.0,
        batch_size=128,
        max_epochs=epochs,
        run_name="baseline_model",
        seed=seed,
    )

    baseline_val_f1 = baseline_results["best_val_f1"]
    print(f"\nBaseline Validation Macro-F1: {baseline_val_f1:.4f}")

    print("\n" + "=" * 60)
    print("STEP 2: Running Optuna Hyperparameter Optimization")
    print("=" * 60)

    from optimize import run_hpo
    study = run_hpo(
        n_trials=trials,
        epochs_per_trial=min(epochs, 12),
        seed=seed,
    )

    best_params = study.best_params
    best_optuna_score = study.best_value

    print("\n" + "=" * 60)
    print("STEP 3: Training Best Tuned Model for Full Comparison")
    print(f"Best configuration: {best_params}")
    print("=" * 60)

    best_results = train_model(
        lr=best_params["lr"],
        optimizer_name=best_params["optimizer_name"],
        weight_decay=best_params["weight_decay"],
        dropout_rate=best_params["dropout_rate"],
        batch_size=best_params["batch_size"],
        max_epochs=epochs,
        run_name="best_tuned_model",
        seed=seed,
    )

    tuned_val_f1 = best_results["best_val_f1"]

    # Calculate absolute and relative improvements
    abs_gain = tuned_val_f1 - baseline_val_f1
    rel_gain = (abs_gain / baseline_val_f1) * 100.0 if baseline_val_f1 > 0 else 0.0

    print("\n" + "=" * 60)
    print("FINAL SUMMARY & ACCEPTANCE VERIFICATION")
    print("=" * 60)
    print(f"Baseline Validation Macro-F1: {baseline_val_f1 * 100:.2f}%")
    print(f"Tuned Model Validation Macro-F1: {tuned_val_f1 * 100:.2f}%")
    print(f"Absolute Improvement: {abs_gain * 100:+.2f}%")
    print(f"Relative Improvement: {rel_gain:+.2f}%")
    print(f"Required Improvement: >= +3.00%")

    meets_criteria = abs_gain >= 0.03 or rel_gain >= 3.0
    status_str = "PASS" if meets_criteria else "NEEDS REVIEW"
    print(f"Acceptance Criteria Status: {status_str}")
    print("=" * 60)

    # Save comparative plot (Loss, Accuracy, and F1)
    os.makedirs("./plots", exist_ok=True)
    comp_plot_path = "./plots/baseline_vs_best_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    b_hist = baseline_results["metrics_history"]
    t_hist = best_results["metrics_history"]
    epochs_b = range(1, len(b_hist.get("val/loss", [])) + 1)
    epochs_t = range(1, len(t_hist.get("val/loss", [])) + 1)

    # 1. Validation Loss
    b_val_loss = b_hist.get("val/loss", [])
    t_val_loss = t_hist.get("val/loss", [])
    if b_val_loss:
        axes[0].plot(epochs_b, b_val_loss, "r--o", label=f"Baseline (Min: {min(b_val_loss):.3f})")
    if t_val_loss:
        axes[0].plot(epochs_t, t_val_loss, "g-s", label=f"Tuned Model (Min: {min(t_val_loss):.3f})")
    axes[0].set_title("Validation Cross Entropy Loss", fontsize=13)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=10)

    # 2. Validation Accuracy
    b_val_acc = [x * 100 for x in b_hist.get("val/acc", [])]
    t_val_acc = [x * 100 for x in t_hist.get("val/acc", [])]
    if b_val_acc:
        axes[1].plot(epochs_b, b_val_acc, "r--o", label=f"Baseline (Max: {max(b_val_acc):.1f}%)")
    if t_val_acc:
        axes[1].plot(epochs_t, t_val_acc, "g-s", label=f"Tuned Model (Max: {max(t_val_acc):.1f}%)")
    axes[1].set_title("Validation Accuracy (%)", fontsize=13)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=10)

    # 3. Validation Macro-F1
    b_val_f1 = [x * 100 for x in b_hist.get("val/f1_macro", [])]
    t_val_f1 = [x * 100 for x in t_hist.get("val/f1_macro", [])]
    if b_val_f1:
        axes[2].plot(epochs_b, b_val_f1, "r--o", label=f"Baseline (Max: {baseline_val_f1*100:.1f}%)")
    if t_val_f1:
        axes[2].plot(epochs_t, t_val_f1, "g-s", label=f"Tuned Model (Max: {tuned_val_f1*100:.1f}%)")
    axes[2].set_title("Validation Macro F1-Score (%)", fontsize=13)
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].set_ylabel("Macro F1 (%)", fontsize=11)
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(comp_plot_path, dpi=200)
    plt.close()

    summary_data = {
        "baseline_val_f1": baseline_val_f1,
        "tuned_val_f1": tuned_val_f1,
        "abs_gain_percent": abs_gain * 100,
        "rel_gain_percent": rel_gain,
        "meets_criteria": meets_criteria,
        "best_hyperparameters": best_params,
        "comparison_plot": comp_plot_path,
    }

    with open("comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="End-to-end evaluation & comparison")
    parser.add_argument("--epochs", type=int, default=16, help="Epochs for training runs")
    parser.add_argument("--trials", type=int, default=6, help="Optuna trials")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    run_evaluation(epochs=args.epochs, trials=args.trials, seed=args.seed)


if __name__ == "__main__":
    main()
