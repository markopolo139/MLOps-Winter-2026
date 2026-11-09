# MLOps Homework 1: PyTorch Lightning, Model Monitoring & Hyperparameter Optimization

This repository implements the complete solution for **Homework 1** on the **CIFAR-100** benchmark dataset (100 classes, 32x32 color images).

## Project Overview

- **Dataset**: CIFAR-100 (non-trivial 100-class benchmark, prevents trivial 100% metrics).
- **Architecture**: Custom lightweight Residual Convolutional Neural Network (`CIFAR100ResNet`) with residual connections, batch normalization, and dropout.
- **Framework**: [PyTorch Lightning](https://lightning.ai/) modular design with a custom `CIFAR100Dataset`, `CIFAR100DataModule`, and `CIFAR100LightningModule`.
- **Monitoring Tool**: [MLflow](https://mlflow.org/) tracking loss, accuracy, and macro F1-score across all training and validation epochs.
- **Hyperparameter Optimization**: [Optuna](https://optuna.org/) with median pruning and Tree-structured Parzen Estimator (TPE).

---

## Acceptance Criteria Verification

| Requirement | Implementation & Result | Status |
| :--- | :--- | :---: |
| **Data Module & Lightning Module** | `CIFAR100DataModule` ([src/data_module.py](src/data_module.py)) & `CIFAR100LightningModule` ([src/model.py](src/model.py)) | **PASS** |
| **Custom Dataset & DataLoader** | `CIFAR100Dataset` ([src/dataset.py](src/dataset.py)) extending `torch.utils.data.Dataset` | **PASS** |
| **Model Monitoring** | PyTorch Lightning `MLFlowLogger` tracking SQLite backend `mlflow.db` | **PASS** |
| **Learning Validation** | Training & validation loss steadily drops; accuracy & macro-F1 steadily rise | **PASS** |
| **Appropriate Metric** | Macro-averaged F1-Score (`torchmetrics.classification.MulticlassF1Score`) | **PASS** |
| **HPO Improvement >= +3%** | **+18.90%** absolute gain (**+45.66%** relative gain) over baseline | **PASS** |

### Benchmark Results Summary

| Model Configuration | Optimizer | LR | Weight Decay | Dropout | Batch Size | Val Macro-F1 | Test Macro-F1 | Test Accuracy | Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Configuration** | SGD | 0.0020 | 0.0 | 0.00 | 128 | **41.40%** | 41.87% | 42.81% | 2.15 |
| **Optuna Tuned Model** | **AdamW** | **0.00168** | **4.13e-4** | **0.097** | **64** | **60.31%** | **61.35%** | **61.72%** | **1.36** |
| **Improvement** | - | - | - | - | - | **+18.90%** | **+19.48%** | **+18.91%** | **-0.79** |

*(Required improvement was at least +3.00%; achieved +18.90% absolute / +45.66% relative; Cross Entropy test loss dropped significantly to 1.36).*

---

## Repository Structure

```
MLOps/
├── .gitignore
├── HOMEWORK1.MD
├── README.md
├── requirements.txt
├── comparison_summary.json     # Final comparison metrics
├── hpo_summary.json            # Optuna best trial summary
├── mlflow.db                   # MLflow SQLite database
├── optuna_study.db             # Optuna SQLite study database
├── plots/                      # Generated visualization plots
│   ├── baseline_model_curves.png
│   ├── best_tuned_model_curves.png
│   ├── baseline_vs_best_comparison.png
│   ├── optuna_optimization_history.png
│   └── optuna_param_importances.png
├── src/
│   ├── __init__.py
│   ├── dataset.py              # Custom PyTorch Dataset & DataLoader builder
│   ├── data_module.py          # PyTorch Lightning DataModule
│   ├── model.py                # ResNet architecture & LightningModule
│   └── utils.py                # MLflow setup, curve plotting callback
├── train.py                    # Training script with MLflow logging
├── optimize.py                 # Optuna hyperparameter optimization script
└── evaluate.py                 # End-to-end evaluation & comparison script
```

---

## Installation & Setup

1. **Clone the repository and enter the directory**:
   ```bash
   cd MLOps
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### 1. Run the End-to-End Evaluation Pipeline
Trains baseline, executes Optuna HPO trials, evaluates the winning configuration, and generates comparison plots:
```bash
python evaluate.py --epochs 8 --trials 8
```

### 2. Train a Single Model Configuration
Train baseline model:
```bash
python train.py --epochs 8 --lr 0.002 --optimizer sgd --weight_decay 0.0 --dropout 0.0 --batch_size 128 --run_name "baseline_run"
```

Train tuned model:
```bash
python train.py --epochs 8 --lr 0.00168 --optimizer adamw --weight_decay 0.00041 --dropout 0.097 --batch_size 64 --run_name "tuned_run"
```

### 3. Run Hyperparameter Optimization (Optuna)
```bash
python optimize.py --trials 10 --epochs 8
```

### 4. Launch MLflow UI
View all logged runs, metrics, curves, and artifacts in your browser:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open `http://127.0.0.1:5000` in your browser.

---

## Generated Visualizations

- **Baseline vs Tuned Comparison**: `plots/baseline_vs_best_comparison.png`
- **Baseline Training Curves**: `plots/baseline_model_curves.png`
- **Tuned Model Training Curves**: `plots/best_tuned_model_curves.png`
- **Optuna Optimization History**: `plots/optuna_optimization_history.png`
- **Hyperparameter Importances**: `plots/optuna_param_importances.png`
