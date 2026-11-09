from typing import Any, Dict
import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class ResidualBlock(nn.Module):
    """
    Standard pre-activation residual block for 2D convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout_rate: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class CIFAR100ResNet(nn.Module):
    """
    Lightweight ResNet tailored for 32x32 CIFAR-100 images.
    """

    def __init__(self, num_classes: int = 100, base_channels: int = 64, dropout_rate: float = 0.1):
        super().__init__()
        c = base_channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ResidualBlock(c, c, stride=1, dropout_rate=dropout_rate)
        self.layer2 = ResidualBlock(c, c * 2, stride=2, dropout_rate=dropout_rate)       # 16x16
        self.layer3 = ResidualBlock(c * 2, c * 4, stride=2, dropout_rate=dropout_rate)   # 8x8
        self.layer4 = ResidualBlock(c * 4, c * 4, stride=2, dropout_rate=dropout_rate)   # 4x4

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(c * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.initial_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.fc_dropout(out)
        out = self.fc(out)
        return out


class CIFAR100LightningModule(L.LightningModule):
    """
    LightningModule for CIFAR-100 classification.
    Tracks Multiclass Accuracy and Macro F1-score for training and validation.
    """

    def __init__(
        self,
        num_classes: int = 100,
        base_channels: int = 64,
        lr: float = 1e-3,
        optimizer_name: str = "adamw",
        weight_decay: float = 1e-4,
        dropout_rate: float = 0.1,
        use_scheduler: bool = True,
        max_epochs: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = CIFAR100ResNet(
            num_classes=num_classes,
            base_channels=base_channels,
            dropout_rate=dropout_rate,
        )
        self.criterion = nn.CrossEntropyLoss()

        # Metrics for train, val, and test
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1_macro", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1_macro", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/f1_macro", self.test_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        opt_name = self.hparams.optimizer_name.lower()
        lr = self.hparams.lr
        weight_decay = self.hparams.weight_decay

        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        if self.hparams.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=1e-6,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        return {"optimizer": optimizer}
