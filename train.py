"""Train the AgriVision crop-disease classifier from a folder-per-class dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


DEFAULT_DATA_DIR = Path(r"C:\Users\nem32\Downloads\agri-vision\data\raw")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "model"
IMAGE_SIZE = 128
SEED = 42


class TransformedSubset(Dataset):
    """Apply a split-specific transform while sharing ImageFolder's class mapping."""

    def __init__(self, image_folder: datasets.ImageFolder, indices: list[int], transform):
        self.image_folder = image_folder
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image_path, label = self.image_folder.samples[self.indices[index]]
        image = self.image_folder.loader(image_path)
        return self.transform(image), label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_datasets(data_dir: Path, validation_fraction: float, seed: int):
    base_dataset = datasets.ImageFolder(data_dir)
    if len(base_dataset.classes) < 2:
        raise ValueError("Expected at least two class folders under data/raw.")

    labels = [label for _, label in base_dataset.samples]
    indices = list(range(len(base_dataset)))
    counts = np.bincount(labels)
    if counts.min() < 2:
        too_small = [base_dataset.classes[i] for i, count in enumerate(counts) if count < 2]
        raise ValueError(f"Each class needs at least 2 images for a split. Too small: {too_small}")

    train_indices, val_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=labels,
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return (
        TransformedSubset(base_dataset, train_indices, train_transform),
        TransformedSubset(base_dataset, val_indices, evaluation_transform),
        base_dataset.classes,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {args.data_dir}")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, classes = make_datasets(
        args.data_dir, args.validation_fraction, args.seed
    )
    print(f"Device: {device}")
    print(f"Classes: {len(classes)} | train: {len(train_dataset)} | validation: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda"
    )

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model = model.to(device)

    # First three epochs train only the new prediction layer; later epochs fine-tune all layers.
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    patience = 3
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        if epoch == 4:
            for parameter in model.features.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

        model.train()
        if epoch <= 3:
            # Keep BatchNorm running statistics frozen with the pretrained feature extractor.
            model.features.eval()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        val_loss, val_accuracy = evaluate(model, val_loader, device)
        train_loss = running_loss / seen
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy})
        print(f"Epoch {epoch}/{args.epochs} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val accuracy: {val_accuracy:.2%}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": classes,
                    "image_size": IMAGE_SIZE,
                },
                args.output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping: validation accuracy has not improved for 3 epochs.")
                break

    metrics = {
        "best_validation_accuracy": best_accuracy,
        "class_names": classes,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved best model: {args.output_dir / 'best.pt'}")
    print(f"Best validation accuracy: {best_accuracy:.2%}")


if __name__ == "__main__":
    main()
