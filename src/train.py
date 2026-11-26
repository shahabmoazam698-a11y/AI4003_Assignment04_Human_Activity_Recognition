
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.prepare import prepare_data
from src.model import ActivityRecognitionModel


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "best_model.pth"
HISTORY_PATH = TABLE_DIR / "training_history.csv"


# --------------------------------------------------
# Training configuration
# --------------------------------------------------

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
GRADIENT_CLIP = 1.0


def train():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # --------------------------------------------------
    # Load prepared dataset
    # --------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        mean,
        std,
        activity_names
    ) = prepare_data()

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------
    # Create model and optimizer
    # --------------------------------------------------

    model = ActivityRecognitionModel().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTrainable parameters:", total_parameters)
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # --------------------------------------------------
    # Resume from existing checkpoint
    # --------------------------------------------------

    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0

    if CHECKPOINT_PATH.exists():

        print("\nLoading saved checkpoint...")

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        best_epoch = start_epoch

        print("Checkpoint loaded successfully!")
        print("Resuming after epoch:", start_epoch)
        print("Best validation loss:", best_val_loss)

    else:

        print("\nNo checkpoint found. Starting from epoch 1.")

    if start_epoch >= EPOCHS:
        print("\nTraining has already reached the requested epoch count.")
        return

    # --------------------------------------------------
    # Preserve consistent training history
    # --------------------------------------------------

    old_history = []

    if HISTORY_PATH.exists():

        with open(HISTORY_PATH, "r", newline="") as file:

            reader = csv.DictReader(file)

            for row in reader:

                if int(row["epoch"]) <= start_epoch:
                    old_history.append(row)

    with open(HISTORY_PATH, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy"
        ])

        for row in old_history:

            writer.writerow([
                row["epoch"],
                row["train_loss"],
                row["train_accuracy"],
                row["val_loss"],
                row["val_accuracy"]
            ])

        file.flush()

        # --------------------------------------------------
        # Training loop
        # --------------------------------------------------

        for epoch in range(start_epoch, EPOCHS):

            start_time = time.time()

            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs, _ = model(inputs)

                loss = criterion(outputs, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRADIENT_CLIP
                )

                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_accuracy = correct / total

            # --------------------------------------------------
            # Validation
            # --------------------------------------------------

            model.eval()

            val_running_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():

                for inputs, labels in val_loader:

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs, _ = model(inputs)

                    loss = criterion(outputs, labels)

                    val_running_loss += (
                        loss.item() * inputs.size(0)
                    )

                    predictions = outputs.argmax(dim=1)

                    val_correct += (
                        predictions == labels
                    ).sum().item()

                    val_total += labels.size(0)

            val_loss = val_running_loss / val_total
            val_accuracy = val_correct / val_total

            # --------------------------------------------------
            # Save best model
            # --------------------------------------------------

            if val_loss < best_val_loss:

                best_val_loss = val_loss
                best_epoch = epoch + 1

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "mean": torch.tensor(mean, dtype=torch.float32),
                        "std": torch.tensor(std, dtype=torch.float32),
                        "activity_names": activity_names
                    },
                    CHECKPOINT_PATH
                )

                print("New best model saved!")

            # --------------------------------------------------
            # Record training history
            # --------------------------------------------------

            writer.writerow([
                epoch + 1,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy
            ])

            file.flush()

            elapsed = time.time() - start_time

            print(
                f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

    # --------------------------------------------------
    # Training completed
    # --------------------------------------------------

    print("\nTraining completed!")

    print("Best epoch:", best_epoch)
    print("Best validation loss:", round(best_val_loss, 6))

    print("\nBest model saved at:")
    print(CHECKPOINT_PATH)

    print("\nTraining history saved at:")
    print(HISTORY_PATH)


if __name__ == "__main__":
    train()
