
from pathlib import Path

import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.prepare import prepare_data
from src.model import ActivityRecognitionModel
from src.metrics import (
    calculate_confusion_matrix,
    calculate_classification_metrics,
    calculate_overall_accuracy
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def test():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # --------------------------------------------------
    # Load prepared test dataset
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

    X_test = X_test.to(device)

    y_test_numpy = y_test.numpy()

    print("\nTest samples:", len(y_test))
    print("Test tensor shape:", list(X_test.shape))

    # --------------------------------------------------
    # Load best saved model
    # --------------------------------------------------

    print("\nLoading best model...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model = ActivityRecognitionModel().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Best checkpoint epoch:", checkpoint["epoch"])
    print("Best validation loss:", checkpoint["best_val_loss"])

    # --------------------------------------------------
    # Make predictions
    # --------------------------------------------------

    print("\nEvaluating test dataset...")

    predictions = []

    with torch.no_grad():

        outputs, attention_weights = model(X_test)

        predicted_labels = outputs.argmax(dim=1)

        predictions = predicted_labels.cpu().numpy()

    # --------------------------------------------------
    # Overall accuracy
    # --------------------------------------------------

    accuracy = calculate_overall_accuracy(
        y_test_numpy,
        predictions
    )

    print("\nOverall Test Accuracy:", round(accuracy * 100, 2), "%")

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    confusion = calculate_confusion_matrix(
        y_test_numpy,
        predictions,
        num_classes=6
    )

    print("\nConfusion Matrix:")
    print(confusion)

    np.savetxt(
        TABLE_DIR / "confusion_matrix.csv",
        confusion,
        delimiter=",",
        fmt="%d"
    )

    # --------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------

    class_metrics = calculate_classification_metrics(
        confusion
    )

    print("\nClass-wise Performance:")

    metrics_path = TABLE_DIR / "class_metrics.csv"

    with open(metrics_path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Activity",
            "Precision",
            "Recall",
            "F1 Score",
            "Accuracy"
        ])

        for i, metrics in enumerate(class_metrics):

            activity = activity_names[i + 1]

            print(
                f"{activity}: "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1_score']:.4f}, "
                f"Accuracy={metrics['accuracy']:.4f}"
            )

            writer.writerow([
                activity,
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["accuracy"]
            ])

    # --------------------------------------------------
    # Plot confusion matrix
    # --------------------------------------------------

    labels = [
        activity_names[i + 1]
        for i in range(6)
    ]

    plt.figure(figsize=(10, 8))

    plt.imshow(confusion, interpolation="nearest")

    plt.title("Confusion Matrix - Test Dataset")
    plt.colorbar()

    positions = np.arange(6)

    plt.xticks(
        positions,
        labels,
        rotation=45,
        ha="right"
    )

    plt.yticks(positions, labels)

    plt.xlabel("Predicted Activity")
    plt.ylabel("Actual Activity")

    for i in range(6):
        for j in range(6):

            plt.text(
                j,
                i,
                str(confusion[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # Save overall results
    # --------------------------------------------------

    with open(
        TABLE_DIR / "test_summary.txt",
        "w"
    ) as file:

        file.write(f"Best checkpoint epoch: {checkpoint['epoch']}\n")
        file.write(
            f"Best validation loss: {checkpoint['best_val_loss']:.6f}\n"
        )
        file.write(f"Test samples: {len(y_test)}\n")
        file.write(f"Test accuracy: {accuracy * 100:.2f}%\n")

    print("\nEvaluation completed!")

    print("\nSaved files:")
    print(TABLE_DIR / "confusion_matrix.csv")
    print(TABLE_DIR / "class_metrics.csv")
    print(TABLE_DIR / "test_summary.txt")
    print(FIGURE_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    test()
