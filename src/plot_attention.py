
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.prepare import prepare_data
from src.model import ActivityRecognitionModel


# --------------------------------------------------
# Project paths
# ---------------------------------there is not plot attention file


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def plot_attention():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # --------------------------------------------------
    # Load test dataset
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

    # --------------------------------------------------
    # Load best model
    # --------------------------------------------------

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

    print("Loaded best model from epoch:", checkpoint["epoch"])

    # --------------------------------------------------
    # Select one correctly classified sample
    # for each activity
    # --------------------------------------------------

    selected_samples = {}

    with torch.no_grad():

        for index in range(len(X_test)):

            inputs = X_test[index:index + 1].to(device)

            outputs, attention_weights = model(inputs)

            prediction = outputs.argmax(dim=1).item()

            actual = y_test[index].item()

            if prediction == actual and actual not in selected_samples:

                selected_samples[actual] = (
                    X_test[index].numpy(),
                    attention_weights.squeeze().cpu().numpy()
                )

            if len(selected_samples) == 6:
                break

    # --------------------------------------------------
    # Plot attention for all six activities
    # --------------------------------------------------

    for class_id in range(6):

        if class_id not in selected_samples:

            print(
                "No correctly classified example found for:",
                activity_names[class_id + 1]
            )

            continue

        sample, weights = selected_samples[class_id]

        activity = activity_names[class_id + 1]

        plt.figure(figsize=(12, 4))

        plt.plot(
            np.arange(128),
            weights,
            linewidth=1.5
        )

        plt.title(
            f"Temporal Attention Weights - {activity}"
        )

        plt.xlabel("Time Step")
        plt.ylabel("Attention Weight")

        plt.grid(alpha=0.3)

        plt.tight_layout()

        filename = (
            f"attention_{activity.lower().replace(' ', '_')}.png"
        )

        plt.savefig(
            FIGURE_DIR / filename,
            dpi=300
        )

        plt.close()

        print("Saved:", filename)

    print("\nAttention visualization completed!")


if __name__ == "__main__":
    plot_attention()
