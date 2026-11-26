
import matplotlib.pyplot as plt
from pathlib import Path

from src.load_data import load_raw


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"


def plot_activity_signals(X, y, activity_names):
    """
    Plot the nine sensor channels for one example
    of each human activity.
    """

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    signal_names = [
        "Body Acc X", "Body Acc Y", "Body Acc Z",
        "Body Gyro X", "Body Gyro Y", "Body Gyro Z",
        "Total Acc X", "Total Acc Y", "Total Acc Z"
    ]

    # Create one figure for each activity
    for activity_id in sorted(activity_names):

        # Find the first recording belonging to this activity
        indices = (y == activity_id).nonzero()[0]

        if len(indices) == 0:
            continue

        sample_index = indices[0]
        sample = X[sample_index]

        fig, axes = plt.subplots(3, 3, figsize=(15, 9))

        for channel, ax in enumerate(axes.flat):

            ax.plot(sample[:, channel])
            ax.set_title(signal_names[channel])
            ax.set_xlabel("Time step")
            ax.set_ylabel("Sensor value")
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Human Activity: {activity_names[activity_id]}",
            fontsize=16
        )

        fig.tight_layout()

        filename = (
            f"activity_{activity_id}_"
            f"{activity_names[activity_id].replace(' ', '_')}.png"
        )

        fig.savefig(FIGURE_DIR / filename, dpi=150)
        plt.close(fig)

        print(f"Saved plot: {filename}")


def main():

    print("Loading dataset...")

    X_train, y_train, _, _, _, _, activity_names = load_raw()

    print("Generating sensor signal plots...")

    plot_activity_signals(X_train, y_train, activity_names)

    print("\nAll activity plots generated successfully!")
    print("Saved in:", FIGURE_DIR)


if __name__ == "__main__":
    main()
