
from pathlib import Path
import numpy as np


# Find the original UCI HAR Dataset folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "UCI HAR Dataset"


def load_signals(split):
    """
    Load the nine raw sensor signals.

    Returns:
        X: Array with shape (samples, 128, 9)
    """

    signal_names = [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
        "total_acc_x",
        "total_acc_y",
        "total_acc_z"
    ]

    signal_folder = DATASET_PATH / split / "Inertial Signals"

    signals = []

    for signal in signal_names:
        file_path = signal_folder / f"{signal}_{split}.txt"

        data = np.loadtxt(file_path)
        signals.append(data)

    # Combine nine signals along the last dimension
    X = np.stack(signals, axis=-1)

    return X.astype(np.float32)


def load_raw():
    """
    Load training and testing data, labels and subject IDs.
    """

    X_train = load_signals("train")
    X_test = load_signals("test")

    y_train = np.loadtxt(
        DATASET_PATH / "train" / "y_train.txt",
        dtype=int
    )

    y_test = np.loadtxt(
        DATASET_PATH / "test" / "y_test.txt",
        dtype=int
    )

    subject_train = np.loadtxt(
        DATASET_PATH / "train" / "subject_train.txt",
        dtype=int
    )

    subject_test = np.loadtxt(
        DATASET_PATH / "test" / "subject_test.txt",
        dtype=int
    )

    activity_names = {}

    with open(DATASET_PATH / "activity_labels.txt", "r") as file:
        for line in file:
            label, name = line.strip().split(maxsplit=1)
            activity_names[int(label)] = name

    return (
        X_train,
        y_train,
        subject_train,
        X_test,
        y_test,
        subject_test,
        activity_names
    )


if __name__ == "__main__":

    print("Loading UCI HAR dataset...")

    (
        X_train,
        y_train,
        subject_train,
        X_test,
        y_test,
        subject_test,
        activity_names
    ) = load_raw()

    print("\nDataset loaded successfully!")

    print("\nTraining data shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Training subjects shape:", subject_train.shape)

    print("\nTesting data shape:", X_test.shape)
    print("Testing labels shape:", y_test.shape)
    print("Testing subjects shape:", subject_test.shape)

    print("\nActivity names:")

    for label, name in activity_names.items():
        print(label, ":", name)
