
import numpy as np
import torch

from src.load_data import load_raw


VALIDATION_SUBJECTS = [3, 15, 25, 30]


def prepare_data():

    print("Loading raw dataset...")

    (
        X_train,
        y_train,
        subject_train,
        X_test,
        y_test,
        subject_test,
        activity_names
    ) = load_raw()

    # --------------------------------------------------
    # Step 1: Convert labels from 1-6 to 0-5
    # --------------------------------------------------

    y_train = y_train - 1
    y_test = y_test - 1

    # --------------------------------------------------
    # Step 2: Split training data into training
    # and validation sets using subject IDs
    # --------------------------------------------------

    validation_mask = np.isin(
        subject_train,
        VALIDATION_SUBJECTS
    )

    train_mask = ~validation_mask

    X_train_new = X_train[train_mask]
    y_train_new = y_train[train_mask]

    X_val = X_train[validation_mask]
    y_val = y_train[validation_mask]

    # --------------------------------------------------
    # Step 3: Calculate mean and standard deviation
    # using training data only
    # --------------------------------------------------

    mean = X_train_new.mean(axis=(0, 1), keepdims=True)

    std = X_train_new.std(axis=(0, 1), keepdims=True)

    std[std == 0] = 1.0

    # --------------------------------------------------
    # Step 4: Standardize all three datasets
    # using training statistics
    # --------------------------------------------------

    X_train_new = (X_train_new - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # --------------------------------------------------
    # Step 5: Convert NumPy arrays into PyTorch tensors
    # --------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train_new,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train_new,
        dtype=torch.long
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.long
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.long
    )

    # --------------------------------------------------
    # Step 6: Display dataset information
    # --------------------------------------------------

    print("\nPreprocessing completed!")

    print("\nTraining samples:", len(X_train_tensor))
    print("Validation samples:", len(X_val_tensor))
    print("Testing samples:", len(X_test_tensor))

    print("\nTraining tensor shape:", X_train_tensor.shape)
    print("Validation tensor shape:", X_val_tensor.shape)
    print("Testing tensor shape:", X_test_tensor.shape)

    print("\nNumber of activity classes:", len(activity_names))

    print("\nTraining label range:",
          y_train_tensor.min().item(),
          "to",
          y_train_tensor.max().item())

    # Return all prepared data and normalization statistics
    return (
        X_train_tensor,
        y_train_tensor,
        X_val_tensor,
        y_val_tensor,
        X_test_tensor,
        y_test_tensor,
        mean,
        std,
        activity_names
    )


if __name__ == "__main__":

    prepare_data()
