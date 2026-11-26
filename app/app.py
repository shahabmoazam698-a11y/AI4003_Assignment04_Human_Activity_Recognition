
import sys
import csv
from pathlib import Path

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Project paths and Python import configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare import prepare_data
from src.model import ActivityRecognitionModel


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Activity and sensor names
# --------------------------------------------------

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

CHANNEL_NAMES = [
    "Body Acceleration X",
    "Body Acceleration Y",
    "Body Acceleration Z",
    "Body Gyroscope X",
    "Body Gyroscope Y",
    "Body Gyroscope Z",
    "Total Acceleration X",
    "Total Acceleration Y",
    "Total Acceleration Z"
]


# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="📱",
    layout="wide"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = ActivityRecognitionModel().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


# --------------------------------------------------
# Load prepared test dataset
# --------------------------------------------------

@st.cache_data
def load_dataset():

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

    return X_test, y_test


# --------------------------------------------------
# Application heading
# --------------------------------------------------

st.title("📱 Human Activity Recognition")

st.subheader(
    "Recognizing Human Activities Using Smartphone Motion Sensors"
)

st.write(
    """
    This application uses a manually implemented **Bidirectional LSTM
    with Temporal Attention** to recognize six human activities from
    smartphone accelerometer and gyroscope measurements.
    """
)

st.divider()


# --------------------------------------------------
# Load model and dataset
# --------------------------------------------------

try:

    model, checkpoint = load_model()

    X_test, y_test = load_dataset()

except Exception as error:

    st.error(f"Could not load model or dataset: {error}")
    st.stop()


# --------------------------------------------------
# Project information
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Test Accuracy",
    "90.16%"
)

col2.metric(
    "Test Samples",
    len(y_test)
)

col3.metric(
    "Best Epoch",
    checkpoint["epoch"]
)

col4.metric(
    "Activity Classes",
    6
)

st.caption(
    f"Device: {DEVICE} | "
    f"Best validation loss: {checkpoint['best_val_loss']:.6f}"
)

st.divider()


# --------------------------------------------------
# Activity prediction
# --------------------------------------------------

st.header("Activity Prediction")

st.write(
    "Select a test sample to see the model's prediction."
)

sample_index = st.slider(
    "Select Test Sample",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0
)


# --------------------------------------------------
# Make prediction
# --------------------------------------------------

sample = X_test[
    sample_index:sample_index + 1
].to(DEVICE)

actual_label = int(y_test[sample_index].item())

with torch.no_grad():

    outputs, attention_weights = model(sample)

    probabilities = torch.softmax(outputs, dim=1)

    predicted_label = int(
        probabilities.argmax(dim=1).item()
    )

    confidence = float(
        probabilities[0, predicted_label].item()
    )

attention_values = (
    attention_weights.squeeze().cpu().numpy()
)

actual_activity = ACTIVITY_NAMES[actual_label]
predicted_activity = ACTIVITY_NAMES[predicted_label]


# --------------------------------------------------
# Display prediction results
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("Actual Activity")
    st.info(actual_activity)

with col2:

    st.subheader("Predicted Activity")

    if actual_label == predicted_label:
        st.success(predicted_activity)
    else:
        st.error(predicted_activity)


st.metric(
    "Prediction Confidence",
    f"{confidence * 100:.2f}%"
)

if actual_label == predicted_label:

    st.success(
        "The model correctly classified this sample."
    )

else:

    st.warning(
        "The model incorrectly classified this sample."
    )


# --------------------------------------------------
# Class probabilities
# --------------------------------------------------

st.subheader("Class Prediction Probabilities")

probability_values = probabilities[0].cpu().numpy()

for i, activity in enumerate(ACTIVITY_NAMES):

    st.write(
        f"{activity}: {probability_values[i] * 100:.2f}%"
    )

    st.progress(
        float(probability_values[i])
    )


st.divider()


# --------------------------------------------------
# Sensor signal visualization
# --------------------------------------------------

st.header("Sensor Signal Visualization")

channel = st.selectbox(
    "Select Sensor Channel",
    CHANNEL_NAMES
)

channel_index = CHANNEL_NAMES.index(channel)

sample_values = X_test[sample_index].numpy()

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    np.arange(128),
    sample_values[:, channel_index]
)

ax.set_title(f"Sensor Signal: {channel}")
ax.set_xlabel("Time Step")
ax.set_ylabel("Normalized Sensor Value")
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)


st.divider()


# --------------------------------------------------
# Temporal attention visualization
# --------------------------------------------------

st.header("Temporal Attention Visualization")

st.write(
    """
    The attention mechanism assigns a weight to each of the
    128 time steps. Higher weights indicate time steps that
    contribute more strongly to the attention context.
    """
)

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    np.arange(128),
    attention_values,
    linewidth=1.5
)

ax.fill_between(
    np.arange(128),
    attention_values,
    alpha=0.2
)

ax.set_title("Temporal Attention Weights")
ax.set_xlabel("Time Step")
ax.set_ylabel("Attention Weight")
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

st.caption(
    f"Sum of attention weights: {attention_values.sum():.4f}"
)


st.divider()


# --------------------------------------------------
# Model evaluation results
# --------------------------------------------------

st.header("Model Evaluation Results")

summary_path = TABLE_DIR / "test_summary.txt"

if summary_path.exists():

    with open(summary_path, "r") as file:
        st.text(file.read())


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

st.subheader("Confusion Matrix")

confusion_path = TABLE_DIR / "confusion_matrix.csv"

if confusion_path.exists():

    confusion = np.loadtxt(
        confusion_path,
        delimiter=",",
        dtype=int
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(confusion)

    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(6))
    ax.set_yticks(range(6))

    ax.set_xticklabels(
        ACTIVITY_NAMES,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(ACTIVITY_NAMES)

    ax.set_xlabel("Predicted Activity")
    ax.set_ylabel("Actual Activity")
    ax.set_title("Test Confusion Matrix")

    for i in range(6):
        for j in range(6):

            ax.text(
                j,
                i,
                str(confusion[i, j]),
                ha="center",
                va="center"
            )

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

else:

    st.warning("Confusion matrix file not found.")


# --------------------------------------------------
# Class-wise metrics
# --------------------------------------------------

st.subheader("Class-wise Performance")

metrics_path = TABLE_DIR / "class_metrics.csv"

if metrics_path.exists():

    with open(metrics_path, "r", newline="") as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    st.table(rows)

else:

    st.warning("Class metrics file not found.")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AI4003 Applied Machine Learning | "
    "Human Activity Recognition Project"
)
