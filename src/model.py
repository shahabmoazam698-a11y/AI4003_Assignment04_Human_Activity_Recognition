
import torch
import torch.nn as nn

from src.layers import ManualBiLSTM, TemporalAttention


class ActivityRecognitionModel(nn.Module):

    def __init__(
        self,
        input_size=9,
        hidden_size=64,
        num_classes=6
    ):

        super().__init__()

        # Bidirectional LSTM
        self.bilstm = ManualBiLSTM(
            input_size=input_size,
            hidden_size=hidden_size
        )

        # BiLSTM output size is 64 * 2 = 128
        attention_input_size = hidden_size * 2

        # Temporal attention
        self.attention = TemporalAttention(
            input_size=attention_input_size
        )

        # Final classification layer
        self.classifier = nn.Linear(
            attention_input_size,
            num_classes
        )

    def forward(self, x):

        # Extract temporal features
        features = self.bilstm(x)

        # Apply temporal attention
        context, attention_weights = self.attention(features)

        # Predict six activity classes
        logits = self.classifier(context)

        return logits, attention_weights


if __name__ == "__main__":

    print("Testing Activity Recognition Model...")

    model = ActivityRecognitionModel()

    # Four example recordings
    sample_input = torch.randn(4, 128, 9)

    # Forward pass
    predictions, attention_weights = model(sample_input)

    print("\nModel test successful!")

    print("Input shape:", sample_input.shape)
    print("Prediction shape:", predictions.shape)
    print("Attention shape:", attention_weights.shape)

    print(
        "Attention weight sums:",
        attention_weights.sum(dim=1).flatten()
    )

    # Count trainable parameters
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nTrainable parameters:", total_parameters)
