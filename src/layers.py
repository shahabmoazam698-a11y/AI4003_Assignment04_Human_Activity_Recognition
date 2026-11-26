
import torch
import torch.nn as nn


# ============================================================
# 1. MANUAL RNN CELL
# ============================================================

class ManualRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):

        h_next = torch.tanh(
            self.input_layer(x) + self.hidden_layer(h)
        )

        return h_next


# ============================================================
# 2. MANUAL LSTM CELL
# ============================================================

class ManualLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        # One linear layer calculates all four LSTM gates
        self.gates = nn.Linear(
            input_size + hidden_size,
            4 * hidden_size
        )

    def forward(self, x, states):

        h, c = states

        combined = torch.cat((x, h), dim=1)

        gates = self.gates(combined)

        # Divide the result into four gates
        i, f, g, o = gates.chunk(4, dim=1)

        # Input gate
        i = torch.sigmoid(i)

        # Forget gate
        f = torch.sigmoid(f)

        # Candidate cell state
        g = torch.tanh(g)

        # Output gate
        o = torch.sigmoid(o)

        # Update cell state
        c_next = f * c + i * g

        # Update hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# ============================================================
# 3. MANUAL RNN FOR COMPLETE SEQUENCES
# ============================================================

class ManualRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.cell = ManualRNNCell(input_size, hidden_size)

    def forward(self, x):

        batch_size, sequence_length, _ = x.shape

        h = x.new_zeros(batch_size, self.hidden_size)

        outputs = []

        for t in range(sequence_length):

            h = self.cell(x[:, t, :], h)

            outputs.append(h)

        return torch.stack(outputs, dim=1)


# ============================================================
# 4. MANUAL LSTM FOR COMPLETE SEQUENCES
# ============================================================

class ManualLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.cell = ManualLSTMCell(input_size, hidden_size)

    def forward(self, x):

        batch_size, sequence_length, _ = x.shape

        h = x.new_zeros(batch_size, self.hidden_size)
        c = x.new_zeros(batch_size, self.hidden_size)

        outputs = []

        for t in range(sequence_length):

            h, c = self.cell(
                x[:, t, :],
                (h, c)
            )

            outputs.append(h)

        return torch.stack(outputs, dim=1)


# ============================================================
# 5. MANUAL BIDIRECTIONAL LSTM
# ============================================================

class ManualBiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.forward_lstm = ManualLSTM(
            input_size,
            hidden_size
        )

        self.backward_lstm = ManualLSTM(
            input_size,
            hidden_size
        )

    def forward(self, x):

        # Process sequence from beginning to end
        forward_output = self.forward_lstm(x)

        # Reverse the sequence along the time dimension
        reversed_x = torch.flip(x, dims=[1])

        # Process sequence from end to beginning
        backward_output = self.backward_lstm(reversed_x)

        # Restore the original time order
        backward_output = torch.flip(
            backward_output,
            dims=[1]
        )

        # Combine forward and backward representations
        output = torch.cat(
            (forward_output, backward_output),
            dim=2
        )

        return output


# ============================================================
# 6. TEMPORAL ATTENTION
# ============================================================

class TemporalAttention(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.score_layer = nn.Linear(input_size, 1)

    def forward(self, x):

        # Calculate an importance score for every time step
        scores = self.score_layer(x)

        # Convert scores into attention weights
        weights = torch.softmax(scores, dim=1)

        # Calculate weighted combination of all time steps
        context = torch.sum(
            weights * x,
            dim=1
        )

        return context, weights
