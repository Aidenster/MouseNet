import torch
import torch.nn as nn

# Set the default tensor type to double precision.
torch.set_default_tensor_type(torch.DoubleTensor)

class MouseNet(nn.Module):
    def __init__(self, time_steps, device="cpu"):
        super(MouseNet, self).__init__()

        self.device = device
        self.time_steps = time_steps

        # Define the number of LSTM layers and the size of the hidden state.
        self.num_layers = 1
        self.hidden_size = 128

        # Fully connected layer to map the input (start and target coordinates) to the LSTM input size.
        self.fc_input = nn.Linear(4, self.hidden_size)

        # LSTM layer to model the temporal dependencies in the mouse movement.
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        # Fully connected output layer to map the LSTM output to the path coordinates (x, y).
        self.fc_output = nn.Linear(self.hidden_size, 2)


    def forward(self, x):
        # Initial hidden and cell states for the LSTM.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Pass the input through the fully connected input layer and apply ReLU activation.
        x = torch.relu(self.fc_input(x))

        # Duplicate the input along the time dimension (time_steps) to form the LSTM input.
        x = x.unsqueeze(1).repeat(1, self.time_steps, 1)

        # Pass the input through the LSTM layer.
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Pass the LSTM output through the fully connected output layer.
        x = self.fc_output(lstm_out)

        return x
