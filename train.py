import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from mouse_net import MouseNet
from data_loader import PathDataset
import argparse

# Parse command line arguments.
parser = argparse.ArgumentParser(description="Train the MouseNet model.")
parser.add_argument('--time_steps', type=int, default=300, help="Number of time steps for the model.")
parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate for the optimizer.")
parser.add_argument('--num_epochs', type=int, default=10000, help="Number of training epochs.")
parser.add_argument('--improvement_patience', type=int, default=50, help="Patience for early stopping.")
parser.add_argument('--scheduler_patience', type=int, default=20, help="Patience for learning rate scheduler.")
parser.add_argument('--batch_size', type=int, default=512, help="Batch size for training.")
parser.add_argument('--data_file', type=str, default='training_data.pkl', help="Path to the training data file.")
parser.add_argument('--model_dir', type=str, default='models', help="Directory to save the trained model.")
parser.add_argument('--model_name', type=str, default='mouse_net.pth', help="Name of the trained model file.")
args = parser.parse_args()

# Model Parameters.
time_steps = args.time_steps

# Training Parameters.
learning_rate = args.learning_rate
num_epochs = args.num_epochs
improvement_patience = args.improvement_patience
scheduler_patience = args.scheduler_patience
batch_size = args.batch_size
data_file = args.data_file
model_dir = args.model_dir
model_name = args.model_name
model_path = os.path.join(model_dir, model_name)

# Create the model directory if it doesn't exist.
os.makedirs(model_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MouseNet(time_steps, device)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience, factor=0.1, verbose=True, threshold=1e-4)

# Load the most recent model if one exists.
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded best model from '{model_path}'.")

# Load the training data from a file
dataset = PathDataset(data_file, time_steps)
print(f"Training data loaded from {data_file}.")

# Split the dataset into training and validation sets.
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop.
early_stopping_counter = 0
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation.
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f"[Epoch {epoch + 1}/{num_epochs}] (Train Loss: {train_loss:.6f}) (Validation Loss: {val_loss:.6f})")

    # Early stopping.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), model_path)
        print(f"Saved best model to '{model_path}'.")
    else:
        early_stopping_counter += 1
        print(f"No improvement in validation loss for {early_stopping_counter} epoch(s).")
        if early_stopping_counter >= improvement_patience:
            print("Early stopping triggered. Stopping training.")
            break
