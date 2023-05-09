import torch
import pickle
import numpy as np

class PathDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, time_steps):
        self.time_steps = time_steps

        # Load the dataset from the provided file.
        with open(data_file, 'rb') as file:
            self.data = pickle.load(file)


    # Return the length of the dataset.
    def __len__(self):
        return len(self.data)


    # Get the path at the specified index.
    def __getitem__(self, idx):
        path = self.data[idx]

        # Extract the start and target coordinates from the path.
        start_x, start_y = path[0][:2]
        target_x, target_y = path[-1][:2]

        # Create input tensor from the start and target coordinates.
        input_data = torch.tensor([start_x, start_y, target_x, target_y])

        # Interpolate the path and get the target data (x, y).
        target_data = self.interpolate_path(path)

        return input_data, target_data


    def interpolate_path(self, path):
        # Convert the path to a NumPy array.
        path_array = np.array(path)

        # Create new indices for interpolation.
        new_indices = np.linspace(0, len(path) - 1, self.time_steps)

        # Perform linear interpolation for each column (x, y) in the path array.
        interpolated_path = np.column_stack([
            np.interp(new_indices, np.arange(len(path)), path_array[:, 0]),
            np.interp(new_indices, np.arange(len(path)), path_array[:, 1])
        ])

        return interpolated_path