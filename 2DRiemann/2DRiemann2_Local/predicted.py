import os
import sys
import json
import pickle
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch import autograd

class FNN(nn.Module):
    def __init__(self, inputs, layers, init_type='xavier', output2='exp'):
        super(FNN, self).__init__()

        # Normalize inputs
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        self.register_buffer('mean', inputs_tensor.mean(dim=0))
        self.register_buffer('std', inputs_tensor.std(dim=0))

        self.layers = layers
        self.init_type = init_type
        self.output2 = output2  # Transformation for the second output

        self.hidden_weights = nn.ParameterList([
            nn.Parameter(torch.empty(layers[i], layers[i + 1]))
            for i in range(len(layers) - 1)
        ])
        self.hidden_biases = nn.ParameterList([
            nn.Parameter(torch.empty(1, layers[i + 1]))
            for i in range(len(layers) - 1)
        ])

        self.initialize_weights()

    def initialize_weights(self):
        init_methods = {
            'xavier': init.xavier_normal_,
            'he': lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')
        }
        initializer = init_methods.get(self.init_type, init.xavier_normal_)

        for weight in self.hidden_weights:
            initializer(weight)
        for bias in self.hidden_biases:
            init.constant_(bias, 0)

    def forward(self, x):
        # Convert NumPy array to PyTorch tensor if necessary
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.mean.device)
        x = x.to(self.mean.device)

        # Normalize input
        x = (x - self.mean) / self.std

        for i, (weight, bias) in enumerate(zip(self.hidden_weights, self.hidden_biases)):
            x = torch.matmul(x, weight) + bias
            if i < len(self.hidden_weights) - 1:  # Apply activation for hidden layers
                x = torch.tanh(x)

        # Apply transformations to outputs
        x[:, 0] = torch.exp(x[:, 0])  # Exponential transformation for the first output
        if self.output2 == 'exp':
            x[:, 1] = torch.exp(x[:, 1])
        elif self.output2 == '10pow':
            x[:, 1] = 10 ** x[:, 1]

        return x

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

    def save_predictions(self, x_input, filename):
        predictions = self.predict(x_input).cpu().numpy()
        inputs = (
            x_input if isinstance(x_input, np.ndarray) else x_input.cpu().numpy()
        )
        data = np.concatenate((inputs, predictions), axis=1)
        np.save(filename, data)

    def save_parameters(self, filepath):
        params = {
            'hidden_weights': [w.detach().cpu().numpy() for w in self.hidden_weights],
            'hidden_biases': [b.detach().cpu().numpy() for b in self.hidden_biases],
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)

    def print_state_dict(self):
        print("Model's state_dict:")
        for name, param in self.state_dict().items():
            print(name, "\t", param.size())
            
            
# Define the function to perform predictions
def load_and_predict(pth_file, grid, output_file, layers, init_type='xavier', output2='exp'):
    """
    Loads the trained FNN model, performs predictions on a given grid, and saves the results.

    Parameters:
        pth_file (str): Path to the trained .pth file.
        grid (torch.Tensor): Input grid for predictions.
        output_file (str): Path to save the predictions in .npy format.
        layers (list): Layer configuration of the FNN model.
        init_type (str): Initialization type for the model ('xavier' or 'he').
        output2 (str): Transformation type for the second output ('exp' or '10pow').

    Returns:
        None
    """

    # Ensure the grid is a torch.Tensor
    if not isinstance(grid, torch.Tensor):
        raise ValueError("Input grid must be a torch.Tensor")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dummy input data to compute normalization stats
    dummy_inputs = grid.clone()

    # Initialize the model
    model = FNN(inputs=dummy_inputs, layers=layers, init_type=init_type, output2=output2)
    model.to(device)

    # Load the trained weights
    model.load_state_dict(torch.load(pth_file, map_location=device))
    model.eval()

    # Perform predictions
    with torch.no_grad():
        predictions = model(grid).cpu().numpy()

    # Convert the grid to NumPy if needed
    grid_np = grid.cpu().numpy()

    # Combine grid input and predictions
    results = np.concatenate((grid_np, predictions), axis=1)

    # Save the results to a file
    np.save(output_file, results)
    print(f"Predictions saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Define step size and generate grid input
    step_size = 0.01
    step_size_time = 0.01
    x = torch.arange(0.0, 1.0 + step_size, step_size)
    y = torch.arange(0.0, 1.0 + step_size, step_size)
    t = torch.arange(0.0, 0.25+ step_size_time, step_size_time)

    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
    flat_X = X.flatten()
    flat_Y = Y.flatten()
    flat_T = T.flatten()

    # Stack the flattened coordinates into a single tensor
    grid = torch.stack([flat_X, flat_Y, flat_T], dim=1)
    grid = grid.to(dtype=torch.float32)

    # Define model parameters
    layers = [3, 100, 100, 100, 100, 100, 4]  # Example configuration
    pth_file = "./results/model_adam.pth"  # Path to the trained model
    output_file = "./results/predictions101.npy"  # Path to save the predictions

    # Call the function to load the model, predict, and save results
    load_and_predict(pth_file, grid, output_file, layers, init_type='xavier', output2='exp')
