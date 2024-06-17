import pandas as pd
import numpy as np
import math

from src.config import config
import src.preprocessing.preprocessors as pp
from src.preprocessing.data_management import load_dataset, save_model, load_model



def layer_neurons_output(current_layer_neurons_weighted_sums, current_layer_neurons_activation_function):
    if current_layer_neurons_activation_function == "linear":
        return current_layer_neurons_weighted_sums
    elif current_layer_neurons_activation_function == "sigmoid":
        return 1 / (1 + np.exp(-current_layer_neurons_weighted_sums))
    elif current_layer_neurons_activation_function == "tanh":
        return (np.exp(current_layer_neurons_weighted_sums) - np.exp(-current_layer_neurons_weighted_sums)) / \
               (np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums))
    elif current_layer_neurons_activation_function == "relu":
        return np.maximum(0, current_layer_neurons_weighted_sums)
    else:
        raise ValueError("Unsupported activation function: {}".format(current_layer_neurons_activation_function))

def predict(X):
    model = load_model("two_input_xor_nn.pkl")
    theta0 = model["params"]["biases"]
    theta = model["params"]["weights"]
    activations = model["activations"]

    # Initialize the input layer
    h = [None] * config.NUM_LAYERS
    h[0] = X

    # Forward pass through the layers
    for l in range(1, config.NUM_LAYERS):
        z = theta0[l] + np.dot(h[l-1], theta[l])
        h[l] = layer_neurons_output(z, activations[l])

    return h[config.NUM_LAYERS-1][:,1]


    

if __name__ == "__main__":
    # Sample input (for a two-input XOR problem)
    X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    

    predictions = predict(X_test)
    print("Predictions:\n", predictions)

