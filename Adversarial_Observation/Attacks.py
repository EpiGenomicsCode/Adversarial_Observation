import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def activation_map(input_data: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """
    Generate an activation map for an input data given a pre-trained PyTorch model.

    Args:
        input_data (torch.Tensor): Input data as a PyTorch tensor. 
            EX. For 10 MNIST images, input_data.shape = (10, 1, 28, 28)
        model (torch.nn.Module): Pre-trained PyTorch model used to generate the activation map.

    Returns:
        activation_map (numpy.array): Activation map for the input data based on the input model.
    """
    # Set the model to evaluation mode and enable gradient computation for the input data
    model.eval()

    # assert input data is a PyTorch tensor
    assert isinstance(input_data, torch.Tensor), "Input data must be a PyTorch tensor"

    # assert that the input data and model are on the same device
    assert input_data.device == next(model.parameters()).device, "Input data and model must be on the same device"

    # Enable gradient computation for the input data
    input_data.requires_grad = True

    # Make a forward pass through the model and get the predicted class scores for the input data
    preds = model(input_data)

    # Get the indices corresponding to the maximum predicted class scores
    pred_class_idxs = torch.argmax(preds, dim=1)

    # Compute gradients of the scores with respect to the input data pixels
    preds[:, pred_class_idxs].backward(torch.ones_like(preds[:, pred_class_idxs]))

    # Compute the activation map as the absolute value of the gradients
    activation_map = input_data.grad

    # Detach the activation map and move it to CPU as a numpy array
    activation_map = activation_map.detach().cpu().numpy()

    # Return the activation map
    return activation_map

def fgsm_attack(input_data: torch.Tensor, labels: torch.Tensor, epsilon: float, model: torch.nn.Module, loss: torch.nn.Module = torch.nn.CrossEntropyLoss(), device: str = "cpu") -> np.ndarray:
    """
    Generates adversarial example using fast gradient sign method.

    Args:
        input_data (torch.Tensor): A list of input_data to be modified.
            EX: For 10 MNIST images, input_data.shape = (10, 1, 28, 28)
        labels (torch.Tensor): The true labels of the input data.
            EX: For 10 MNIST images, labels.shape = (10,)
        epsilon (float): Magnitude of the perturbation added to the input data.
        model (torch.nn.Module): The neural network model.
        loss (torch.nn.Module): The loss function to use for the computation.
        device (str): Device to use for the computation.
            default: "cpu"

    Returns:
        perturbed (np.ndarray): The perturbed input data.
    """
    # assert that epsilon is a positive value
    assert epsilon >= 0, "Epsilon must be a positive value"

    original_device = input_data.device
    model.eval()

    # assert input data is a PyTorch tensor
    assert isinstance(input_data, torch.Tensor), "Input data must be a PyTorch tensor"

    # assert that the input data and model are on the same device
    assert len(input_data) == len(labels), "Input data and labels must have the same length"


    # Enable gradient computation for the input data
    input_data.requires_grad = True

    # Make a forward pass through the model and get the predicted class scores
    output = model(input_data)

    # Calculate the loss between the predicted output and the true labels
    loss_value = loss(output, labels)

    # Zero the gradients
    model.zero_grad()

    # Perform backpropagation to compute the gradients
    loss_value.backward()

    # Get the gradients of the input data
    gradients = input_data.grad.data

    # Calculate the sign of the gradients
    sign_gradients = gradients.sign()

    # Create the perturbation by scaling the sign of the gradients with epsilon
    perturbation = epsilon * sign_gradients

    # Add the perturbation to the input data to generate the adversarial example
    perturbed = input_data + perturbation

    # Move the perturbed data back to the original device
    perturbed = perturbed.to(original_device)

    # Return the perturbed input data as a numpy array
    return perturbed.detach().cpu().numpy()
