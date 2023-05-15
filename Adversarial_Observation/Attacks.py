import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def activation_map(input_data: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """
    Generate an activation map for an input data given a pre-trained PyTorch model.

    Args:
        input_data (torch.Tensor): Input data as a PyTorch tensor.
        model (torch.nn.Module): Pre-trained PyTorch model used to generate the activation map.

    Returns:
        activation_map (numpy.array): Activation map for the input data based on the input model.
    """
    # Set the model to evaluation mode and enable gradient computation for the input data
    model.eval()
    output_data = []

    
    #  assert it has a batch dimension of 1 
    assert input_data.device == model.parameters().__next__().device, "Input data and model must be on the same device"

    for input in input_data:
        # Enable gradient computation for the input data
        input.requires_grad = True

        # Make a forward pass through the model and get the predicted class scores for the input data
        preds = model(input.unsqueeze(0))

        # Get the index corresponding to the maximum predicted class score
        pred_class_idx = torch.argmax(preds)

        # Get the predicted class score corresponding to the maximum predicted class score
        score = preds[0, pred_class_idx]
        
        # Compute gradients of the score with respect to the input data pixels
        score.backward()

        # Compute the activation map as the absolute value of the gradients
        activation_map = input.grad

        # save the activation map
        output_data.append(activation_map.detach().cpu().numpy())

    # Return the activation map
    return np.array(output_data)


def fgsm_attack(input_data: torch.Tensor, labels: torch.Tensor, epsilon: float, model: torch.nn.Module, loss: torch.nn.Module = torch.nn.CrossEntropyLoss(),  device: str = "cpu") -> np.ndarray:
    """
    Generates adversarial example using fast gradient sign method.

    Args:
        input_data (torch.Tensor): A list of input_data to be modified.
        labels (torch.Tensor): The true labels of the input data.
        epsilon (float): Magnitude of the perturbation added to the input data.
        model (torch.nn.Module): The neural network model.
        loss (torch.nn.Module): The loss function to use for the computation.
        device (str): Device to use for the computation.
            default: "cpu"

    Returns:
        perturbed (np.ndarray): the noise to be added to the input data.
    """
    # assert that epsilon is a positive value
    assert epsilon >= 0, "Epsilon must be a positive value"

    original_device = input_data.device
    model.eval()

    assert len(input_data) == len(labels), "Input data and labels must have the same length"

    output_data = []
    for input_data, label in zip(input_data, labels):
        data = input_data.to(device)
        data.requires_grad = True

        # Make a forward pass through the model and get the predicted class scores
        output = model(data.unsqueeze(0))

        # Get the index corresponding to the maximum predicted class score
        loss_value = loss(output, torch.tensor([label]).to(device))

        # Perform backpropagation to compute gradients
        model.zero_grad()
        loss_value.backward()

        # Create the perturbed data by adjusting each pixel of the input data
        perturbed = epsilon * data.grad.data.sign()
        perturbed = data + perturbed

        # copy the perturbed data back to the original device
        perturbed = perturbed.to(original_device)

        # reshape perturbed data to match the input data shape
        perturbed = perturbed.reshape(input_data.shape)

        # save the perturbed data
        output_data.append(perturbed.detach().cpu().numpy())

    # Return the perturbed data
    return np.array(output_data)
