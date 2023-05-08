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
        activation_map (torch.Tensor): Activation map for the input data based on the input model.
    """
    # Set the model to evaluation mode and enable gradient computation for the input data
    model.eval()

    # send model to the same device as the input data
    device = input_data.device
    model.to(device)
    
    #  assert it has a batch dimension of 1 
    assert input_data.shape[0] == 1, "Input data must have a batch dimension of 1"
    assert input_data.device == model.parameters().__next__().device, "Input data and model must be on the same device"
    
    # Enable gradient computation for the input data
    input_data.requires_grad = True

    # Make a forward pass through the model and get the predicted class scores for the input data
    preds = model(input_data)

    # Get the index corresponding to the maximum predicted class score
    pred_class_idx = torch.argmax(preds)

    # Get the predicted class score corresponding to the maximum predicted class score
    score = preds[0, pred_class_idx]
    
    # Compute gradients of the score with respect to the input data pixels
    score.backward()

    # Compute the activation map as the absolute value of the gradients
    activation_map = input_data.grad

    # Return the activation map
    return activation_map

def fgsm_attack(input_data: torch.Tensor, label: int, epsilon: float, model: torch.nn.Module, loss:torch.nn.functional = torch.nn.functional.nll_loss,  device: str = "cpu") -> torch.Tensor:
    """
    Generates adversarial example using fast gradient sign method.

    Args:
        input_data (torch.Tensor): The input_data to be modified.
        label (int): The true label of the input data.
        epsilon (float): Magnitude of the perturbation added to the input data.
        model (torch.nn.Module): The neural network model.
        loss (torch.nn.functional): The loss function to use for the computation.
        device (str): Device to use for the computation. 
            default: "cpu"

    Returns:
        perturbed (torch.Tensor): the noise to be added to the input data.
    """
    # assert that epsilon is a positive value
    assert epsilon >= 0, "Epsilon must be a positive value"


    original_device = input_data.device
    model.eval()
    
    data = input_data.clone().to(device)
    data.requires_grad = True

    # Forward pass to get the prediction
    output = model(data)

    # Calculate the loss
    loss = loss(output, torch.tensor([label]).to(device))

    # Backward pass to get the gradient
    model.zero_grad()
    loss.backward()

    # Create the perturbed data by adjusting each pixel of the input data
    perturbed = epsilon * torch.sign(data.grad.data)
    perturbed = input_data + perturbed

    # copy the perturbed data back to the original device
    perturbed = perturbed.to(original_device)

    # reshape perturbed data to match the input data shape
    perturbed = perturbed.reshape(input_data.shape)

    # Add the perturbation to the input data
    perturbed = torch.clamp(perturbed, 0, 1)

    # Return the perturbed data
    return perturbed.detach()
