import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt



def saliency_map(input_batch_data, model: torch.nn.Module, input_shape: tuple) -> torch.Tensor:
    """
    Generate a saliency map for an input image given a pre-trained PyTorch model.

    Args:
        input_batch_data (ndarray): Batch of input images as a 4D numpy array.
        model (nn.Module): Pre-trained PyTorch model used to generate the saliency map.
        input_shape (tuple): Shape of the input array.

    Returns:
        saliency_map (ndarray): Saliency map for the input images.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model and input batch data to the appropriate device
    model.to(device)
    input_batch_data = torch.tensor(input_batch_data).to(device)

    # Disable gradient computation for all model parameters
    for param in model.parameters():
        param.requires_grad = False

    saliency_maps = []
    for img in input_batch_data:
        # Convert the input image to a PyTorch tensor with dtype=torch.float32 and enable gradient computation
        img = img.clone().detach().to(torch.float32).requires_grad_(True)
        
        # Move the input image tensor to the same device as the model
        img = img.to(device)

        # Make a forward pass through the model and get the predicted class scores for the input image
        preds = model(img.reshape(input_shape))
        
        # Compute the score and index of the class with the highest predicted score
        score, _ = torch.max(preds, 1)
        
        # Compute gradients of the score with respect to the input image pixels
        score.backward()
        
        # Compute the saliency map by taking the maximum absolute gradient across color channels and normalize the values
        saliency_map = img.grad.abs().max(dim=0)[0]
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        
        saliency_maps.append(saliency_map.cpu().detach().numpy())

    return saliency_maps