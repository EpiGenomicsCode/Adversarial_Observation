import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def pgd_attack(input_batch_data: torch.Tensor, labels: torch.Tensor, epsilon: float, alpha: float, num_iterations: int, model: torch.nn.Module, loss: torch.nn.Module = torch.nn.CrossEntropyLoss(), device: str = "cpu") -> np.ndarray:
    """
    Generates adversarial examples using the Projected Gradient Descent (PGD) attack.

    Args:
        input_batch_data (torch.Tensor): Input data to be modified. Shape: (batch_size, channels, height, width)
        labels (torch.Tensor): The true labels of the input data. Shape: (batch_size,)
        epsilon (float): Magnitude of the perturbation added to the input data.
        alpha (float): Step size for each iteration of perturbation.
        num_iterations (int): Number of iterations to perform the attack.
        model (torch.nn.Module): The neural network model.
        loss (torch.nn.Module): The loss function to use for the computation. Default: torch.nn.CrossEntropyLoss()
        device (str): Device to use for the computation. Default: "cpu"

    Returns:
        np.ndarray: The perturbed input data.
    """
    assert epsilon >= 0, "Epsilon must be a non-negative value"
    assert alpha > 0, "Alpha must be a positive value"
    assert num_iterations > 0, "Number of iterations must be greater than 0"
    assert isinstance(input_batch_data, torch.Tensor), "Input data must be a PyTorch tensor"
    assert len(input_batch_data) == len(labels), "Input data and labels must have the same length"
    assert input_batch_data.device == next(model.parameters()).device, "Input data and model must be on the same device"

    model.eval()
    original_device = input_batch_data.device

    new_input_data = []
    for input_data, label in zip(input_batch_data, labels):
        perturbed = input_data.clone().detach().to(device)
        perturbed.requires_grad = True

        for _ in range(num_iterations):
            output = model(perturbed.unsqueeze(0))
            loss_value = loss(output, label.unsqueeze(0))

            model.zero_grad()
            loss_value.backward()

            with torch.no_grad():
                gradients = perturbed.grad.data
                sign_gradients = torch.sign(gradients)

                perturbation = alpha * sign_gradients
                perturbed = perturbed + perturbation

                perturbed = torch.max(torch.min(perturbed, input_data + epsilon), input_data - epsilon)
                perturbed = torch.clamp(perturbed, min=0, max=1)

            perturbed.requires_grad = True

        perturbed = perturbed.to(original_device)
        new_input_data.append(perturbed.cpu().detach().numpy())

    return np.array(new_input_data)


def deepfool_attack(input_batch_data: torch.Tensor, model: torch.nn.Module, num_classes: int, max_iter: int = 50, epsilon: float = 1e-6) -> np.ndarray:
    """
    Generates adversarial examples using the DeepFool attack.

    Args:
        input_batch_data (torch.Tensor): Input data to be modified. Shape: (batch_size, channels, height, width)
        model (torch.nn.Module): The neural network model.
        num_classes (int): Number of classes in the classification task.
        max_iter (int): Maximum number of iterations for the attack. Default: 50.
        epsilon (float): Small value to avoid division by zero. Default: 1e-6.

    Returns:
        np.ndarray: The perturbed input data.
    """
    assert isinstance(input_batch_data, torch.Tensor), "Input data must be a PyTorch tensor"

    model.eval()

    with torch.no_grad():
        perturbed_data = input_batch_data.clone()

        for i in range(len(perturbed_data)):
            input_data = perturbed_data[i].unsqueeze(0).clone().detach()
            input_data.requires_grad = True

            output = model(input_data)
            _, initial_pred = output.max(1)

            perturbation = torch.zeros_like(input_data)

            for _ in range(max_iter):
                input_data.grad = None
                output = model(input_data)
                _, current_pred = output.max(1)

                if current_pred != initial_pred:
                    break

                grad = torch.autograd.grad(output[:, initial_pred], input_data, retain_graph=True)[0]
                grad_flat = grad.view(-1)
                output_flat = output.view(-1)

                perturbation_norms = torch.abs(output_flat - output_flat[initial_pred]) / grad_flat.norm()

                perturbed_label = perturbation_norms.argmin()
                perturbation += (perturbation_norms[perturbed_label] + epsilon) * grad[perturbed_label]

                input_data.data += perturbation

            perturbed_data[i] = input_data.squeeze(0)

    return perturbed_data.numpy()


def bim_attack(input_batch_data: torch.Tensor, labels: torch.Tensor, epsilon: float, alpha: float, num_iterations: int, model: torch.nn.Module, loss: torch.nn.Module = torch.nn.CrossEntropyLoss(), device: str = "cpu") -> np.ndarray:
    """
    Generates adversarial examples using the Basic Iterative Method (BIM) attack.

    Args:
        input_batch_data (torch.Tensor): Input data to be modified. Shape: (batch_size, channels, height, width)
        labels (torch.Tensor): The true labels of the input data. Shape: (batch_size,)
        epsilon (float): Magnitude of the perturbation added to the input data.
        alpha (float): Step size for each iteration of perturbation.
        num_iterations (int): Number of iterations to perform the attack.
        model (torch.nn.Module): The neural network model.
        loss (torch.nn.Module): The loss function to use for the computation. Default: torch.nn.CrossEntropyLoss()
        device (str): Device to use for the computation. Default: "cpu"

    Returns:
        np.ndarray: The perturbed input data.
    """
    assert epsilon >= 0, "Epsilon must be a non-negative value"
    assert alpha > 0, "Alpha must be a positive value"
    assert num_iterations > 0, "Number of iterations must be greater than 0"
    assert isinstance(input_batch_data, torch.Tensor), "Input data must be a PyTorch tensor"
    assert len(input_batch_data) == len(labels), "Input data and labels must have the same length"
    assert input_batch_data.device == next(model.parameters()).device, "Input data and model must be on the same device"

    model.eval()
    original_device = input_batch_data.device

    new_input_data = []
    for input_data, label in zip(input_batch_data, labels):
        perturbed = input_data.clone().detach().to(device)
        perturbed.requires_grad = True

        for _ in range(num_iterations):
            output = model(perturbed.unsqueeze(0))
            loss_value = loss(output, label.unsqueeze(0))

            model.zero_grad()
            loss_value.backward()

            gradients = perturbed.grad.data
            sign_gradients = torch.sign(gradients)

            perturbation = alpha * sign_gradients
            perturbed = torch.clamp(perturbed + perturbation, min=0, max=1)

            perturbed = perturbed.detach()

        perturbed = perturbed.to(original_device)
        new_input_data.append(perturbed.cpu().detach().numpy())

    return np.array(new_input_data)


def fgsm_attack(input_batch_data: torch.Tensor, labels: torch.Tensor, epsilon: float, model: torch.nn.Module, loss: torch.nn.Module = torch.nn.CrossEntropyLoss(), device: str = "cpu") -> np.ndarray:
    """
    Generates adversarial examples using the Fast Gradient Sign Method (FGSM).

    Args:
        input_batch_data (torch.Tensor): Input data to be modified. Shape: (batch_size, channels, height, width)
        labels (torch.Tensor): The true labels of the input data. Shape: (batch_size,)
        epsilon (float): Magnitude of the perturbation added to the input data.
        model (torch.nn.Module): The neural network model.
        loss (torch.nn.Module): The loss function to use for the computation. Default: torch.nn.CrossEntropyLoss()
        device (str): Device to use for the computation. Default: "cpu"

    Returns:
        np.ndarray: The perturbed input data.
    """
    assert epsilon >= 0, "Epsilon must be a non-negative value"
    assert isinstance(input_batch_data, torch.Tensor), "Input data must be a PyTorch tensor"
    assert len(input_batch_data) == len(labels), "Input data and labels must have the same length"
    assert input_batch_data.device == next(model.parameters()).device, "Input data and model must be on the same device"

    model.eval()
    original_device = input_batch_data.device

    new_input_data = []
    for input_data, label in zip(input_batch_data, labels):
        input_data = input_data.to(device)
        input_data.requires_grad = True

        output = model(input_data.unsqueeze(0))
        loss_value = loss(output, label.unsqueeze(0))

        model.zero_grad()
        loss_value.backward()

        gradients = input_data.grad.data
        sign_gradients = torch.sign(gradients)

        perturbation = epsilon * sign_gradients
        perturbed = input_data + perturbation

        perturbed = perturbed.to(original_device)

        new_input_data.append(perturbed.cpu().detach().numpy())

    return np.array(new_input_data)


