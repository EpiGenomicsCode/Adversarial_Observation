import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


def fgsm_attack(input_batch_data: torch.Tensor, model: torch.nn.Module, input_shape: tuple, epsilon: float = 0.0) -> torch.Tensor:
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_batch_data = input_batch_data.to(device).detach().requires_grad_(True)
    preds = model(input_batch_data)
    targets = torch.argmax(preds, dim=1)
    loss = F.cross_entropy(preds, targets)

    model.zero_grad()
    loss.backward()

    adversarial_batch_data = input_batch_data + (epsilon * input_batch_data.grad.sign())

    # Debugging
    if epsilon == 0:
        assert torch.allclose(adversarial_batch_data, input_batch_data), "Adversarial images should be the same as original."

    return adversarial_batch_data.detach()

def compute_gradients(model, img, target_class):
    preds = model(img)
    target_score = preds[0, target_class]
    return torch.autograd.grad(target_score, img)[0]

def generate_adversarial_examples(input_batch_data, model, method='fgsm', **kwargs):
    if method == 'fgsm':
        return fgsm_attack(input_batch_data, model, **kwargs)
    # Implement other attack methods as needed

def visualize_adversarial_examples(original, adversarial):
    # Code to visualize original vs adversarial images
    pass

def log_metrics(success_rate, average_perturbation):
    logging.info(f'Success Rate: {success_rate}, Average Perturbation: {average_perturbation}')

class Config:
    def __init__(self, epsilon=0.1, attack_method='fgsm'):
        self.epsilon = epsilon
        self.attack_method = attack_method
