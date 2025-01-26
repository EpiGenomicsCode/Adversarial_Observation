import torch
import logging
import os
from datetime import datetime
from torch.nn import Softmax
from .utils import fgsm_attack, pgd_attack, compute_success_rate, log_metrics, visualize_adversarial_examples
from .utils import seed_everything

class AdversarialTester:
    def __init__(self, model: torch.nn.Module, epsilon: float = 0.1, attack_method: str = 'fgsm', alpha: float = 0.01, 
                 num_steps: int = 40, device=None, save_dir: str = './results', seed: int = 42):
        seed_everything(seed)
        self.model = model
        self.epsilon = epsilon
        self.attack_method = attack_method
        self.alpha = alpha
        self.num_steps = num_steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir

        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        self.model.to(self.device)
        self.model.eval()

        self._setup_logging()

    def _setup_logging(self):
        log_file = os.path.join(self.save_dir, f"attack_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(filename=log_file, level=logging.DEBUG)
        logging.info(f"Started adversarial testing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Using model: {self.model.__class__.__name__}")
        logging.info(f"Attack Method: {self.attack_method}, Epsilon: {self.epsilon}, Alpha: {self.alpha}, Steps: {self.num_steps}")

    def test_attack(self, input_batch_data: torch.Tensor):
        input_batch_data = input_batch_data.to(self.device)
        adversarial_images = self._generate_adversarial_images(input_batch_data)

        # Save and log images
        self._save_images(input_batch_data, adversarial_images)
        self._compute_and_log_metrics(input_batch_data, adversarial_images)

    def _generate_adversarial_images(self, input_batch_data: torch.Tensor):
        logging.info(f"Starting attack with method: {self.attack_method}")
        if self.attack_method == 'fgsm':
            return fgsm_attack(input_batch_data, self.model, self.epsilon, self.device)
        elif self.attack_method == 'pgd':
            return pgd_attack(input_batch_data, self.model, self.epsilon, self.alpha, self.num_steps, self.device)
        else:
            raise ValueError(f"Unsupported attack method: {self.attack_method}")

    def _save_images(self, original_images: torch.Tensor, adversarial_images: torch.Tensor):
        for i in range(original_images.size(0)):
            original_image_path = os.path.join(self.save_dir, f"original_{i}.png")
            adversarial_image_path = os.path.join(self.save_dir, f"adversarial_{i}.png")
            visualize_adversarial_examples(original_images, adversarial_images, original_image_path, adversarial_image_path)

    def _compute_and_log_metrics(self, original_images: torch.Tensor, adversarial_images: torch.Tensor):
        original_predictions = torch.argmax(self.model(original_images), dim=1)
        adversarial_predictions = torch.argmax(self.model(adversarial_images), dim=1)

        success_rate = compute_success_rate(original_predictions, adversarial_predictions)
        average_perturbation = torch.mean(torch.abs(adversarial_images - original_images)).item()

        log_metrics(success_rate, average_perturbation)
        self._save_metrics(success_rate, average_perturbation)

        logging.info(f"Success Rate: {success_rate:.4f}, Average Perturbation: {average_perturbation:.4f}")

    def _save_metrics(self, success_rate: float, avg_perturbation: float):
        """
        Save the metrics (success rate and average perturbation) to a file.
        """
        metrics_file = os.path.join(self.save_dir, "attack_metrics.txt")
        with open(metrics_file, 'a') as f:
            f.write(f"Success Rate: {success_rate:.4f}, Average Perturbation: {avg_perturbation:.4f}\n")