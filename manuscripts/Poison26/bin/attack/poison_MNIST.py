import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os, json
import argparse
import time
import numpy as np
from tqdm import trange
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import csv

from PIL import Image

from Adversarial_Observation.Swarm import PSO as ParticleSwarm
from Adversarial_Observation.utils import seed_everything as seedEverything
from captum.attr import Saliency, IntegratedGradients, DeepLiftShap

from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")  # disables GUI, enables PNG/PDF saving

# Import our new model architectures
from MobileNet import MobileNet
from RegNetX import RegNetX_400MF


# ----------------------------
# Model definitions
# ----------------------------
class MNISTModel1(nn.Module):
    def __init__(self):
        super(MNISTModel1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # raw logits
        return x

class MNISTModel2(nn.Module):
    def __init__(self):
        super(MNISTModel2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)          # 28x28 -> 26x26
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)         # 26 -> 24
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)  # 24 -> 12
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.4)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)         # 12 -> 10
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)         # 10 -> 8
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)  # 8 -> 4
        self.bn6 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.4)

        self.conv7 = nn.Conv2d(64, 128, kernel_size=4)        # 4 -> 1
        self.bn7 = nn.BatchNorm2d(128)

        self.dropout3 = nn.Dropout(0.4)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn7(self.conv7(x)))
        x = torch.flatten(x, 1)
        x = self.dropout3(x)
        x = self.fc(x)
        return x

class MNISTWrapper2D(nn.Module):
    def __init__(self, backbone_class, num_classes=10):
        super(MNISTWrapper2D, self).__init__()
        dummy_batch = torch.zeros(1, 1, 28, 28)
        self.backbone = backbone_class(one_batch=dummy_batch, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

# ----------------------------
# Utility: Dynamic Model Loader
# ----------------------------
def load_model(model_path, arch):
    if arch == "basic":
        model = MNISTModel1()
    elif arch == "adv":
        model = MNISTModel2()
    elif arch == "MobileNet":
        dummy_batch = torch.zeros(1, 1, 28, 28)
        model = MobileNet(one_batch=dummy_batch, num_classes=10)
    elif arch == "RegNetX":
        dummy_batch = torch.zeros(1, 1, 28, 28)
        model = RegNetX_400MF(one_batch=dummy_batch, num_classes=10)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
        
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    return model

# --- Cost function: targeted attack ---
def costFunc(model, input, target_label):
    img_tensor = input.clone().detach().float()

    # Ensure correct shape: [1, 1, 28, 28]
    if img_tensor.ndim == 1:
        img_tensor = img_tensor.view(1, 1, 28, 28)
    elif img_tensor.ndim == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    output = model(img_tensor)
    prob_target = output[0, target_label]   # shape: scalar
    return prob_target.item()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run APSO adversarial attack on MNIST test image')
    parser.add_argument('--modelPath', type=str, required=True, help='Path to trained MNIST model (.pt/.pth)')
    parser.add_argument('--arch', type=str, required=True, choices=['basic', 'adv', 'MobileNet', 'RegNetX'], help='Architecture of the trained model')
    parser.add_argument('--outputPath', type=str, default="./output/", help='Output directory')
    parser.add_argument('--particleNum', type=int, default=100, help='Number of particles in swarm')
    parser.add_argument('--epochs', type=int, default=50, help='Number of APSO epochs')
    parser.add_argument('--inertiaWeight', type=float, default=0.8, help='Inertia weight for APSO')
    parser.add_argument('--cognitiveWeight', type=float, default=0.2, help='Cognitive weight for APSO')
    parser.add_argument('--socialWeight', type=float, default=1.5, help='Social weight for APSO')
    parser.add_argument('--randomSeed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--sourceIndex', type=int, default=0, help="Index of source image.")
    parser.add_argument('--targetLabel', type=int, default=-1, help='Target label for the attack (0-9)')
    parser.add_argument('--maxRetries', type=int, default=5, help="Maximum number of retries if attack fails.")
    parser.add_argument('--particleGrowth', type=float, default=2.0, help="Growth factor for number of particles on each retry (e.g., 1.5 means increase by 50% per retry).")
    parser.add_argument('--startFromBaseline', action='store_true', help='If set, initialize particles as sparse noise added to the baseline image.')
    parser.add_argument('--variableInit', action='store_true', help='If set, sample epsilon and sparsity per particle from specified min/max ranges.')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Static epsilon for particle initialization')
    parser.add_argument('--epsilonMin', type=float, default=0.1, help='Minimum epsilon when using variable initialization')
    parser.add_argument('--epsilonMax', type=float, default=1.0, help='Maximum epsilon when using variable initialization')
    parser.add_argument('--sparsity', type=float, default=0.5, help='Sparsity of initialization vectors (0-1)')
    parser.add_argument('--sparsityMin', type=float, default=0.1, help='Minimum sparsity when using variable initialization')
    parser.add_argument('--sparsityMax', type=float, default=0.9, help='Maximum sparsity when using variable initialization')
    return parser.parse_args()

# --- Logging ---
def log_probabilities(model, img_tensor, epoch, prob_log):
    with torch.no_grad():
        pred_logits = model(img_tensor)
        probs = torch.softmax(pred_logits, dim=1).cpu().numpy().flatten()
    prob_log.append([epoch] + probs.tolist())
    return probs

# --- XAI attribution ---
def run_xai(model, img_tensor, target_label, apsoswarm, output_path, epoch_padded, name, suffix=""):
    saliency = Saliency(model)
    attr_sal = saliency.attribute(img_tensor, target=target_label).detach().cpu().numpy().squeeze()
    plt.figure(figsize=(3, 3))
    plt.imshow(attr_sal, cmap="seismic", vmin=0, vmax=1.0)
    plt.axis("off")
    plt.savefig(f"{output_path}/{epoch_padded}_{name}_saliency{suffix}.png", format="png", bbox_inches="tight", pad_inches=0)
    plt.close()

    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(img_tensor)
    attr_ig = ig.attribute(img_tensor, baselines=baseline, target=target_label, n_steps=50)
    attr_ig = attr_ig.detach().cpu().numpy().squeeze()
    plt.figure(figsize=(3, 3))
    plt.imshow(attr_ig, cmap="seismic", vmin=0, vmax=0.8)
    plt.axis("off")
    plt.savefig(f"{output_path}/{epoch_padded}_{name}_intgrad{suffix}.png", format="png", bbox_inches="tight", pad_inches=0)
    plt.close()

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    bg_indices = np.random.choice(len(test_dataset), 50, replace=False)
    bg_imgs = torch.stack([test_dataset[i][0] for i in bg_indices]).to("cpu")
    dl_shap = DeepLiftShap(model)
    attr_shap = dl_shap.attribute(img_tensor, baselines=bg_imgs, target=target_label)
    attr_shap = attr_shap.detach().cpu().numpy().squeeze()
    plt.figure(figsize=(3, 3))
    plt.imshow(attr_shap, cmap="seismic", vmin=0, vmax=0.8)
    plt.axis("off")
    plt.savefig(f"{output_path}/{epoch_padded}_{name}_shap{suffix}.png", format="png", bbox_inches="tight", pad_inches=0)
    plt.close()

def visualize_position(APSO, epoch, name, output_path, target_label, original_img: np.ndarray = None, denoise: bool = False):
    os.makedirs(output_path, exist_ok=True)
    epoch_padded = f"{epoch:02d}"

    if denoise and original_img is not None:
        best_particle_vector = reduce_excess_perturbations(APSO, original_img.squeeze(), APSO.pos_best_g.detach().cpu().numpy().squeeze().reshape(28, 28).copy(), target_label)
        suffix = "_denoise"
    else:
        best_particle_vector = APSO.pos_best_g.detach().cpu().numpy().copy()
        suffix = ""
    img_28x28 = best_particle_vector.reshape(28, 28)

    plt.figure(figsize=(3, 3))
    plt.imshow(img_28x28, cmap='gray', vmin=0, vmax=1 if img_28x28.max() <= 1 else 255)
    plt.axis('off')
    plt.savefig(f"{output_path}/{epoch_padded}_{name}_particle_best{suffix}.png", format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

    model = APSO.model
    model.eval()
    img_tensor = torch.tensor(best_particle_vector, dtype=torch.float32).reshape(1, 1, 28, 28)
    img_tensor.requires_grad = True

    run_xai(model, img_tensor, target_label, APSO.swarm, output_path, epoch_padded, name, suffix)

def particle_comparison_analysis(attacker, adv_img: np.ndarray, original_img: np.ndarray, single_misclassification_target: int, outputPath: str, denoise: bool = False):
    if denoise:
        suffix = "_denoise"
    else:
        suffix = ""

    adv_img_np = (adv_img.detach().cpu().numpy().squeeze() if isinstance(adv_img, torch.Tensor) else np.array(adv_img).squeeze())
    original_img_np = (original_img.detach().cpu().numpy().squeeze() if isinstance(original_img, torch.Tensor) else np.array(original_img).squeeze())

    plt.figure(figsize=(3, 3))
    plt.imshow(adv_img_np, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.savefig(f"{outputPath}/best_particle{suffix}.png", format='png', bbox_inches='tight', pad_inches=0)
    np.savetxt(f"{outputPath}/best_particle_image{suffix}.csv", [adv_img_np.flatten()], delimiter=',')

    diff_image = adv_img_np - original_img_np
    plt.figure(figsize=(3, 3))
    plt.imshow(diff_image, cmap="seismic", vmin=-1, vmax=1)
    plt.axis("off")
    plt.savefig(f"{outputPath}/attack-vector_best_particle{suffix}.png", format='png', bbox_inches='tight', pad_inches=0)
    np.savetxt(f"{outputPath}/attack-vector_best_particle{suffix}.csv", [diff_image.flatten()], delimiter=',')

    device = next(attacker.model.parameters()).device
    output = attacker.model(torch.tensor(adv_img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device))
    softmax_output = torch.nn.functional.softmax(output.squeeze(), dim=0)
    confidence_values = softmax_output.detach().cpu().numpy().tolist()
    max_output_value = float(max(confidence_values))
    max_output_class = confidence_values.index(max_output_value)

    with open(f"{outputPath}/best_particle_stats{suffix}.tsv", 'w') as f:
        f.write("Class\tConfidence\n")
        for i, conf in enumerate(confidence_values):
            f.write(f"{i}\t{conf}\n")
        f.write(f"\nBest Class\t{max_output_class}\n")
        f.write(f"Max Confidence\t{max_output_value}\n")
        f.write(f"Target Class\t{single_misclassification_target}\n")

    print(f"Best Class\t{max_output_class}")
    print(f"Target Class\t{single_misclassification_target}")

def reduce_excess_perturbations(attacker, original_img, adv_img, target_label, tol=1e-3, max_iter=10, margin=1e-1, max_passes=10, clip_min=0.0, clip_max=1.0,):
    if isinstance(original_img, torch.Tensor):
        original_img = original_img.detach().cpu().numpy()
    if isinstance(adv_img, torch.Tensor):
        adv_img = adv_img.detach().cpu().numpy()

    adv_img = adv_img.copy()

    model = attacker.model
    model.eval()
    device = next(model.parameters()).device

    def confident(img_np):
        img_t = (torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0).to(device))
        with torch.no_grad():
            logits = model(img_t).squeeze(0)

        target_logit = logits[target_label]
        other_logits = torch.cat([logits[:target_label], logits[target_label + 1 :]])
        return target_logit > other_logits.max() + margin

    for _ in range(max_passes):
        any_change = False

        diff = np.abs(adv_img - original_img)
        flat_order = np.argsort(-diff, axis=None)

        for idx in flat_order:
            i, j = np.unravel_index(idx, diff.shape)

            orig = original_img[i, j]
            adv = adv_img[i, j]

            if abs(orig - adv) < tol:
                continue

            adv_img[i, j] = orig
            if confident(adv_img):
                any_change = True
                continue

            adv_img[i, j] = adv
            delta = adv - orig

            low, high = 0.0, 1.0
            best_frac = 1.0

            for _ in range(max_iter):
                mid = 0.5 * (low + high)
                candidate = orig + mid * delta
                candidate = np.clip(candidate, clip_min, clip_max)

                adv_img[i, j] = candidate

                if confident(adv_img):
                    best_frac = mid
                    high = mid
                    any_change = True
                else:
                    low = mid

                if high - low < tol:
                    break

            adv_img[i, j] = np.clip(
                orig + best_frac * delta,
                clip_min,
                clip_max,
            )

        if not any_change:
            break

    return adv_img

def main() -> None:
    args = parse_arguments()

    if args.randomSeed is not None:
        print(f"Random seed set: {args.randomSeed}")
        seedEverything(args.randomSeed)

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    modelSize = 784
    currentParticles = args.particleNum
    epochs = args.epochs
    sparsity = args.sparsity

    epsilon_static = args.epsilon 
    epsilon_min = args.epsilonMin
    epsilon_max = args.epsilonMax
    sparsity_static = args.sparsity 
    sparsity_min = args.sparsityMin
    sparsity_max = args.sparsityMax

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.modelPath, args.arch)
    model.to(device)
    model.eval()

    if args.sourceIndex < 0 or args.sourceIndex >= len(test_dataset):
        raise ValueError(f"Image index {args.sourceIndex} is out of bounds. Dataset size: {len(test_dataset)}")

    baseline_img, baseline_label = test_dataset[args.sourceIndex]

    single_misclassification_target = args.targetLabel
    if args.targetLabel < 0 or args.targetLabel > 9:
        single_misclassification_target = (baseline_label + 1) % 10
        print(f"Invalid target label detected. Overriding to {single_misclassification_target}")

    def wrapped_cost(_, img):
        return costFunc(model, img, single_misclassification_target)

    assert baseline_label != single_misclassification_target, \
        "Target classes should be different for misclassification."

    print(f"Original class: {baseline_label}")
    print(f"Misclassification target class: {single_misclassification_target}")

    os.makedirs(args.outputPath, exist_ok=True)
    plt.figure(figsize=(3, 3))
    plt.imshow(baseline_img.squeeze(), cmap='gray', vmin=0, vmax=1 if baseline_img.max() <= 1 else 255)
    plt.axis('off')
    plt.savefig(os.path.join(args.outputPath, "original.png"), format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

    attempt = 0
    success = False
    while attempt < args.maxRetries and not success:
        print(f"\nAttempt #{attempt + 1} with {currentParticles} particles")

        initialPoints = []
        baseline_np = baseline_img.view(-1).detach().cpu().numpy()

        for _ in range(currentParticles):
            if args.variableInit:
                eps_i = np.random.uniform(epsilon_min, epsilon_max)
                sparsity_i = np.clip(np.random.uniform(sparsity_min, sparsity_max), 0, 1)
            else:
                eps_i = epsilon_static
                sparsity_i = np.clip(sparsity_static, 0, 1)

            if args.startFromBaseline:
                noise = np.random.uniform(-eps_i, eps_i, size=(modelSize,)).astype(np.float32)
                mask = np.random.choice([0, 1], size=noise.shape, p=[sparsity_i, 1 - sparsity_i]).astype(np.float32)
                sparse_noise = noise * mask
                arr = np.clip(baseline_np + sparse_noise, 0.0, 1.0)
            else:
                arr = np.random.uniform(0.0, 1.0, size=(modelSize,)).astype(np.float32)
                mask = np.random.choice([0, 1], size=arr.shape, p=[sparsity_i, 1 - sparsity_i]).astype(np.float32)
                arr *= mask

            initialPoints.append(arr)

        initialPoints = np.array(initialPoints, dtype=np.float32)
        APSO = ParticleSwarm(torch.from_numpy(initialPoints).reshape(-1, modelSize), wrapped_cost, model, w=args.inertiaWeight, c1=args.cognitiveWeight, c2=args.socialWeight)
        
        prob_log = []
        prob_log_denoise = []
        for epoch in trange(epochs + 1, desc="APSO Optimization", unit="epoch"):
            APSO.step()
            probs = log_probabilities(model, APSO.pos_best_g.detach().cpu().float().view(1, 1, 28, 28).to(device), epoch, prob_log)
            perturbed_img = reduce_excess_perturbations(APSO, baseline_img.squeeze(), APSO.pos_best_g.detach().cpu().numpy().squeeze().reshape(28, 28).copy(), single_misclassification_target)
            img_tensor = torch.from_numpy(perturbed_img).float().unsqueeze(0).unsqueeze(0).to(device)
            probs = log_probabilities(model, img_tensor, epoch, prob_log_denoise)

        final_output = model(APSO.pos_best_g.detach().cpu().float().view(1, 1, 28, 28).to(device))
        probs = F.softmax(final_output.squeeze(), dim=0)
        predicted_class = torch.argmax(probs).item()

        success = (predicted_class == single_misclassification_target)
        if success:
            print(f"Attack succeeded. Final predicted class: {predicted_class}")
        else:
            print(f"Attack failed. Final predicted class: {predicted_class}")
            currentParticles = int(currentParticles * args.particleGrowth)
        attempt += 1

    if not success:
        print(f"Attack failed after {args.maxRetries} retries.")

    particle_comparison_analysis(APSO, APSO.pos_best_g.detach().cpu().numpy().squeeze().reshape(28, 28).copy(), baseline_img.squeeze(), single_misclassification_target, args.outputPath)
    reduced_img = reduce_excess_perturbations(APSO, baseline_img.squeeze(), APSO.pos_best_g.detach().cpu().numpy().squeeze().reshape(28, 28).copy(), single_misclassification_target)
    particle_comparison_analysis(APSO, reduced_img.reshape(28, 28).copy(), baseline_img.squeeze(), single_misclassification_target, args.outputPath, denoise=True)

    prob_path = os.path.join(args.outputPath, "epoch_probabilities.csv")
    with open(prob_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + [f"class_{i}" for i in range(10)])
        writer.writerows(prob_log)
    print(f"Saved probability log to {prob_path}")
    prob_path = os.path.join(args.outputPath, "epoch_probabilities_denoise.csv")
    with open(prob_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + [f"class_{i}" for i in range(10)])
        writer.writerows(prob_log_denoise)
    print(f"Saved probability log to {prob_path}")

if __name__ == "__main__":
    main()