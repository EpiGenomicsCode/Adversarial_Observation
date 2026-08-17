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
from pathlib import Path

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
class CIFARModel1(nn.Module):
    def __init__(self):
        super(CIFARModel1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 3-channel input
        self.pool1 = nn.MaxPool2d(2, 2)  # 32→16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16→8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # flatten size = 64*8*8
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

class CIFARModel2(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFARModel2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=0)  # (32x32 → 30x30)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=0)  # (30x30 → 28x28)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # (28x28 → 14x14)
        self.drop1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # same padding
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=0)  # (14x14 → 12x12)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # (12x12 → 6x6)
        self.drop2 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # same padding
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=0)  # (6x6 → 4x4)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # (4x4 → 2x2)
        self.drop3 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 2 * 2, 128)  # flatten from 128 channels × 2×2
        self.drop4 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.drop3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop4(x)
        x = self.fc2(x)
        return x

class CIFARWrapper2D(nn.Module):
    def __init__(self, backbone_class, num_classes=10):
        super(CIFARWrapper2D, self).__init__()
        dummy_batch = torch.zeros(1, 3, 32, 32)
        self.backbone = backbone_class(one_batch=dummy_batch, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

# ----------------------------
# Utility: Dynamic Model Loader
# ----------------------------
def load_model(model_path, arch):
    if arch == "basic":
        model = CIFARModel1()
    elif arch == "adv":
        model = CIFARModel2()
    elif arch == "MobileNet":
        model = CIFARWrapper2D(MobileNet)
    elif arch == "RegNetX":
        model = CIFARWrapper2D(RegNetX_400MF)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    # Only wrapped architectures expect the backbone. prefix
    if arch in ["MobileNet", "RegNetX"]:
        if not any(k.startswith("backbone.") for k in state_dict.keys()):
            state_dict = {
                f"backbone.{k}": v
                for k, v in state_dict.items()
            }

    model.load_state_dict(state_dict)

    return model

# --- Cost function: targeted attack ---
def costFunc(model, input_img, target_label):
    img_tensor = input_img.clone().detach() if isinstance(input_img, torch.Tensor) else torch.from_numpy(input_img).float()
    if img_tensor.ndim == 1:
        img_tensor = img_tensor.view(1, 3, 32, 32)
    elif img_tensor.ndim == 3 and img_tensor.shape[0] == 3:
        img_tensor = img_tensor.unsqueeze(0)
    elif img_tensor.ndim == 4:
        pass
    img_tensor = img_tensor.to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        out = model(img_tensor)
    return float(out[0, target_label].item())

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run APSO adversarial attack on CIFAR10 test image')
    parser.add_argument('--modelPath', type=str, required=True, help='Path to trained CIFAR model (.pth)')
    parser.add_argument('--arch', type=str, required=True, choices=['basic', 'adv', 'MobileNet', 'RegNetX'], help='Architecture of the trained model')
    parser.add_argument('--outputPath', type=str, default="./output/", help='Output directory')
    parser.add_argument('--particleNum', type=int, default=100, help='Number of particles in swarm')
    parser.add_argument('--epochs', type=int, default=50, help='Number of APSO epochs')
    parser.add_argument('--sparsity', type=float, default=0.75, help='Sparsity of initialization vectors (0-1)')
    parser.add_argument('--inertiaWeight', type=float, default=0.8, help='Inertia weight for APSO')
    parser.add_argument('--cognitiveWeight', type=float, default=1.0, help='Cognitive weight for APSO')
    parser.add_argument('--socialWeight', type=float, default=1.0, help='Social weight for APSO')
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
def _to_numpy_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def _plot_and_save(arr2d, path, cmap="seismic", vmin=None, vmax=None):
    plt.figure(figsize=(3, 3))
    plt.imshow(arr2d, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.savefig(path, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()

def run_xai(model, img_tensor, target_label, apsoswarm, output_path, epoch_padded, name, suffix=""):
    os.makedirs(output_path, exist_ok=True)
    model.eval()

    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    # --- Saliency ---
    saliency = Saliency(model)
    try:
        attr_sal = saliency.attribute(img_tensor, target=target_label)
        attr_sal_np = _to_numpy_cpu(attr_sal).squeeze()  # (3,32,32)
        attr_sal_mean = np.mean(np.abs(attr_sal_np), axis=0)
        _plot_and_save(attr_sal_mean, f"{output_path}/{epoch_padded}_{name}_saliency{suffix}.png", cmap="seismic")
    except Exception as e:
        print("Saliency failed:", e)

    # --- Integrated Gradients ---
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(img_tensor).to(device)
    try:
        attr_ig = ig.attribute(img_tensor, baselines=baseline, target=target_label, n_steps=50)
        attr_ig_np = _to_numpy_cpu(attr_ig).squeeze()
        attr_ig_mean = np.mean(np.abs(attr_ig_np), axis=0)
        _plot_and_save(attr_ig_mean, f"{output_path}/{epoch_padded}_{name}_intgrad{suffix}.png", cmap="seismic")
    except Exception as e:
        print("IntegratedGradients failed:", e)

    # --- DeepLiftShap ---
    try:
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        bg_indices = np.random.choice(len(test_dataset), min(50, len(test_dataset)), replace=False)
        bg_imgs = torch.stack([test_dataset[i][0] for i in bg_indices]).to(device)
        dl_shap = DeepLiftShap(model)
        attr_shap = dl_shap.attribute(img_tensor, baselines=bg_imgs, target=target_label)
        attr_shap_np = _to_numpy_cpu(attr_shap).squeeze()
        attr_shap_mean = np.mean(np.abs(attr_shap_np), axis=0)
        _plot_and_save(attr_shap_mean, f"{output_path}/{epoch_padded}_{name}_shap{suffix}.png", cmap="seismic")
    except Exception as e:
        print("DeepLiftShap failed:", e)

# --- Main visualization wrapper ---
def visualize_position(APSO, epoch, name, output_path, target_label, original_img: np.ndarray = None, denoise: bool = False):
    os.makedirs(output_path, exist_ok=True)
    epoch_padded = f"{epoch:02d}"

    if denoise and original_img is not None:
        best_particle_vector = reduce_excess_perturbations(APSO, original_img.copy(), APSO.pos_best_g.detach().cpu().numpy().squeeze().reshape(3, 32, 32).copy(), target_label)
        suffix = "_denoise"
    else:
        best_particle_vector = APSO.pos_best_g.detach().cpu().numpy().copy().reshape(3, 32, 32)
        suffix = ""

    img_3x32x32 = best_particle_vector
    img_HxWxC = np.transpose(img_3x32x32, (1, 2, 0))
    img_HxWxC = np.clip(img_HxWxC, 0, 1)
    
    plt.figure(figsize=(3, 3))
    plt.imshow(img_HxWxC)
    plt.axis('off')
    plt.savefig(f"{output_path}/{epoch_padded}_{name}_particle_best{suffix}.png", format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

    model = APSO.model
    model.eval()
    img_tensor = torch.tensor(img_3x32x32, dtype=torch.float32).unsqueeze(0)  # (1,3,32,32)
    img_tensor.requires_grad = True

    run_xai(model, img_tensor, target_label, APSO, output_path, epoch_padded, name, suffix)

def particle_comparison_analysis(attacker, adv_img: np.ndarray, original_img: np.ndarray, single_misclassification_target: int, outputPath: str, denoise: bool = False):
    os.makedirs(outputPath, exist_ok=True)
    suffix = "_denoise" if denoise else ""

    def to_chw(arr):
        a = np.array(arr)
        if a.shape == (32, 32, 3):
            return np.transpose(a, (2, 0, 1))
        return a

    adv_img_np = to_chw(adv_img)
    if adv_img_np.shape != (3, 32, 32):
        adv_img_np = adv_img_np.reshape(3, 32, 32)
    original_img_np = to_chw(original_img)
    if original_img_np.shape != (3, 32, 32):
        original_img_np = original_img_np.reshape(3, 32, 32)

    img_rgb = np.transpose(adv_img_np, (1, 2, 0))
    img_rgb = np.clip(img_rgb, 0, 1)
    plt.figure(figsize=(3, 3))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.savefig(f"{outputPath}/best_particle{suffix}.png", format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    np.savetxt(f"{outputPath}/best_particle_image{suffix}.csv", adv_img_np.flatten()[None, :], delimiter=',')

    diff_chw = adv_img_np - original_img_np 
    max_abs = float(np.max(np.abs(diff_chw)))
    if max_abs == 0:
        max_abs = 1.0  
    
    diff_hwc = np.transpose(diff_chw, (1, 2, 0))
    vis_rgb = 0.5 + (diff_hwc / (2 * max_abs))
    vis_rgb = np.clip(vis_rgb, 0.0, 1.0)
    
    plt.figure(figsize=(3, 3), dpi=100)
    plt.imshow(vis_rgb)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f"{outputPath}/attack-vector_best_particle{suffix}.png", format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    np.savetxt(f"{outputPath}/attack-vector_best_particle{suffix}.csv", diff_chw.flatten()[None, :], delimiter=',')

    model = attacker.model
    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(adv_img_np).float().unsqueeze(0).to(next(model.parameters()).device)
        output = model(inp)
        softmax_output = torch.nn.functional.softmax(output.squeeze(), dim=0)
        confidence_values = softmax_output.detach().cpu().numpy().tolist()
        max_output_value = float(max(confidence_values))
        max_output_class = int(np.argmax(confidence_values))

    with open(f"{outputPath}/best_particle_stats{suffix}.tsv", 'w') as f:
        f.write("Class\tConfidence\n")
        for i, conf in enumerate(confidence_values):
            f.write(f"{i}\t{conf}\n")
        f.write(f"\nBest Class\t{max_output_class}\n")
        f.write(f"Max Confidence\t{max_output_value}\n")
        f.write(f"Target Class\t{single_misclassification_target}\n")

    print(f"Best Class\t{max_output_class}")
    print(f"Target Class\t{single_misclassification_target}")

def reduce_excess_perturbations(attacker, original_img, adv_img, target_label, tol=1e-3, max_iter=10, margin=0.0001, max_passes=10, clip_min=0.0, clip_max=1.0):
    if isinstance(original_img, torch.Tensor):
        original_img = original_img.detach().cpu().numpy()
    if isinstance(adv_img, torch.Tensor):
        adv_img = adv_img.detach().cpu().numpy()

    if original_img.ndim == 3 and original_img.shape[2] == 3:
        original_img = np.transpose(original_img, (2, 0, 1))
    if adv_img.ndim == 3 and adv_img.shape[2] == 3:
        adv_img = np.transpose(adv_img, (2, 0, 1))

    adv_img = adv_img.copy()
    original_img = original_img.copy()
    device = next(attacker.model.parameters()).device

    def confident(adv_img_arr):
        inp = torch.from_numpy(adv_img_arr).float().unsqueeze(0).to(device)
        with torch.no_grad():
            output = attacker.model(inp)
            probs = F.softmax(output.squeeze(), dim=0)
        topk = torch.topk(probs, 2)
        top1 = int(torch.argmax(probs).item())
        second_val = float(topk.values[1].item()) if topk.values.size(0) > 1 else 0.0
        return (top1 == target_label and float(probs[target_label].item()) - second_val > margin), probs

    for _ in range(max_passes):
        any_change = False
        diff = np.abs(adv_img - original_img)
        flat_order = np.argsort(-diff, axis=None)

        for idx in flat_order:
            c, i, j = np.unravel_index(idx, adv_img.shape)

            orig_val = original_img[c, i, j]
            adv_val = adv_img[c, i, j]

            if abs(orig_val - adv_val) < tol:
                continue

            adv_img[c, i, j] = orig_val
            ok, _ = confident(adv_img)
            if ok:
                any_change = True
                continue

            adv_img[c, i, j] = adv_val
            delta = adv_val - orig_val

            low, high = 0.0, 1.0
            best_frac = 1.0
            for _ in range(max_iter):
                mid = 0.5 * (low + high)
                candidate = np.clip(orig_val + mid * delta, clip_min, clip_max)
                adv_img[c, i, j] = candidate
                ok, _ = confident(adv_img)
                if ok:
                    best_frac = mid
                    high = mid
                    any_change = True
                else:
                    low = mid
                if high - low < tol:
                    break

            adv_img[c, i, j] = np.clip(orig_val + best_frac * delta, clip_min, clip_max)

        if not any_change:
            break

    return adv_img

def main() -> None:
    args = parse_arguments()

    if args.randomSeed is not None:
        seedEverything(args.randomSeed)

    transform = transforms.Compose([transforms.ToTensor()])

    # Test if CIFAR10 is locally available, download if not
    cifar_dir = Path("./data/cifar-10-batches-py")
    download = not cifar_dir.exists()
    print(download)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=download)
#    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    modelSize = 3 * 32 * 32
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
        raise ValueError(f"Image index {args.sourceIndex} is out of bounds.")

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
    base_rgb = np.transpose(baseline_img.numpy(), (1, 2, 0))
    plt.figure(figsize=(3, 3))
    plt.imshow(np.clip(base_rgb, 0, 1))
    plt.axis('off')
    plt.savefig(os.path.join(args.outputPath, "original.png"), format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

    orig_output = model(baseline_img.unsqueeze(0).to(device))
    probs = F.softmax(orig_output.squeeze(), dim=0)
    predicted_class = torch.argmax(probs).item()
    print(f"Original prediction: {predicted_class}")

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
        APSO = ParticleSwarm(torch.from_numpy(initialPoints), wrapped_cost, model, w=args.inertiaWeight, c1=args.cognitiveWeight, c2=args.socialWeight)

        prob_log = []
        prob_log_denoise = []

        for epoch in trange(epochs + 1, desc="APSO Optimization", unit="epoch"):
            APSO.step()

            best_tensor = APSO.pos_best_g.detach().cpu().float().view(1, 3, 32, 32)
            probs = log_probabilities(model, best_tensor.to(device), epoch, prob_log)

            perturbed_img = reduce_excess_perturbations(APSO, baseline_img.numpy(), APSO.pos_best_g.detach().cpu().numpy().squeeze().reshape(3, 32, 32).copy(), single_misclassification_target)
            img_tensor = torch.from_numpy(perturbed_img).float().unsqueeze(0).to(device)
            probs = log_probabilities(model, img_tensor, epoch, prob_log_denoise)

        final_best = APSO.pos_best_g.detach().cpu().float().view(1, 3, 32, 32).to(device)
        final_output = model(final_best)
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

    best_raw = APSO.pos_best_g.detach().cpu().numpy().squeeze().reshape(3, 32, 32).copy()
    particle_comparison_analysis(APSO, best_raw, baseline_img.numpy(), single_misclassification_target, args.outputPath)
    reduced_img = reduce_excess_perturbations(APSO, baseline_img.numpy(), best_raw.copy(), single_misclassification_target)
    particle_comparison_analysis(APSO, reduced_img.copy(), baseline_img.numpy(), single_misclassification_target, args.outputPath, denoise=True)

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
