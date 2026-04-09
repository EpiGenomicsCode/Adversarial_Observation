#!/usr/bin/env python3
import glob
import librosa
import soundfile as sf
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys, os, json
import argparse
import time
import numpy as np
from torchvision import datasets, transforms
import hashlib
import csv

from PIL import Image

from Adversarial_Observation.Swarm_Observer.Swarm import PSO as ParticleSwarm
from Adversarial_Observation.Adversarial_Observation.utils import seedEverything
from captum.attr import Saliency, IntegratedGradients, DeepLiftShap

from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from captum.attr import Saliency, IntegratedGradients, DeepLiftShap
import torchaudio.transforms as T

from RegNetX import RegNetX_400MF
from MobileNet import MobileNet

# ----------------------------
# Constants
# ----------------------------
SAMPLING_RATE = 16000
NUM_CLASSES = 10
MAX_AUDIO_LENGTH = 16000
BATCH_SIZE = 32

# ----------------------------
# Audio Preprocessing
# ----------------------------
def normalize_audio(x):
    x = np.array(x, dtype=np.float32)
    max_val = np.max(np.abs(x))
    return x / max_val if max_val > 0 else x

def pad_audio(audio, max_len=MAX_AUDIO_LENGTH):
    if len(audio) > max_len:
        return audio[:max_len]
    return np.pad(audio, (0, max_len - len(audio)), 'constant')

# ----------------------------
# Dataset
# ----------------------------
class AudioMNISTDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        self.labels = []

        wav_files = glob.glob(os.path.join(data_path, '*', '*.wav'))
        wav_files = sorted(wav_files, key=lambda x: hashlib.md5(x.encode()).hexdigest())
        for audio_path in tqdm(wav_files, desc="Loading audio files"):
            audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE)
            audio = normalize_audio(audio)
            audio = pad_audio(audio)
            audio = audio.astype(np.float32)[None, :]
            label = int(os.path.basename(audio_path)[0])
            self.data.append(audio)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio = torch.tensor(self.data[idx], dtype=torch.float32)  # (1, L)
        label = self.labels[idx]
        return audio, label

# ----------------------------
# Model Definition (1D convs)
# ----------------------------
class AudioMNISTModel1(nn.Module):
    def __init__(self, input_length=MAX_AUDIO_LENGTH):
        super(AudioMNISTModel1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)

        conv_out_length = self._get_conv_output_length(input_length)
        self.fc1 = nn.Linear(64 * conv_out_length, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def _get_conv_output_length(self, input_length):
        length = (input_length - 5 + 1) // 2
        length = (length - 5 + 1) // 2
        return length

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class AudioMNISTModel2(nn.Module):
    def __init__(self):
        super(AudioMNISTModel2, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        self.drop3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(2)
        self.drop4 = nn.Dropout(0.25)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        self.drop_fc = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool4(F.relu(self.bn4(self.conv4(x)))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        return self.fc2(x)

class AudioModelWrapper(nn.Module):
    def __init__(self, backbone_class, num_classes=10):
        super(AudioModelWrapper, self).__init__()
        self.spectrogram = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        dummy_batch = torch.zeros(1, 1, 64, 32)
        self.backbone = backbone_class(one_batch=dummy_batch, num_classes=num_classes)

    def forward(self, x):
        x = self.spectrogram(x)
        x = torch.log(x + 1e-9)
        return self.backbone(x)


# ----------------------------
# Utility: Dynamic Model Loader
# ----------------------------
def load_model(model_path, arch):
    if arch == "basic":
        model = AudioMNISTModel1()
    elif arch == "adv":
        model = AudioMNISTModel2()
    elif arch == "MobileNet":
        model = AudioModelWrapper(MobileNet)
    elif arch == "RegNetX":
        model = AudioModelWrapper(RegNetX_400MF)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
        
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    return model

# --- Cost function: targeted attack (1D audio) ---
def costFunc(model, input_audio, target_label):
    if isinstance(input_audio, np.ndarray):
        x = torch.from_numpy(input_audio).float()
    elif isinstance(input_audio, torch.Tensor):
        x = input_audio.clone().detach().float()
    else:
        raise TypeError("input_audio must be numpy array or torch tensor")

    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(0)  # (1,1,L)
    elif x.ndim == 2:
        if x.shape[0] == 1:
            x = x.unsqueeze(1)
        else:
            x = x.unsqueeze(0)  
    elif x.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported tensor shape: {x.shape}")

    device = next(model.parameters()).device
    x = x.to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
    return float(out[0, target_label].item())

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run APSO adversarial attack on MNIST test image with SimpleCNN')
    parser.add_argument("--dataPath", type=str, default="./AudioMNIST/data", help="Path to AudioMNIST dataset")
    parser.add_argument('--modelPath', type=str, required=True, help='Path to trained audio model (.pth)')
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
    parser.add_argument('--batchSize', type=int, default=BATCH_SIZE, help='Batch size for data loading')
    return parser.parse_args()

# --- Logging ---
def log_probabilities(model, audio_tensor, epoch, prob_log):
    with torch.no_grad():
        model.eval()
        pred_logits = model(audio_tensor)
        probs = torch.softmax(pred_logits, dim=1).cpu().numpy().flatten()
    prob_log.append([epoch] + probs.tolist())
    return probs

# --- XAI attribution for 1D audio ---
def _to_numpy_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def _plot_1d_and_save(arr1d, path, title=None):
    plt.figure(figsize=(8, 2))
    plt.plot(arr1d, linewidth=0.6)
    plt.xlabel("Sample index")
    plt.ylabel("Attribution")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, format="png", bbox_inches="tight", pad_inches=0.02)
    plt.close()

def run_xai(model, audio_tensor, target_label, apsoswarm, output_path, epoch_padded, name, suffix=""):
    os.makedirs(output_path, exist_ok=True)
    model.eval()

    device = next(model.parameters()).device
    audio_tensor = audio_tensor.to(device)

    audio_tensor = audio_tensor.clone().detach()
    audio_tensor.requires_grad = True

    # --- Saliency ---
    saliency = Saliency(model)
    try:
        attr_sal = saliency.attribute(audio_tensor, target=target_label)  
        attr_sal_np = _to_numpy_cpu(attr_sal).squeeze()  
        if attr_sal_np.ndim == 2:
            attr_sal_np = attr_sal_np[0]
        _plot_1d_and_save(attr_sal_np, f"{output_path}/{epoch_padded}_{name}_saliency{suffix}.png", title="Saliency")
    except Exception as e:
        print("Saliency failed:", e)

    # --- Integrated Gradients ---
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(audio_tensor).to(device)
    try:
        attr_ig = ig.attribute(audio_tensor, baselines=baseline, target=target_label, n_steps=50)
        attr_ig_np = _to_numpy_cpu(attr_ig).squeeze()
        if attr_ig_np.ndim == 2:
            attr_ig_np = attr_ig_np[0]
        _plot_1d_and_save(attr_ig_np, f"{output_path}/{epoch_padded}_{name}_intgrad{suffix}.png", title="Integrated Gradients")
    except Exception as e:
        print("IntegratedGradients failed:", e)

    # --- DeepLiftShap ---
    try:
        bg_list = []
        if hasattr(apsoswarm, "dataset"):
            ds = apsoswarm.dataset
            n_bgs = min(30, len(ds))
            indices = np.random.choice(len(ds), n_bgs, replace=False)
            for i in indices:
                sample, _ = ds[i]  
                if isinstance(sample, torch.Tensor):
                    sample = sample.unsqueeze(0).to(device) if sample.ndim == 2 else sample.to(device)
                bg_list.append(sample.to(device))
        else:
            for _ in range(10):
                noise = torch.randn_like(audio_tensor) * 0.001
                bg_list.append((audio_tensor * 0.0 + noise).to(device))

        if len(bg_list) == 0:
            bg = torch.zeros_like(audio_tensor)
        else:
            bg = torch.cat([b if b.ndim == 3 else b.unsqueeze(0) for b in bg_list], dim=0)

        dl_shap = DeepLiftShap(model)
        attr_shap = dl_shap.attribute(audio_tensor, baselines=bg, target=target_label)
        attr_shap_np = _to_numpy_cpu(attr_shap).squeeze()
        if attr_shap_np.ndim == 2:
            attr_shap_np = attr_shap_np[0]
        _plot_1d_and_save(attr_shap_np, f"{output_path}/{epoch_padded}_{name}_shap{suffix}.png", title="DeepLiftShap")
    except Exception as e:
        print("DeepLiftShap failed:", e)

# --- Main visualization wrapper (audio) ---
def visualize_position(APSO, epoch, name, output_path, target_label, original_audio: np.ndarray = None, denoise: bool = False):
    os.makedirs(output_path, exist_ok=True)
    epoch_padded = f"{epoch:02d}"

    if denoise and original_audio is not None:
        best_vec = reduce_excess_perturbations(APSO, original_audio.copy(), APSO.pos_best_g.detach().cpu().numpy().squeeze().copy(), target_label)
        suffix = "_denoise"
    else:
        best_vec = APSO.pos_best_g.detach().cpu().numpy().copy().squeeze()
        suffix = ""

    plt.figure(figsize=(10, 2))
    plt.plot(best_vec, linewidth=0.6)
    plt.title(f"{name} best particle {suffix}")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(f"{output_path}/{epoch_padded}_{name}_particle_best{suffix}.png", format='png', bbox_inches='tight', pad_inches=0.02)
    plt.close()

    model = APSO.model
    model.eval()
    audio_tensor = torch.tensor(best_vec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
    audio_tensor.requires_grad = True

    run_xai(model, audio_tensor, target_label, APSO, output_path, epoch_padded, name, suffix)

def particle_comparison_analysis(attacker, adv_audio: np.ndarray, original_audio: np.ndarray, single_misclassification_target: int, outputPath: str, denoise: bool = False):
    os.makedirs(outputPath, exist_ok=True)
    suffix = "_denoise" if denoise else ""

    adv = np.array(adv_audio).squeeze().astype(np.float32)
    orig = np.array(original_audio).squeeze().astype(np.float32)

    L = adv.shape[0]

    plt.figure(figsize=(10, 2))
    plt.plot(adv, linewidth=0.6)
    plt.title("Adversarial waveform")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(outputPath, f"best_particle{suffix}.png"), format='png', bbox_inches='tight', pad_inches=0.02)
    plt.close()

    np.savetxt(os.path.join(outputPath, f"best_particle_audio{suffix}.csv"), adv[None, :], delimiter=',')

    diff = adv - orig
    max_abs = float(np.max(np.abs(diff)))
    if max_abs == 0:
        max_abs = 1.0

    plt.figure(figsize=(10, 2))
    plt.plot(diff, linewidth=0.6)
    plt.title("Perturbation (adv - orig)")
    plt.xlabel("Sample index")
    plt.ylabel("Delta amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(outputPath, f"attack-vector_best_particle{suffix}.png"), format='png', bbox_inches='tight', pad_inches=0.02)
    plt.close()

    np.savetxt(os.path.join(outputPath, f"attack-vector_best_particle{suffix}.csv"), diff[None, :], delimiter=',')

    model = attacker.model
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        inp = torch.from_numpy(adv).float().unsqueeze(0).unsqueeze(0).to(device)  
        output = model(inp)
        softmax_output = torch.nn.functional.softmax(output.squeeze(), dim=0)
        confidence_values = softmax_output.detach().cpu().numpy().tolist()
        max_output_value = float(max(confidence_values))
        max_output_class = int(np.argmax(confidence_values))

    stats_file = os.path.join(outputPath, f"best_particle_stats{suffix}.tsv")
    with open(stats_file, 'w') as f:
        f.write("Class\tConfidence\n")
        for i, conf in enumerate(confidence_values):
            f.write(f"{i}\t{conf}\n")
        f.write(f"\nBest Class\t{max_output_class}\n")
        f.write(f"Max Confidence\t{max_output_value}\n")
        f.write(f"Target Class\t{single_misclassification_target}\n")

    print(f"Best Class\t{max_output_class}")
    print(f"Target Class\t{single_misclassification_target}")

def reduce_excess_perturbations(attacker, original_audio, adv_audio, target_label, tol=1e-5, max_iter=20, margin=1e-4):
    if isinstance(original_audio, torch.Tensor):
        original_audio = original_audio.detach().cpu().numpy().squeeze()
    if isinstance(adv_audio, torch.Tensor):
        adv_audio = adv_audio.detach().cpu().numpy().squeeze()

    original_audio = original_audio.copy().astype(np.float32).squeeze()
    adv_audio = adv_audio.copy().astype(np.float32).squeeze()

    device = next(attacker.model.parameters()).device
    L = original_audio.shape[0]

    def confident(audio_arr):
        inp = torch.from_numpy(audio_arr).float().unsqueeze(0).unsqueeze(0).to(device) 
        attacker.model.eval()
        with torch.no_grad():
            output = attacker.model(inp)
            probs = F.softmax(output.squeeze(), dim=0)
        topk = torch.topk(probs, 2)
        top1 = int(torch.argmax(probs).item())
        second_val = float(topk.values[1].item()) if topk.values.size(0) > 1 else 0.0
        return (top1 == target_label and float(probs[target_label].item()) - second_val > margin), probs

    adv = adv_audio.copy()
    changed = True
    while changed:
        changed = False
        for idx in range(L):
            if np.isclose(original_audio[idx], adv[idx], atol=1e-12):
                continue
            orig_val = original_audio[idx]
            adv_val = adv[idx]

            adv[idx] = orig_val
            ok, _ = confident(adv)
            if ok:
                changed = True
                continue

            low, high = orig_val, adv_val
            best_val = adv_val
            for _ in range(max_iter):
                mid = (low + high) / 2.0
                adv[idx] = mid
                ok, _ = confident(adv)
                if ok:
                    best_val = mid
                    high = mid
                    changed = True
                else:
                    low = mid
                if abs(high - low) < tol:
                    break
            adv[idx] = best_val

    return adv

# ----------------------------
# Load Data
# ----------------------------
def load_data(data_path, batch_size):
    dataset = AudioMNISTDataset(data_path)
    train_size = int(0.8 * len(dataset))
    train_indices = list(range(0, train_size))
    test_indices  = list(range(train_size, len(dataset)))

    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    test_dataset  = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# ----------------------------
# Main 
# ----------------------------
def main() -> None:
    args = parse_arguments() 

    if args.randomSeed is not None:
        seedEverything(args.randomSeed)

    train_loader, test_loader = load_data(args.dataPath, BATCH_SIZE)

    modelSize = MAX_AUDIO_LENGTH
    currentParticles = args.particleNum
    epochs = args.epochs
    sparsity = args.sparsity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.modelPath, args.arch)
    model.to(device)
    model.eval()

    def wrapped_cost(model_arg, vec):
        return costFunc(model, vec, args.targetLabel)

    # Note: Added safety bound checking to array access
    adj_index = args.sourceIndex - 24000 if args.sourceIndex >= 24000 else args.sourceIndex
    if adj_index < 0 or adj_index >= len(test_loader.dataset):
        raise ValueError(f"Index {adj_index} out of bounds.")
    baseline_audio, baseline_label = test_loader.dataset[adj_index] 

    baseline_label = int(baseline_label)
    baseline_audio = baseline_audio.squeeze(0)  

    single_misclassification_target = args.targetLabel
    if args.targetLabel < 0 or args.targetLabel > NUM_CLASSES - 1:
        single_misclassification_target = (baseline_label + 1) % NUM_CLASSES
        print(f"Invalid target label detected. Overriding to {single_misclassification_target}")

    assert baseline_label != single_misclassification_target, \
        "Target classes should be different for misclassification."

    print(f"Original class: {baseline_label}")
    print(f"Misclassification target class: {single_misclassification_target}")

    os.makedirs(args.outputPath, exist_ok=True)
    base_audio_np = baseline_audio.numpy()
    plt.figure(figsize=(10, 2))
    plt.plot(base_audio_np, linewidth=0.6)
    plt.title("Original waveform")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outputPath, "original_waveform.png"), format='png', bbox_inches='tight', pad_inches=0.02)
    plt.close()
    sf.write(os.path.join(args.outputPath, "original.wav"), base_audio_np, SAMPLING_RATE)

    with torch.no_grad():
        inp = baseline_audio.unsqueeze(0).unsqueeze(0).to(device)  
        orig_output = model(inp)
        probs = F.softmax(orig_output.squeeze(), dim=0)
        predicted_class = torch.argmax(probs).item()
    print(f"Original prediction: {predicted_class}")

    attempt = 0
    success = False
    while attempt < args.maxRetries and not success:
        print(f"\nAttempt #{attempt + 1} with {currentParticles} particles")

        epsilon = 1.0  
        initialPoints = []
        baseline_np = baseline_audio.detach().cpu().numpy().reshape(1, -1)  

        for _ in range(currentParticles):
            noise = np.random.uniform(-epsilon, epsilon, size=(1, modelSize)).astype(np.float32)
            mask = np.random.choice([0, 1], size=noise.shape, p=[sparsity, 1 - sparsity])
            sparse_noise = noise * mask
            arr = np.clip(baseline_np + sparse_noise, -1.0, 1.0) 
            initialPoints.append(arr)

        initialPoints = torch.tensor(np.array(initialPoints, dtype=np.float32).reshape(-1, modelSize))
        APSO = ParticleSwarm(initialPoints, wrapped_cost, model, w=args.inertiaWeight, c1=args.cognitiveWeight, c2=args.socialWeight, minclamp=-1, maxclamp=1)
        if hasattr(APSO, "__dict__"):
            APSO.dataset = test_loader

        prob_log = []

        for epoch in trange(epochs + 1, desc="APSO Optimization", unit="epoch"):
            APSO.step()
            best_vec = APSO.pos_best_g.detach().cpu().float().squeeze().numpy()  
            best_tensor = torch.from_numpy(best_vec).float().unsqueeze(0).unsqueeze(0).to(device)  
            probs = log_probabilities(model, best_tensor.to(device), epoch, prob_log)

        final_best_vec = APSO.pos_best_g.detach().cpu().float().squeeze().numpy()
        final_best_tensor = torch.from_numpy(final_best_vec).float().unsqueeze(0).unsqueeze(0).to(device)
        final_output = model(final_best_tensor)
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

    best_raw = APSO.pos_best_g.detach().cpu().numpy().squeeze().copy()
    particle_comparison_analysis(APSO, best_raw, baseline_audio.numpy(), args.targetLabel, args.outputPath)
    sf.write(os.path.join(args.outputPath, "best_raw_final.wav"), best_raw, SAMPLING_RATE)

    reduced_audio = reduce_excess_perturbations(APSO, baseline_audio.numpy(), best_raw.copy(), args.targetLabel)
    particle_comparison_analysis(APSO, reduced_audio.copy(), baseline_audio.numpy(), args.targetLabel, args.outputPath, denoise=True)
    sf.write(os.path.join(args.outputPath, "reduced_audio_final.wav"), reduced_audio, SAMPLING_RATE)

    prob_path = os.path.join(args.outputPath, "epoch_probabilities.csv")
    with open(prob_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + [f"class_{i}" for i in range(NUM_CLASSES)])
        writer.writerows(prob_log)
    print(f"Saved probability log to {prob_path}")

if __name__ == "__main__":
    main()