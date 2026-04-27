import argparse
import os
import glob
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

import librosa
import torchaudio.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score

import hashlib
import csv
import random

# --- UPDATED IMPORT ---
from HybridViT import HybridViT

# ----------------------------
# Constants
# ----------------------------
SAMPLING_RATE = 16000
NUM_CLASSES = 10
MAX_AUDIO_LENGTH = 16000

# ----------------------------
# Audio Preprocessing
# ----------------------------
def normalize_audio(x):
    max_val = np.max(np.abs(x))
    return x / max_val if max_val > 0 else x

def pad_audio(audio, max_len=MAX_AUDIO_LENGTH):
    return audio[:max_len] if len(audio) > max_len else np.pad(audio, (0, max_len - len(audio)), 'constant')

# ----------------------------
# Dataset & Wrapper
# ----------------------------
class AudioMNISTBaseDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        self.labels = []

        wav_files = glob.glob(os.path.join(data_path, '*', '*.wav'))
        wav_files = sorted(wav_files, key=lambda x: hashlib.md5(x.encode()).hexdigest())
        self.wav_files = wav_files.copy()

        for audio_path in tqdm(wav_files, desc="Loading audio files"):
            audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE)
            audio = normalize_audio(audio)
            audio = pad_audio(audio)
            label = int(os.path.basename(audio_path)[0])
            self.data.append(audio)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class AudioSubsetWrapper(Dataset):
    def __init__(self, subset, augment=False):
        self.subset = subset
        self.augment = augment

    def __len__(self):
        return len(self.subset)

    def apply_augmentation(self, x):
        if random.random() < 0.5:
            x = np.clip(x + np.random.randn(len(x)) * 0.005, -1.0, 1.0)
        if random.random() < 0.5:
            x = np.roll(x, np.random.randint(-200, 200))
        if random.random() < 0.5:
            x = np.clip(x * np.random.uniform(0.8, 1.2), -1.0, 1.0)
        return x

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.augment:
            x = self.apply_augmentation(x)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        return x, y

def load_data(data_path, batch_size, augment_train=False, split_tsv="split_indices_standard.tsv"):
    base_dataset = AudioMNISTBaseDataset(data_path)
    
    train_size = int(0.8 * len(base_dataset))
    train_indices = list(range(0, train_size))
    test_indices  = list(range(train_size, len(base_dataset)))

    train_dataset = AudioSubsetWrapper(Subset(base_dataset, train_indices), augment=augment_train)
    test_dataset  = AudioSubsetWrapper(Subset(base_dataset, test_indices), augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with open(split_tsv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["index", "split", "label", "file_path"])
        for idx in train_indices:
            writer.writerow([idx, "train", base_dataset.labels[idx], base_dataset.wav_files[idx]])
        for idx in test_indices:
            writer.writerow([idx, "test", base_dataset.labels[idx], base_dataset.wav_files[idx]])
            
    return train_loader, test_loader

# ----------------------------
# Model Definition
# ----------------------------
class AudioMNISTModel(nn.Module):
    def __init__(self, backbone_class, num_classes=NUM_CLASSES):
        super(AudioMNISTModel, self).__init__()
        
        # 1D Audio to 2D Spectrogram
        self.spectrogram = T.MelSpectrogram(
            sample_rate=SAMPLING_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        
        # 16000 samples / 512 hop_length ~= 32 time frames. Shape: (1, 64, 32)
        dummy_batch = torch.zeros(1, 1, 64, 32)
        self.backbone = backbone_class(one_batch=dummy_batch, num_classes=num_classes)

    def forward(self, x):
        # x is raw audio: (Batch, 1, 16000)
        x = self.spectrogram(x)
        x = torch.log(x + 1e-9)  # Log scaling for numerical stability
        return self.backbone(x)

# ----------------------------
# Training loop
# ----------------------------
def train(model, train_loader, device, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        avg_acc = running_correct / total
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {elapsed:.2f}s - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

# ----------------------------
# Evaluation
# ----------------------------
def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_onehot = np.eye(NUM_CLASSES)[y_true]
    
    auroc = roc_auc_score(y_true_onehot, y_pred, multi_class="ovr")
    auprc = average_precision_score(y_true_onehot, y_pred)

    print(f"Test Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | auROC: {auroc:.4f} | auPRC: {auprc:.4f}")

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="AudioMNIST Standard Training")
    parser.add_argument("--data", type=str, default="./data/AudioMNIST", help="Path to dataset")
    parser.add_argument("--output", type=str, default="audiomnist_hybridvit_standard.pt", help="Model output name")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- UPDATED MODEL CALL ---
    model = AudioMNISTModel(backbone_class=HybridViT, num_classes=NUM_CLASSES)

    train_loader, test_loader = load_data(args.data, args.batch_size, augment_train=False)

    train(model, train_loader, device, epochs=args.epochs)
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

    print("Model statistics on test dataset")
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()