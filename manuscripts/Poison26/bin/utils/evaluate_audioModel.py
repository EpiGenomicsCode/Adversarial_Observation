import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import librosa
import torchaudio.transforms as T

import glob
import hashlib
import os, csv
from tqdm import tqdm

from MobileNet import MobileNet
from RegNetX import RegNetX_400MF

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
    max_val = np.max(np.abs(x))
    return x / max_val if max_val > 0 else x

def pad_audio(audio, max_len=MAX_AUDIO_LENGTH):
    return audio[:max_len] if len(audio) > max_len else np.pad(audio, (0, max_len - len(audio)), 'constant')

# ----------------------------
# Dataset
# ----------------------------
class AudioMNISTDataset(Dataset):
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
        audio = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        label = self.labels[idx]
        return audio, label

# ----------------------------
# Model definitions
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
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
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
            sample_rate=SAMPLING_RATE,
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
# Evaluation
# ----------------------------
def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_onehot = np.eye(NUM_CLASSES)[y_true]

    auroc = roc_auc_score(y_true_onehot, y_pred, multi_class="ovr")
    auprc = average_precision_score(y_true_onehot, y_pred)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUROC: {auroc:.4f}")
    print(f"Test AUPRC: {auprc:.4f}")

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate audioMNIST model")
    parser.add_argument("--dataset", type=str, default="./AudioMNIST/data", help="Path to AudioMNIST dataset")
    parser.add_argument("--modelPath", type=str, required=True, help="Path to the trained model (.pt file)")
    parser.add_argument("--arch", type=str, required=True, choices=["basic", "adv", "MobileNet", "RegNetX"], help="Architecture of the trained model")
    parser.add_argument("--batchSize", type=int, default=128, help="Batch size for evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = load_data(args.dataset, args.batchSize)

    # Load trained model weights using specific architecture
    model = load_model(args.modelPath, args.arch)

    print(f"Evaluating {args.dataset} model from {args.modelPath} ({args.arch} architecture)...")
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()