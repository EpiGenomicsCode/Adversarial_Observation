import argparse
import os
import glob
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import librosa
from sklearn.metrics import roc_auc_score, average_precision_score

import hashlib
import csv

# ----------------------------
# Import the 2D Vision Models
# ----------------------------
from RegNetX import RegNetX_400MF
from MobileNet import MobileNet
# from ConvNetX import ConvNeXt

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
    return x / np.max(np.abs(x))

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
        # Deterministic shuffle using md5 hash of path
        wav_files = sorted(wav_files, key=lambda x: hashlib.md5(x.encode()).hexdigest())
        self.wav_files = wav_files.copy()  # store for TSV

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
# Model Definition (Updated Wrapper)
# ----------------------------
class AudioMNISTModel(nn.Module):
    """
    A wrapper that takes a 1D audio signal, reshapes it into a 2D format, 
    and passes it to any standard 2D Vision Backbone (RegNetX, ConvNeXt, MobileNet).
    """
    def __init__(self, backbone_class, num_classes=NUM_CLASSES):
        super(AudioMNISTModel, self).__init__()
        
        # Reshape dimensions: 16000 = 128 * 125
        # This turns our (B, 1, 16000) audio into a (B, 1, 128, 125) "image" grid
        # so 2D Convolutional Neural Networks can process it natively.
        self.reshape_dims = (1, 128, 125)
        
        # Create a dummy batch to let the backbone calculate its Linear layer dynamically
        dummy_batch = torch.zeros(1, *self.reshape_dims)
        
        # Instantiate the passed model class
        self.backbone = backbone_class(one_batch=dummy_batch, num_classes=num_classes)

    def forward(self, x):
        # x arrives as shape (B, 1, 16000)
        # Reshape 1D audio to 2D
        x = x.view(x.size(0), *self.reshape_dims)
        return self.backbone(x)

# ----------------------------
# Load Data
# ----------------------------
def load_data(data_path, batch_size, split_tsv="split_indices_model1.tsv"):
    dataset = AudioMNISTDataset(data_path)
    # Fixed 80/20 split (after deterministic shuffle)
    train_size = int(0.8 * len(dataset))
    train_indices = list(range(0, train_size))
    test_indices  = list(range(train_size, len(dataset)))

    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    test_dataset  = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Write split info to TSV ---
    with open(split_tsv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["index", "split", "label", "file_path"])
        for idx in train_indices:
            writer.writerow([idx, "train", dataset.labels[idx], dataset.wav_files[idx]])
        for idx in test_indices:
            writer.writerow([idx, "test", dataset.labels[idx], dataset.wav_files[idx]])
    print(f"Saved split information to {split_tsv}")
    
    return train_loader, test_loader


# ----------------------------
# Training loop
# ----------------------------
def train(model, train_loader, device, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
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
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # compute AUROC and AUPRC
    y_true_onehot = np.eye(10)[y_true]
    auroc = roc_auc_score(y_true_onehot, y_pred, multi_class="ovr")
    auprc = average_precision_score(y_true_onehot, y_pred)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test auROC: {auroc:.4f}")
    print(f"Test auPRC: {auprc:.4f}")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="MNIST training code (PyTorch)")
    parser.add_argument("--output", type=str, default="mnist_model1.pt", help="Model output name")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader = load_data(batch_size=args.batch_size)
    
    # Initialize model
    # Pass a dummy batch to configure the MobileNet stem for 1-channel MNIST images
    # and properly calculate the fully-connected layer inputs for 28x28 resolution.
    
    dummy_batch = train_loader.dataset[0][0].unsqueeze(0) 
    model = MobileNet(one_batch=dummy_batch, num_classes=10)


    # Train
    train(model, train_loader, device, epochs=args.epochs)

    # Save model
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

    # Evaluate
    print("Model statistics on test dataset")
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()