import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from MobileNet import MobileNet
from RegNetX import RegNetX_400MF

# ----------------------------
# Model definitions (MNIST)
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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)          
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)         
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)  
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.4)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)         
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)         
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)  
        self.bn6 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.4)

        self.conv7 = nn.Conv2d(64, 128, kernel_size=4)        
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

# ----------------------------
# Model definitions (CIFAR10)
# ----------------------------
class CIFARModel1(nn.Module):
    def __init__(self):
        super(CIFARModel1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  
        self.pool1 = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  
        return x

class CIFARModel2(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFARModel2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=0)  
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=0)  
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  
        self.drop1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=0)  
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  
        self.drop2 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=0)  
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  
        self.drop3 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 2 * 2, 128)  
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

# ----------------------------
# 2D Backbone Wrapper
# ----------------------------
class ImageWrapper2D(nn.Module):
    def __init__(self, backbone_class, dummy_batch, num_classes=10):
        super(ImageWrapper2D, self).__init__()
        self.backbone = backbone_class(one_batch=dummy_batch, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

# ----------------------------
# Utility: Dynamic Model Loader
# ----------------------------
def load_model(model_path, dataset, arch):
    if dataset == "MNIST":
        dummy_batch = torch.zeros(1, 1, 28, 28)
        if arch == "basic":
            model = MNISTModel1()
        elif arch == "adv":
            model = MNISTModel2()
        elif arch == "MobileNet":
            model = ImageWrapper2D(MobileNet, dummy_batch)
        elif arch == "RegNetX":
            model = ImageWrapper2D(RegNetX_400MF, dummy_batch)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    elif dataset == "CIFAR10":
        dummy_batch = torch.zeros(1, 3, 32, 32)
        if arch == "basic":
            model = CIFARModel1()
        elif arch == "adv":
            model = CIFARModel2()
        elif arch == "MobileNet":
            model = ImageWrapper2D(MobileNet, dummy_batch)
        elif arch == "RegNetX":
            model = ImageWrapper2D(RegNetX_400MF, dummy_batch)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
            
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    return model

# ----------------------------
# Evaluation function
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
    parser = argparse.ArgumentParser(description="Evaluate MNIST or CIFAR10 model")
    parser.add_argument("--dataset", type=str, choices=["MNIST", "CIFAR10"], required=True, help="Dataset to evaluate on: MNIST or CIFAR10")
    parser.add_argument("--modelPath", type=str, required=True, help="Path to the trained model (.pt file)")
    parser.add_argument("--arch", type=str, required=True, choices=["basic", "adv", "MobileNet", "RegNetX"], help="Architecture of the trained model")
    parser.add_argument("--batchSize", type=int, default=128, help="Batch size for evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "MNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        test_loader = DataLoader(test_set, batch_size=args.batchSize, shuffle=False)

    elif args.dataset == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor()])
        test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        test_loader = DataLoader(test_set, batch_size=args.batchSize, shuffle=False)

    # Load trained model weights using specific architecture
    model = load_model(args.modelPath, args.dataset, args.arch)

    print(f"Evaluating {args.dataset} model from {args.modelPath} ({args.arch} architecture)...")
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()