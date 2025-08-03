# tests/test_model_data.py

import pytest
import torch
from Adversarial_Observation.utils import load_MNIST_model, load_data
from torch.utils.data import DataLoader

def test_model_loading():
    model = load_MNIST_model()

    # Check if the model is a subclass of nn.Module
    assert isinstance(model, torch.nn.Module), "Loaded model is not a valid PyTorch model"

def test_data_loading():
    train_loader, test_loader = load_data(batch_size=32)

    # Check if data loaders are of correct type
    assert isinstance(train_loader, DataLoader), "Train loader is not a DataLoader"
    assert isinstance(test_loader, DataLoader), "Test loader is not a DataLoader"

    # Check if the data batch has the expected shape
    for data, target in train_loader:
        assert data.shape == (32, 1, 28, 28), f"Unexpected data shape: {data.shape}"
        assert target.shape == (32,), f"Unexpected target shape: {target.shape}"
        break  # Only check the first batch

