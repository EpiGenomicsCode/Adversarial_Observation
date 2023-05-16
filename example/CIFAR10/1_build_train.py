import torch
import torchvision
import numpy as np
import tqdm
from util import *


def trainModel(model, train_loader, optimizer, loss, epoch, filename):
    """
    Trains the model using the provided data loader, optimizer, and loss function for one epoch.
    Saves the training loss to the specified file.

    Args:
        model: The model to be trained.
        train_loader: The data loader for the training data.
        optimizer: The optimizer used for training.
        loss: The loss function.
        epoch: The current epoch number.
        filename: The name of the file to save the training loss.
    """
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        l = loss(output, target)
        l.backward()
        optimizer.step()
    # Save the loss to a file
    with open(filename, 'a') as f:
        f.write(f'\nTrain Epoch: {epoch} Loss: {l.item():.6f}')


def testModel(model, test_loader, filename):
    """
    Evaluates the model using the provided data loader and calculates the test loss and accuracy.
    Saves the test loss to the specified file.

    Args:
        model: The model to be evaluated.
        test_loader: The data loader for the test data.
        filename: The name of the file to save the test loss.
    """
    model.eval()
    test_loss = 0
    correct = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    # Save the loss to a file
    with open(filename, 'a') as f:
        f.write(f'\nTest set: Average loss: {test_loss:.4f},')


def seedEverything(seed):
    """
    Seeds all the random number generators to ensure reproducibility.

    Args:
        seed: The seed value for random number generators.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    # Seed everything
    seedEverything(42)

    train_loader, test_loader = load_CIFAR10_data()
    model = build_CIFAR10_Model()
        
    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss()

    epochs = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print('Training model on {}'.format(device))
    for epoch in tqdm.tqdm(range(1, epochs + 1)):
        trainModel(model, train_loader, optimizer, loss, epoch, 'log.csv')
        testModel(model, test_loader, 'log.csv')

    # Save model
    torch.save(model.state_dict(), 'CIFAR10_cnn.pt')



if __name__ == '__main__':
    main()
