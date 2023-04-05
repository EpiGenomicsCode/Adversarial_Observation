# imports
import torch
import torchvision
import numpy as np
import pickle
from sklearn.decomposition import PCA
import Adversarial_Observation
from Adversarial_Observation.utils import seedEverything, buildCNN
import os
import tqdm


def loadData():
    """
    Load the MNIST dataset from torchvision.datasets
    :return: train_loader, test_loader
    """
    return  (
            torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(
                    './data',
                    train=True,
                    download=True,
                    transform=torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            (0.1307,), (0.3081,))
                    ])),
                batch_size=64,
                shuffle=True),

            torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(
                    './data',
                    train=False,
                    download=True,
                    transform=torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            (0.1307,), (0.3081,))
                    ])),
                batch_size=1000,
                shuffle=True)
            )


def train(model, device, train_loader, optimizer, epoch):
    """
    Train the model
    :param model: model to train
    :param device: device to use
    :param train_loader: training data
    :param optimizer: optimizer to use
    :param epoch: epoch number
    :return: None
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
            
def test(model, device, test_loader):
    """
    Test the model
    :param model: model to test
    :param device: device to use
    :param test_loader: test data
    :return: None
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')
    
def reduce_data(train_loader, test_loader):
    # Extract data from the PyTorch data loaders
    train_data = []
    train_labels = []
    for batch in train_loader:
        data, labels = batch
        train_data.append(data.numpy())
        train_labels.append(labels.numpy())
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    
    test_data = []
    test_labels = []
    for batch in test_loader:
        data, labels = batch
        test_data.append(data.numpy())
        test_labels.append(labels.numpy())
    test_data = np.concatenate(test_data, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    # Reshape data to 2D array
    train_data_flat = train_data.reshape(train_data.shape[0], -1)
    test_data_flat = test_data.reshape(test_data.shape[0], -1)

    # Use PCA to reduce the data to 50 features
    pca = PCA(n_components=50)
    pca.fit(train_data_flat)
    train_data_reduced = pca.transform(train_data_flat)
    test_data_reduced = pca.transform(test_data_flat)

    # save the pca and transformed data with labels
    np.save('./artifacts/train_data_reduced.npy', train_data_reduced)
    np.save('./artifacts/train_labels.npy', train_labels)

    np.save('./artifacts/test_data_reduced.npy', test_data_reduced)
    np.save('./artifacts/test_labels.npy', test_labels)

    with open('./artifacts/pca.pkl', 'wb') as f:
        pickle.dump(pca, f)

    
def main():
    # seed everything
    seedEverything(42)

    os.makedirs('./artifacts', exist_ok=True)
    # load the data
    train_loader, test_loader = loadData()

    # reduce the data using sklearn pca to 2 dimensions
    reduce_data(train_loader, test_loader)

    # define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the model
    model = buildCNN(10)
    print(model)
    model.to(device)

    # define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    epochs = 10
    
    # train the model
    for epoch in tqdm.tqdm(range(1, epochs + 1)):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # save the model
    torch.save(model.state_dict(), "./artifacts/mnist_cnn.pt")


if __name__ == '__main__':
    main()
    