import torch
import torchvision
import shap
import numpy as np
import matplotlib.pyplot as plt
#  load the data
def loadData():
    return (
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

def getmodel():
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Dropout2d(0.25),
        torch.nn.Flatten(),
        torch.nn.Linear(64 * 14 * 14, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(0.5),
        torch.nn.Linear(128, 10),
        torch.nn.Softmax(dim=1)
    )


def main():
    model = getmodel()
    model.load_state_dict(torch.load('mnist_cnn.pt'))

    train_loader, test_loader = loadData()


    # Select a sample from the test set for explanation
    sample_idx = 0  # Choose the index of the sample to explain
    sample, _ = next(iter(test_loader))
    sample = sample[sample_idx:10]
    sample = torch.tensor(sample).to(torch.float32).reshape(-1,1,28,28)

    # Initialize the SHAP explainer with the trained model
    explainer = shap.DeepExplainer(model, sample)

    # Generate SHAP values for the selected sample
    shap_values = explainer.shap_values(sample)

    # Visualize the SHAP values
    shap.image_plot(shap_values, -sample.detach().numpy())


if __name__ == '__main__':
    main()

