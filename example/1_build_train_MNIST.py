# build a pytorch model that is trained on MNIST dataset
import torch
import torchvision
import numpy as np
import tqdm
from util import *

dataset = "MNIST"
#  train the model saves data to log file
def trainModel(model, train_loader, optimizer,loss, epoch, filename):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        l = loss(output, target)
        l.backward()
        optimizer.step()
    #  save the loss to a file
    with open(filename, 'a') as f:
        f.write(f'\nTrain Epoch: {epoch} Loss: {l.item():.6f}')
        

#  test the model saves data to log file
def testModel(model, test_loader, filename):
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
    #  save the loss to a file
    with open(filename, 'a') as f:
        f.write(f'\nTest set: Average loss: {test_loss:.4f},')


#  seed everything
def seedEverything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def main():
    #  seed everything
    seedEverything(42)
    #  get data
    if dataset == "MNIST":
        train_loader, test_loader = load_MNIST_data()
    
    if dataset == "CIFAR10":
        train_loader, test_loader = load_CIFAR10_data()
        
    #  build model
    model = buildModel()
    #  train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss()
    
    epochs = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print('Training model on {}'.format(device))
    for epoch in tqdm.tqdm(range(1, epochs + 1)):
        trainModel(model, train_loader, optimizer, loss, epoch, 'log.csv')
        testModel(model, test_loader, 'log.csv')
    #  save model
    torch.save(model.state_dict(), 'cnn.pt')
    
    

if __name__ == '__main__':
    main()


