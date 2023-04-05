from Adversarial_Observation.Swarm_Observer.Swarm import PSO
from Adversarial_Observation.utils import seedEverything, buildCNN
from Adversarial_Observation.visualize import visualizeGIF
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import tqdm 
import pickle
import os
from sklearn.decomposition import PCA

# ==== Global Variables ====
# The value of the column we want to optimize for
startValue: int = 0
endValue: int = 3
#================================================

def costFunc(model, input):
    """
    This function takes a model and a tensor input, reshapes the tensor to be a 28 x 28 image of batch size 1 and passes that through
    the model. Then, it returns the output of the column of interest (endValue) as a numpy array.
    """
    input = torch.reshape(input, (1, 1, 28, 28)).to(torch.float32)
    out = model(input)
    return out.detach().numpy()[0][endValue]


def SwarmPSO(model, inputs, costFunc, epochs):        
    swarm = PSO(inputs, costFunc, model, w=.8, c1=.5, c2=.5)
    pred =  model(torch.tensor(swarm.pos_best_g.unsqueeze(0)).to(torch.float32))
    best = np.argmax(pred.detach().numpy())
          
    for i in tqdm.tqdm(range(epochs)):
        swarm.step()
        pred =  model(torch.tensor(swarm.pos_best_g.unsqueeze(0)).to(torch.float32))
        best = np.argmax(pred.detach().numpy())

    #plot the best position
    plt.imshow(torch.reshape(swarm.pos_best_g, (28, 28)).detach().numpy())
    
          
def SwarmPSOVisualize(model, inputs, costFunc, epochs, dirname, specific=None):
    swarm = PSO(inputs, costFunc, model, w=.8, c1=.5, c2=.5)

    data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]))

    train_data = data.data/255
    train_labels = data.targets

    # reduce the data to 2 dimensions for visualization
    pca = PCA(n_components=50)
    train_data_reduced = pca.fit_transform(train_data.reshape(-1, 28*28))
    train_labels_reduced = train_labels

   
    # visualize the swarm in PCA space
    os.makedirs(f'./artifacts/{dirname}', exist_ok=True)
    positions = np.array([i.position_i.numpy() for i in swarm.swarm])
    visualizeSwarm(positions, train_data_reduced, train_labels_reduced, pca, f'./artifacts/{dirname}/epoch_{0}', specific)

    filenames = [f'./artifacts/{dirname}/epoch_{0}.png']
    for i in tqdm.tqdm(range(epochs)):
        swarm.step()
        # visualize the swarm in PCA space
        positions = np.array([i.position_i.numpy() for i in swarm.swarm])
        visualizeSwarm(positions, train_data_reduced, train_labels_reduced, pca, f'./artifacts/{dirname}/epoch_{i+1}', specific)
        filenames.append(f'./artifacts/{dirname}/epoch_{i+1}.png')
        plotInfo(swarm, model, dirname, i+1)


    # create the gif
    visualizeGIF(filenames, f'./artifacts/{dirname}/swarm.gif')
    
    # export the swarm as a csv
    swarm.save(f'./artifacts/{dirname}/swarm.csv')
    
def plotInfo(swarm, model, dirname, epoch):
    """
    This function takes a swarm and plots the average position of the swarm.
    """
    # plot the average position of the swarm
    average = torch.mean(torch.stack([i.position_i for i in swarm.swarm]), dim=0)
    confidence = model(torch.reshape(average, (1, 1, 28, 28)).to(torch.float32)).detach().numpy()[0][endValue]
    plt.title(f'Average Position of Swarm at Epoch {epoch} with Confidence {confidence}')
    plt.imshow(average.reshape(28, 28).detach().numpy(), cmap='gray')
    plt.colorbar()
    plt.savefig(f'./artifacts/{dirname}/average_{epoch}.png')
    plt.clf()
    plt.close()

    # plot the best position of the swarm
    best = swarm.pos_best_g
    confidence = model(torch.reshape(best, (1, 1, 28, 28)).to(torch.float32)).detach().numpy()[0][endValue]
    plt.title(f'Best Position of Swarm at Epoch {epoch} with Confidence {confidence}')
    plt.imshow(best.reshape(28, 28).detach().numpy(), cmap='gray')
    plt.colorbar()
    plt.savefig(f'./artifacts/{dirname}/best_{epoch}.png')
    plt.clf()
    plt.close()




def visualizeSwarm(positions, stable, stable_lables,  pca, title, specific=None):
    """
    This function takes a swarm and plots it in PCA space.
    """
    # plot the swarm in PCA space
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    # plot the stable data
    if specific is None:
        for i in range(10):
            ax.scatter(stable[stable_lables == i][:,0], stable[stable_lables == i][:,1], c=f'C{i}', alpha=.5, label=i)
    else:
        for i in specific:
            ax.scatter(stable[stable_lables == i][:,0], stable[stable_lables == i][:,1], c=f'C{i}', alpha=.5, label=i)


    
    positions = pca.transform(positions.reshape(-1, 28*28))
    # plot the swarm
    ax.scatter(positions[:,0], positions[:,1], c='black', alpha=.5, marker='x', label='swarm')

    # show the legend
    plt.legend()

    plt.savefig(title+".png")
    plt.clf()
    plt.close()
    

def main():
    seedEverything(44)
    model = buildCNN(10)

    model.load_state_dict(torch.load('./artifacts/mnist_cnn.pt'))
    model.eval()

    points = 1000
    input_shape = (points, 1, 28, 28)
    epochs = 50


    random_inputs = np.random.rand(*input_shape)
    sparcity = .80
    # set sparcity of the inputs to 0
    random_inputs[random_inputs < sparcity] = 0
    # SwarmPSO(model, random_inputs, costFunc, epochs)
    SwarmPSOVisualize(model, random_inputs, costFunc, epochs, "ran_attack_vis")

    # load the train data using torchvision
    train_data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]))

    
    train_labels = train_data.targets
    train_data = train_data.data.numpy()/255
    train_data = train_data.reshape(-1, 1, 28,28)
    
    # get all data with label 5
    train_data = train_data[train_labels == startValue][:]
    train_labels = train_labels[train_labels == startValue]

    # SwarmPSO(model, train_data, costFunc, epochs)
    SwarmPSOVisualize(model, train_data, costFunc, epochs, "5_attack_vis", [startValue, endValue])


    



if __name__ == "__main__":
    main()