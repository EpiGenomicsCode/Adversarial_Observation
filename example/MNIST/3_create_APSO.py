from Swarm_Observer import Swarm 
import Adversarial_Observation as AO
import os
import torch
import umap
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


def main():
    #  load the model
    model = AO.utils.load_MNIST_model()
    #  load the data
    train_loader, test_loader = AO.utils.load_MNIST_data()
    #  update weights 
    if os.path.isfile('MNIST_cnn.pt'):
        model.load_state_dict(torch.load('MNIST_cnn.pt'))
    else:
        raise Exception('MNIST_cnn.pt not found. Please run 1_build_train.py first')
    
    #  set the model to evaluation mode
    model.eval()


    otherpoints  = {}
    umap_model = umap.UMAP()

    accumulated_data = []  # Accumulate data samples
    targets = []  # Accumulate targets

    for idx, (data, target) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), desc='Training UMAP'):
        for img, label in zip(data, target):
            img = img.reshape(1, 28*28)
            img = img.detach().numpy()
            accumulated_data.append(img)  # Accumulate data samples
            targets.append(label)  # Accumulate targets
            
    # Convert accumulated data to a single NumPy array
    accumulated_data = np.concatenate(accumulated_data, axis=0)

    # Apply UMAP to the accumulated data
    reduced = umap_model.fit_transform(accumulated_data)

    # Store the reduced data points
    for idx, label in tqdm.tqdm(enumerate(targets), total=len(targets), desc='Storing reduced data'):
        if label.item() not in otherpoints.keys():
            otherpoints[label.item()] = []
        otherpoints[label.item()].append(reduced[idx])

    #  get the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # parameters 
    points = 10
    shape = 28*28
    epochs = 200

    #  get initial 
    initial_points = np.random.rand(points, shape)
    mask = np.random.choice([0, 1], size=initial_points.shape, p=[0.99, 0.01])
    initial_points = initial_points * mask
    initial_points = torch.tensor(initial_points).to(torch.float32)

    #  create the swarm
    runSwarm(initial_points, model, device, umap_model, epochs, otherpoints)

label = 5
def cost_func(model: torch.nn.Sequential, point: torch.tensor):
    global label
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    point = point.to(device)
    point = point.reshape(1, 1, 28, 28)
    output = model(point)

    grad = np.abs(AO.Attacks.gradient_map(point, model, (1,1,28,28))[0].reshape(-1))
    maxgrad = np.max(grad)
    return output[0][label].item() + np.sum(grad)/(maxgrad*len(grad))

def plotSwarm(swarm, umap_model, epoch, otherpoints):
    # Plot the points
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot otherpoints 
    for key in otherpoints.keys():
        x = [i[0] for i in otherpoints[key]]
        y = [i[1] for i in otherpoints[key]]
        ax.scatter(x, y, label=key)
    # Get all the points
    points = swarm.getPoints()

    # Transform the points
    points = umap_model.transform(points)
    ax.scatter(points[:, 0], points[:, 1], c='black', label='Swarm')                
    ax.legend()
    ax.set_title(f'Epoch: {epoch}')
    os.makedirs('./APSO/points', exist_ok=True)
    plt.savefig(f'./APSO/points/epoch_{epoch}.png')
    plt.close()

    plotImages(swarm, epoch)

def plotImages(swarm, epoch):
    points = swarm.getPoints()
    points = points.reshape(-1, 1, 28, 28)
    best = swarm.getBest()
    best = best.reshape(28, 28)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 12))
    axs[0].imshow(best, cmap='gray')
    axs[0].axis('off')
    global label
    axs[0].set_title(f'Confidence of {label}: {swarm.model(best.reshape(1,1,28,28).to(torch.float32).to(device))[0][label].item()}')
    grad = AO.Attacks.gradient_map(best.reshape(1,1,28,28), swarm.model, (1,1, 28, 28))[0].reshape(28,28)
    axs[1].imshow(grad, cmap='jet')
    axs[1].axis('off')
    axs[1].set_title(f'Gradient Map')   
    os.makedirs(f'./APSO/images/epoch_{epoch}', exist_ok=True)
    plt.savefig(f'./APSO/images/epoch_{epoch}/best.png')
    plt.close()
    for idx, point in enumerate(points):
        fig, axs = plt.subplots(1, 2, figsize=(12, 12))
        axs[0].imshow(point[0], cmap='gray')
        axs[0].axis('off')
        axs[0].set_title(f'Confidence of {label}: {swarm.model(best.reshape(1,1,28,28).to(torch.float32).to(device))[0][label].item()}')
        # plot gradient map
        gradient = AO.Attacks.gradient_map(point[0].reshape(1,1,28,28), swarm.model, (1,1, 28, 28))[0].reshape(28,28)
        axs[1].imshow(gradient, cmap='jet')
        axs[1].axis('off')
        axs[1].set_title(f'Gradient Map')
        # save point and gradient as npy
        np.save(f'./APSO/images/epoch_{epoch}/point_{idx}_grad.npy', gradient)
        np.save(f'./APSO/images/epoch_{epoch}/point_{idx}.npy', point[0])
        plt.tight_layout()
        plt.savefig(f'./APSO/images/epoch_{epoch}/point_{idx}.png')
        plt.close()

def runSwarm(inital_points, model, device, umap_model, epochs, otherpoints):
    APSO = Swarm.PSO(inital_points, cost_func, model)
    plotSwarm(APSO, umap_model, 0, otherpoints)
    for i in tqdm.tqdm(range(1, 1+epochs), desc='Running Swarm', total=epochs):
        APSO.step()
        plotSwarm(APSO, umap_model, i, otherpoints)
    plotSwarm(APSO, umap_model, epochs+1, otherpoints)

if __name__ == '__main__':
    main()