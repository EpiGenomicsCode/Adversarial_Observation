from Swarm_Observer import Swarm 
import Adversarial_Observation as AO
import os
import torch
import umap
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# default title size
plt.rcParams['axes.titlesize'] = 18
# default axis size
plt.rcParams['axes.labelsize'] = 20
# default tick size
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
# default legend size
plt.rcParams['legend.fontsize'] = 16
# default figure size
plt.rcParams['figure.figsize'] = [12, 12]
# default dpi
plt.rcParams['figure.dpi'] = 500

label = 3
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
    points = 50
    shape = 28*28
    epochs = 20

    #  get initial 
    initial_points = np.random.rand(points, shape)
    mask = np.random.choice([0, 1], size=initial_points.shape, p=[0.99, 0.01])
    initial_points = initial_points * mask
    initial_points = torch.tensor(initial_points).to(torch.float32)

    #  create the swarm
    runSwarm(initial_points, model, device, umap_model, epochs, otherpoints)

def cost_func(model: torch.nn.Sequential, point: torch.tensor):
    global label
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    point = point.to(device)
    point = point.reshape(1, 1, 28, 28)
    output = model(point)

    grad = np.abs(AO.Attacks.gradient_map(point, model, (1,1,28,28))[0].reshape(-1))
    maxgrad = np.max(grad)
    return (output[0][label].item()) * (np.sum(grad)/(maxgrad*len(grad))) 

def plotSwarm(swarm, umap_model, epoch, otherpoints):

    if epoch > 2 and epoch < 18:
        return
   
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot otherpoints 
    for key in otherpoints.keys():
        if key == label:
            x = [i[0] for i in otherpoints[key]]
            y = [i[1] for i in otherpoints[key]]
            ax.scatter(x, y, label=key)
    # Get all the points
    points = swarm.getPoints()

    # Transform the points
    points = umap_model.transform(points)
    ax.scatter(points[:, 0], points[:, 1], c='black', label='Swarm', marker='x', s=100)                
    ax.legend()
    ax.set_title(f'Epoch: {epoch}')
    plt.tight_layout()
    os.makedirs('./APSO/points', exist_ok=True)
    plt.savefig(f'./APSO/points/epoch_{epoch}.png')
    plt.close()
    plotImages(swarm, epoch)

def plotImages(swarm, epoch):
    points = swarm.getPoints()
    points = points.reshape(-1, 1, 28, 28)
    #  plot the best
    best = swarm.getBest()
    best = best.reshape(28, 28)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(best, cmap='gray')
    ax.axis('off')
    global label
    ax.set_title(f'Confidence of {label}: {round(swarm.model(best.reshape(1,1,28,28).to(torch.float32).to(device))[0][label].item(),2)}')

    grad = AO.Attacks.gradient_map(best.reshape(1,1,28,28), swarm.model, (1,1, 28, 28))[0].reshape(28,28)
    grad = np.abs(grad)  # Make the gradients absolute for better visualization
    grad_normalized = (grad - np.min(grad)) / (np.max(grad) - np.min(grad))  # Normalize gradients to 0-1

    grad_cmap = plt.get_cmap('jet')
    grad_rgba = grad_cmap(grad_normalized)
    grad_rgba[..., 3] = 0.7  # Set alpha channel to control transparency

    ax.imshow(grad_rgba, cmap='jet', interpolation='bilinear')
    fig.colorbar(ax.imshow(grad_rgba, cmap='jet', interpolation='bilinear'), ax=ax, shrink=0.5, orientation='horizontal', pad=0.05)

    os.makedirs(f'./APSO/images/epoch_{epoch}', exist_ok=True)
    plt.savefig(f'./APSO/images/epoch_{epoch}/best.png')
    plt.close()
    # plot the rest
    for idx, point in enumerate(points):
        point = point.reshape(28, 28)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(point, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Confidence of {label}: {round(swarm.model(point.reshape(1,1,28,28).to(torch.float32).to(device))[0][label].item(),3)}')

        grad = AO.Attacks.gradient_map(point.reshape(1,1,28,28), swarm.model, (1,1, 28, 28))[0].reshape(28,28)
        grad = np.abs(grad)  # Make the gradients absolute for better visualization
        grad_normalized = (grad - np.min(grad)) / (np.max(grad) - np.min(grad))  # Normalize gradients to 0-1

        grad_cmap = plt.get_cmap('jet')
        grad_rgba = grad_cmap(grad_normalized)
        grad_rgba[..., 3] = 0.7  # Set alpha channel to control transparency

        ax.imshow(grad_rgba, cmap='jet', interpolation='bilinear')
        # colorbar based on axis resized to be smaller shown horizontally at the bottom close to the image
        fig.colorbar(ax.imshow(grad_rgba, cmap='jet', interpolation='bilinear'), ax=ax, shrink=0.5, orientation='horizontal', pad=0.05)
        plt.tight_layout()
        plt.savefig(f'./APSO/images/epoch_{epoch}/point_{idx}.png')
        plt.close()

def runSwarm(inital_points, model, device, umap_model, epochs, otherpoints):
    APSO = Swarm.PSO(inital_points, cost_func, model, 1, .5, .5)
    plotSwarm(APSO, umap_model, 0, otherpoints)
    for i in tqdm.tqdm(range(1, 1+epochs), desc='Running Swarm', total=epochs):
        APSO.step()
        plotSwarm(APSO, umap_model, i, otherpoints)
    plotSwarm(APSO, umap_model, epochs+1, otherpoints)

    sol = APSO.getPoints()
    sol = sol.reshape(-1, 1, 28, 28)
    particle = []
    particle_costs = []
    for point in sol:
        particle_cost = cost_func(model, point)
        particle_costs.append(particle_cost)
        particle.append(point)

    #  get top 10% of the particles based on cost
    particle_costs = np.array(particle_costs)
    particles = np.array(particle)
    particles = particles[np.argsort(particle_costs)]
    particles = particles[:int(len(particle)*.1)+1]

    particles =  np.mean(np.array([i.detach().cpu().numpy() for i in particles]), axis=0)
    particle = particles.reshape(28, 28)
    fig, ax = plt.subplots(1,2, figsize=(6, 6))
    ax[0].imshow(particle, cmap='gray')
    ax[0].axis('off')
    
    grad = AO.Attacks.gradient_map(particle.reshape(1,1,28,28), model, (1,1, 28, 28))[0].reshape(28,28)
    grad = np.abs(grad)  # Make the gradients absolute for better visualization
    grad_normalized = (grad - np.min(grad)) / (np.max(grad) - np.min(grad))  # Normalize gradients to 0-1
    ax[1].imshow(grad_normalized, cmap='jet', interpolation='bilinear')
    ax[1].axis('off')
    plt.tight_layout()
    os.makedirs('./APSO', exist_ok=True)
    plt.savefig(f'./APSO/avg_final.png')

if __name__ == '__main__':
    main()