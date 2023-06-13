import torch
import numpy as np
import torchvision
import Swarm_Observer as SO 
import Adversarial_Observation as AO
import os
from util import *
from umap import UMAP
import matplotlib.pyplot as plt
from util import *
from sklearn.cluster import KMeans

# global variables
label = 7
initial = 1
epochs = 20
points = 50

def cost_func(model, x):
    global label
    model.eval()
    x = torch.tensor(x.reshape(1,1,28,28))
    x = x.to(torch.float32)
    pred = model(x)[0][label]
    return pred.item()

def umap_data(dataloader):
    #  get the data from dataloader
    dataReduce = []
    dataDic = {}
    for batch_idx, (data, target) in enumerate(dataloader):
        if batch_idx > 2:
            break
        for i in range(data.shape[0]):
            dataReduce.append(data[i].numpy())
            if target[i].item() not in dataDic.keys():
                dataDic[target[i].item()] = []
            dataDic[target[i].item()].append(data[i].numpy())

    dataReduce = np.array(dataReduce)
    dataReduce = np.concatenate(dataReduce, axis=0)

    #  flatten the data
    umapLearner = UMAP(n_components=2)
    print("Fitting UMAP")
    umapLearner.fit(dataReduce.reshape(-1,28*28))
    print("Done Fitting UMAP")

    return umapLearner, dataDic
    
def plotData(data, umap, title, saveName):
    plt.clf()
    global label
    global initial
    for key in data.keys():
        if key in ["Attack", initial, label]:
            #  apply umap transformation
            transformed = umap.transform(np.array(data[key]).reshape(-1,28*28))
            #  plt transformed data
            #  if key is attack plot black x 
            if key == "Attack":
                plt.scatter(transformed[:,0], transformed[:,1], label=key, marker='x', c='black')
            else:
                plt.scatter(transformed[:,0], transformed[:,1], label=key)

    plt.legend()
    plt.title(title)
    plt.savefig(saveName)
    plt.close()
    plt.clf()

def plotPSO(points, step, model, runName):
    global label
    
    for index in range(len(points)):
        os.makedirs(f"APSO/{runName}/{index}", exist_ok=True)
        fig, ax = plt.subplots(1,2)
        img = points[index]
        act = AO.Attacks.activation_map(torch.tensor(img.reshape(1,1,28,28)).to(torch.float32), model).reshape(28,28)

        
        ax[0].set_title(f"Confidence of {label}: {np.round(model(torch.tensor(img).to(torch.float32))[0][label].item(),3)}")
        ax[0].imshow(img.reshape(28,28), cmap='gray')
        ax[0].axis('off')

        ax[1].imshow(act, cmap='jet')
        ax[1].axis('off')
        ax[1].set_title("Activation Map")   
        
        #  add colorbar to both plots
        fig.colorbar(ax[0].imshow(img.reshape(28,28), cmap='gray'), ax=ax[0],fraction=0.046, pad=0.04)
        fig.colorbar(ax[1].imshow(act, cmap='jet'), ax=ax[1],fraction=0.046, pad=0.04)

        plt.savefig(f"./APSO/{runName}/{index}/{step}.png", bbox_inches='tight', pad_inches=0)
        plt.clf()
        os.makedirs(f"./APSO/{runName}/{index}/data", exist_ok=True)
        np.save(f"./APSO/{runName}/{index}/data/{step}.npy", img)
        np.save(f"./APSO/{runName}/{index}/data/{step}_act.npy", act)
        plt.close()

def getPositions(APSO):
    positions = []
    for particle in APSO.swarm:
        #  normalize the position to be between 0 and 1
        positions.append(particle.position_i)
    return positions


def plotBest(APSO, model, runName, epoch):
    # get the best point
    bestPoint = APSO.pos_best_g
    bestPoint = bestPoint.reshape(28,28)
    act = AO.Attacks.activation_map(torch.tensor(bestPoint.reshape(1,1,28,28)).to(torch.float32), model).reshape(28,28)
    condidence = np.round(cost_func(model, bestPoint),3)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(bestPoint, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title(f"Confidence of {label}: {condidence}")
    ax[1].imshow(act, cmap='jet')
    ax[1].axis('off')
    ax[1].set_title("Activation Map")
    #  add colorbar to both plots
    fig.colorbar(ax[0].imshow(bestPoint, cmap='gray'), ax=ax[0],fraction=0.046, pad=0.04)
    fig.colorbar(ax[1].imshow(act, cmap='jet'), ax=ax[1],fraction=0.046, pad=0.04)
    plt.savefig(f"./APSO/{runName}/best_{epoch}.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.clf()
    os.makedirs(f"./APSO/{runName}/bestData", exist_ok=True)
    
    np.save(f"./APSO/{runName}/bestData/{epoch}.npy", bestPoint)
    np.save(f"./APSO/{runName}/bestData/{epoch}_act.npy", AO.Attacks.activation_map(torch.tensor(bestPoint.reshape(1,1,28,28)).to(torch.float32), model).reshape(28,28))

def runAPSO(points, epochs, model, cost_func, dataDic, umap, runName):
    os.makedirs(f"./APSO/{runName}", exist_ok=True)
    # initialize the swarm
    APSO = SO.Swarm.PSO(torch.tensor(points).reshape(-1,1,1,28,28), cost_func, model, w=.8, c1=.5, c2=.5)
    positions = getPositions(APSO)
    plotPSO(positions, 0, model, runName)

    # run the swarm
    for epoch in range(1,1+epochs):
        print(f"Epoch: {epoch}")
        APSO.step()
        #  plot the swarm 
        positions = getPositions(APSO)
        plotPSO(positions, epoch, model, runName)
        dataDic["Attack"] = positions
        plotData(dataDic, umap, 'umap of MNIST Data with Attack', f'./APSO/{runName}/umap{epoch}.png')
        plotBest(APSO, model, runName, epoch)
        
    APSO.save_history(f"./APSO/{runName}/APSO_simulation.csv")
    return getPositions(APSO)

def main():
    global label
    global initial
    global points

    seedEverything()
    clusters = 2
    train_loader, test_loader = load_MNIST_data()
    model = build_MNIST_Model()
    model.load_state_dict(torch.load('MNIST_cnn.pt'))
    
    umap, dataDic = umap_data(train_loader)
    


    # APSO for sparce random noise to another
    dataDic["Attack"] = []
    for i in range(points):
        data = np.random.rand(1,28,28)
        #  set 90% of the data to 0
        data[data < .9] = 0
        dataDic["Attack"].append(data)
    initalPoints = dataDic["Attack"]
    positions = runAPSO(initalPoints, epochs, model, cost_func, dataDic, umap, f"MNIST_sparce_noise_{label}")
    plot_clusters(positions, model, clusters, f"MNIST_sparce_noise_{label}")



    # APSO for low random noise to another
    dataDic["Attack"] = []
    for i in range(points):
        dataDic["Attack"].append(np.random.rand(1,28,28)*.001)
    initalPoints = dataDic["Attack"]
    positions = runAPSO(initalPoints, epochs, model, cost_func, dataDic, umap, f"MNIST_low_noise_{label}")
    plot_clusters(positions, model, clusters, f"MNIST_low_noise_{label}")

    # APSO for all labels to another
    #  get the first 10 images of each label
    dataDic["Attack"] = []
    for key in dataDic.keys():
        if key != "Attack" and label != key:
            dataDic["Attack"] += dataDic[key][:points//10]
    initalPoints = dataDic["Attack"]
    positions = runAPSO(initalPoints, epochs, model, cost_func, dataDic, umap, f"MNIST_all_{label}")
    plot_clusters(positions, model, clusters,f"MNIST_all_{label}")

    # APSO for random noise to another
    dataDic["Attack"] = []
    for i in range(points):
        dataDic["Attack"].append(np.random.rand(1,28,28))
    initalPoints = dataDic["Attack"]
    positions = runAPSO(initalPoints, epochs, model, cost_func, dataDic, umap, f"MNIST_noise_{label}")
    plot_clusters(positions, model, clusters, f"MNIST_noise_{label}")

    
    # APSO for one label to another
    dataDic["Attack"] = dataDic[initial][:points]
    initalPoints = dataDic[initial][:points]
    positions = runAPSO(initalPoints, epochs, model, cost_func, dataDic, umap, f"MNIST_{initial}_{label}")
    plot_clusters(positions, model, clusters,f"MNIST_{initial}_{label}")




def plot_clusters(positions, model, clusters, runName):
    positions = np.array(positions)
    
    positions = positions.reshape(-1,1*28*28)
    os.makedirs(f"./APSO_Cluster/{runName}/umap", exist_ok=True)
    # save the positions
    np.save(f"./APSO_Cluster/{runName}/positions.npy", positions)
    
    # cluster the positions using sklearn 
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(positions)
    
    # for cluser in kmeans.cluster_centers_, plot the cluster average and the activation map
    cluster_index = 0
    for cluster in kmeans.cluster_centers_:
        cluster = cluster.reshape(28,28)
        # normalizer between 0 and 1
        cluster = (cluster - cluster.min())/(cluster.max() - cluster.min())
        plt.imshow(cluster, cmap='gray')
        conf = cost_func(model, torch.tensor(cluster.reshape(1,1,28,28)))
        plt.title("Average of Cluster with Confidence: " + str(np.round(conf,5)))
        plt.savefig(f"./APSO_Cluster/{runName}/umap/cluster{cluster_index}.png")
        plt.close()
        plt.clf()

        act = AO.Attacks.activation_map(torch.tensor(cluster.reshape(1,1,28,28)).to(torch.float32), model)
        act = act.reshape(28,28)
        # normalizer between 0 and 1
        act = (act - act.min())/(act.max() - act.min())
        plt.imshow(act, cmap='jet')
        plt.colorbar()
        plt.savefig(f"./APSO_Cluster/{runName}/umap/cluster{cluster_index}_act.png")
        plt.close()
        plt.clf()
        cluster_index += 1
    
if __name__ == '__main__':
    main()