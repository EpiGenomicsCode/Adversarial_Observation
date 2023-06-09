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
epochs = 50
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

        ax[0].imshow(img.reshape(28,28), cmap='gray')
        ax[0].set_title(f"Confidence of {label}: {np.round(model(torch.tensor(img).to(torch.float32))[0][label].item(),3)}")
        ax[0].axis('off')

        ax[1].imshow(act, cmap='jet')
        ax[1].axis('off')
        ax[1].set_title("Activation Map")   


        plt.savefig(f"./APSO/{runName}/{index}/{step}.png", bbox_inches='tight', pad_inches=0)
        plt.clf()
        os.makedirs(f"./APSO/{runName}/{index}/data", exist_ok=True)
        np.save(f"./APSO/{runName}/{index}/data/{step}.npy", img)
        np.save(f"./APSO/{runName}/{index}/data/{step}_act.npy", act)
        plt.close()
       
def runAPSO(points, epochs, model, cost_func, dataDic, umap, runName):
    os.makedirs(f"./APSO/{runName}", exist_ok=True)
    # initialize the swarm
    APSO = SO.Swarm.PSO(torch.tensor(points).reshape(-1,1,1,28,28), cost_func, model, w=.5, c1=.5, c2=.5)
    # run the swarm
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        APSO.step()
        #  plot the swarm 
        positions = [np.clip(i.position_i, 0,1) for i in APSO.swarm]
        plotPSO(positions, epoch, model, runName)

        dataDic["Attack"] = positions
        plotData(dataDic, umap, 'umap of MNIST Data with Attack', f'./APSO/{runName}/umap{epoch}.png')

        # get the best point
        bestPoint = np.clip(APSO.pos_best_g, 0,1)
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(bestPoint.reshape(28,28), cmap='gray')
        ax[0].axis('off')
        ax[0].set_title(f"Confidence of {label}: {np.round(model(torch.tensor(bestPoint).to(torch.float32))[0][label].item(),3)}")
        ax[1].imshow(AO.Attacks.activation_map(torch.tensor(bestPoint.reshape(1,1,28,28)).to(torch.float32), model).reshape(28,28), cmap='jet')
        ax[1].axis('off')
        ax[1].set_title("Activation Map")
        plt.savefig(f"./APSO/{runName}/best_{epoch}.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.clf()
        os.makedirs(f"./APSO/{runName}/bestData", exist_ok=True)
        
        np.save(f"./APSO/{runName}/bestData/{epoch}.npy", bestPoint)
        np.save(f"./APSO/{runName}/bestData/{epoch}_act.npy", AO.Attacks.activation_map(torch.tensor(bestPoint.reshape(1,1,28,28)).to(torch.float32), model).reshape(28,28))

    positions = [i.position_i for i in APSO.swarm]
    return positions

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

    # APSO for one label to another
    dataDic["Attack"] = dataDic[initial][:points]
    initalPoints = dataDic[initial][:points]
    positions = runAPSO(initalPoints, epochs, model, cost_func, dataDic, umap, f"MNIST_{initial}_{label}")
    plot_clusters(positions, model, clusters,f"MNIST_{initial}_{label}")

    # APSO for all labels to another
    #  get the first 10 images of each label
    dataDic["Attack"] = []
    for key in dataDic.keys():
        if key != "Attack" and label != key:
            dataDic["Attack"] += dataDic[key][:points]
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
        plt.imshow(cluster.reshape(28,28), cmap='gray')
        conf = cost_func(model, torch.tensor(cluster.reshape(1,1,28,28)))
        plt.title("Average of Cluster with Confidence: " + str(np.round(conf,5)))
        plt.savefig(f"./APSO_Cluster/{runName}/umap/cluster{cluster_index}.png")
        plt.close()
        plt.clf()

        act = AO.Attacks.activation_map(torch.tensor(cluster.reshape(1,1,28,28)).to(torch.float32), model)
        plt.imshow(act.reshape(28,28), cmap='jet')
        plt.colorbar()
        plt.savefig(f"./APSO_Cluster/{runName}/umap/cluster{cluster_index}_act.png")
        plt.close()
        plt.clf()
        cluster_index += 1
    
if __name__ == '__main__':
    main()