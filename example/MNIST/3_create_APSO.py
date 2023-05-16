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
label = 3
initial = 0
epochs = 20
points = 20

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
    plt.clf()

def plotPSO(points, step, model):
    global label
    os.makedirs("PSO_images", exist_ok=True)
    for index in range(len(points)):
        os.makedirs(f"PSO_images/{index}", exist_ok=True)
        img = points[index]
        plt.imshow(img.reshape(28,28), cmap='gray')
        confidence = model(torch.tensor(img).to(torch.float32))[0][label].item()
        plt.title(f"Confidence of {label}: {np.round(confidence,3)}")
        plt.savefig(f"PSO_images/{index}/{step}.png")
        plt.colorbar()
        plt.clf()

        act = AO.Attacks.activation_map(torch.tensor(img.reshape(1,1,28,28)).to(torch.float32), model)
        plt.imshow(act.reshape(28,28)/act.max(), cmap='jet')
        plt.colorbar()
        plt.savefig(f"PSO_images/{index}/{step}_act.png")
        plt.clf()

        # save the activation map as numpy array
        np.save(f"PSO_images/{index}/{step}_act.npy", act)

        # save the image as numpy array
        np.save(f"PSO_images/{index}/{step}.npy", img)


def runAPSO(points, epochs, model, cost_func, dataDic, umap, run):
    APSO = SO.Swarm.PSO(torch.tensor(points).reshape(-1,1,1,28,28), cost_func, model, w=.5, c1=.5, c2=.5)
    for epoch in range(epochs):
        APSO.step()
        positions = [i.position_i for i in APSO.swarm]
        plotPSO(positions, epoch, model)
        dataDic["Attack"] = positions
        plotData(dataDic, umap, 'umap of MNIST Data with Attack', f'./umap_images/umap_MNIST_attack_epoch_{epoch}.png')
        # normalize the positions 
        normPos = []
        for pos in positions:
            normPos.append((pos - pos.min()) / (pos.max() - pos.min()))
        for i in range(len(APSO.swarm)):
            APSO.swarm[i].position_i = normPos[i]

    # get the best point
    bestPoint = APSO.pos_best_g
    #  normalize the best point between 0 and 1
    plt.imshow(bestPoint.reshape(28,28), cmap='gray')
    conf = cost_func(model, torch.tensor((bestPoint - bestPoint.min()) / (bestPoint.max() - bestPoint.min())).reshape(1,1,28,28))
    plt.title("Best Point with Confidence: " + str(np.round(conf,3)))
    plt.savefig(f"./umap_images/bestPoint{run}.png")
    plt.clf()

    positions = [i.position_i for i in APSO.swarm]

def main():

    global label
    global initial
    global points

    seedEverything()
    
    train_loader, test_loader = load_MNIST_data()
    model = build_MNIST_Model()
    model.load_state_dict(torch.load('MNIST_cnn.pt'))
    
    umap, dataDic = umap_data(train_loader)
    os.makedirs("umap_images", exist_ok=True)
    plotData(dataDic, umap, 'umap of MNIST Data', f'./umap_images/MNIST_inital.png')
    dataDic["Attack"] = dataDic[initial][:points]

    initalPoints = dataDic[initial][:points]
    positions = runAPSO(initalPoints, epochs, model, cost_func, dataDic, umap, f"MNIST_{initial}_{label}")

    positions = np.array(positions)
    positions = positions.reshape(-1,3*32*32)
    # save the positions
    np.save(f"APSO_Cluster/{initial}_{label}.npy", positions)
    
    # cluster the positions using sklearn 
    kmeans = KMeans(n_clusters=2, random_state=0).fit(positions)
    
    
    os.makedirs("APSO_Cluster", exist_ok=True)
    
    # get the values of the clusters
    cluster1 = []
    cluster2 = []

    for i in range(len(kmeans.labels_)):
        if kmeans.labels_[i] == 0:
            cluster1.append(positions[i])
        else:
            cluster2.append(positions[i])

    # plot average of cluster 1
    cluster1 = np.array(cluster1)
    cluster1 = np.mean(cluster1, axis=0)
    cluster1 = cluster1.reshape(3,32,32)
    cluster1.transpose(1,2,0)/cluster1.max()
    
    plt.imshow(cluster1.transpose(1,2,0)/cluster1.max())
    conf = cost_func(model, torch.tensor(cluster1.reshape(1,3,32,32)))
    plt.title("Average of Cluster 1 with Confidence: " + str(np.round(conf,5)))
    plt.savefig(f"./APSO_Cluster/cluster1.png")
    plt.clf()

    # plot average of cluster 2
    cluster2 = np.array(cluster2)
    cluster2 = np.mean(cluster2, axis=0)
    cluster2 = cluster2.reshape(3,32,32)
    cluster2.transpose(1,2,0)/cluster2.max()
    plt.imshow(cluster2.transpose(1,2,0))
    conf = cost_func(model, torch.tensor(cluster2.reshape(1,3,32,32)))
    plt.title("Average of Cluster 2 with Confidence: " + str(np.round(conf,5)))
    plt.savefig(f"./APSO_Cluster/cluster2.png")
    plt.clf()

    # plot the activation map of cluster 1
    act = AO.Attacks.activation_map(torch.tensor(cluster1.reshape(1,3,32,32)).to(torch.float32), model)
    plt.imshow(act.reshape(3,32,32).transpose(1,2,0)/act.max(), cmap="jet")
    plt.colorbar()
    plt.savefig(f"./APSO_Cluster/cluster1_act.png")
    plt.clf()

    # plot the activation map of cluster 2
    act = AO.Attacks.activation_map(torch.tensor(cluster2.reshape(1,3,32,32)).to(torch.float32), model)
    plt.imshow(act.reshape(3,32,32).transpose(1,2,0)/act.max(), cmap="jet")
    plt.colorbar()
    plt.savefig(f"./APSO_Cluster/cluster2_act.png")
    plt.clf()
    
if __name__ == '__main__':
    main()