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

# global variables
label = 0
initial = 7
epochs = 10
points = 30

def cost_func(model, x):
    global label
    model.eval()
    x = torch.tensor(x.reshape(1,3,32,32))
    x = x.to(torch.float32)
    pred = model(x)[0][label]
    return pred.item()

def umap_data(dataloader):
    #  get the data from dataloader
    dataReduce = []
    dataDic = {}
    
    for batch_idx, (data, target) in enumerate(dataloader):
        if batch_idx <= 10:
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
    umapLearner.fit(dataReduce.reshape(-1,32*32))
    print("Done Fitting UMAP")

    return umapLearner, dataDic
    
def plotData(data, umap, title, saveName):
    plt.clf()
    global label
    global initial
    for key in data.keys():
        if key in ["Attack", initial, label]:
            #  apply umap transformation
            transformed = umap.transform(np.array(data[key]).reshape(-1,32*32))
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

def plotPSO(points, step, model, filename):
    global label
    os.makedirs("PSO_images", exist_ok=True)
    for index in range(len(points)):
        os.makedirs(f"PSO_images/{index}", exist_ok=True)
        img = points[index]
        plt.imshow(img.transpose(1,2,0)/img.max())
        
        confidence = cost_func(model, img)
        plt.title(f"Confidence of {label}: {np.round(confidence,5)}")
        plt.savefig(f"PSO_images/{index}/{filename}_{step}.png")
        plt.colorbar()
        plt.clf()

        act = AO.Attacks.activation_map(torch.tensor(img.reshape(1,3,32,32)).to(torch.float32), model)
        plt.imshow(act.reshape(3,32,32).transpose(1,2,0)/act.max(), cmap="jet")
        plt.colorbar()
        plt.savefig(f"PSO_images/{index}/{filename}_{step}_act.png")
        plt.clf()

def runAPSO(points, epochs, model, cost_func, dataDic, umap, run):
    APSO = SO.Swarm.PSO(torch.tensor(points).reshape(-1,3,32,32), cost_func, model, w=.2, c1=.5, c2=.5)
    for epoch in range(epochs):
        APSO.step()
        positions = [i.position_i for i in APSO.swarm]

        plotPSO(positions, epoch, model, run)
        dataDic["Attack"] = positions
        plotData(dataDic, umap, 'umap of CIFAR10 Data with Attack', f'./umap_images/{run}_epoch_{epoch}.png')
        #  update the swarm with the new positions

        
        
    # get the best point
    bestPoint = APSO.pos_best_g
    #  normalize the best point between 0 and 1
    bestPoint = bestPoint.reshape(3,32,32).transpose(1,2,0)
  
    plt.imshow(bestPoint)
    conf = cost_func(model, torch.tensor(bestPoint).reshape(1,3,32,32))
    plt.title("Best Point with Confidence: " + str(np.round(conf,5)))
    plt.savefig(f"./umap_images/bestPoint{run}.png")
    plt.clf()

def main():

    global label
    global initial
    global points

    seedEverything()
    
    train_loader, test_loader = load_CIFAR10_data()
    labelNames = train_loader.dataset.classes
    model = build_CIFAR10_Model()
    model.load_state_dict(torch.load('CIFAR10_cnn.pt'))
    
    umap, dataDic = umap_data(train_loader)
    os.makedirs("umap_images", exist_ok=True)
    plotData(dataDic, umap, 'umap of CIFAR10 Data', f'./umap_images/CIFAR10_inital.png')
    dataDic["Attack"] = dataDic[initial][:points]

    initalPoints = dataDic[initial][:points]
    
    initalPoints = np.array(initalPoints)

    runAPSO(initalPoints, epochs, model, cost_func, dataDic, umap, f"CIFAR10_{labelNames[initial]}_{labelNames[label]}")
    
if __name__ == '__main__':
    main()