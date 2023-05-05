# Adversarial Observation
---
[![Python Testing](https://github.com/EpiGenomicsCode/Adversarial_Observation/actions/workflows/run_tests.yml/badge.svg)](https://github.com/EpiGenomicsCode/Adversarial_Observation/actions/workflows/run_tests.yml)
[![Upload to Github Pages](https://github.com/EpiGenomicsCode/Adversarial_Observation/actions/workflows/build_docs.yml/badge.svg)](https://github.com/EpiGenomicsCode/Adversarial_Observation/actions/workflows/build_docs.yml)


The Adversarial Observation framework is a novel solution that addresses concerns regarding the fairness and social impact of machine learning models, specifically neural networks. The framework provides a user-friendly approach to adversarial testing and explainability, utilizing an adversarial swarm optimizer to increase the ease of explainability for the network. It comprises two intertwined parts focusing on adversarial testing and explainability, utilizing the fast gradient sign method (FGSM) and the Adversarial Particle Swarm Optimization (APSO) algorithm to identify regions where the network is most susceptible to adversarial attacks. The framework has the potential to significantly improve the social impact of neural network models, enhancing their effectiveness and efficiency while improving transparency and trust between stakeholders.


# Features
---

* Adversarial robustness: The Adversarial Observation framework enables users to generate and evaluate adversarial examples in a user-friendly way, which enhances the neural network's robustness against adversarial attacks.
* Optimization algorithm: The framework utilizes the state-of-the-art Adversarial Particle Swarm Optimization (APSO) algorithm to efficiently search for adversarial perturbations in high-dimensional parameter space, improving the effectiveness and efficiency of adversarial testing.
* Explainability: The framework allows for improved interpretability of the neural network's decision-making process by generating saliency maps for the network, providing insights into its internal workings and improving transparency for stakeholders.

# License
---

This project is licensed under the MIT License. For more information, see the [LICENSE.md](./LICENSE.txt) file.

# Installation
---

To install the Adversarial Observation framework, you can follow each of the following:

## Build From Source
---
1. Clone this repository:
2. Run the following command:
    ``` bash
    python setup.py install
    ```


# Usage
---


## Adversarial Testing
---

# Examples
---

## 1_build_train_save_artifacts.py
---

This code defines a deep convolutional neural network (CNN) that is trained to classify handwritten digits using the MNIST dataset. The code uses the PyTorch framework to build and train the model.

The loadData() function loads the MNIST dataset using the torchvision.datasets module and returns two data loaders - one for training data and the other for testing data. The train() function trains the model using the optimizer, loss function, and training data, while the test() function evaluates the model's accuracy on the test data. The reduce_data() function preprocesses the data by reducing its dimensionality using principal component analysis (PCA) and saves the transformed data to disk for later use. Finally, the main() function calls all the other functions in the correct sequence, builds the CNN, trains it, and saves the model parameters to disk.

Overall, this code demonstrates how to build, train, and evaluate a deep learning model for image classification using PyTorch and the MNIST dataset. The code also illustrates how to preprocess the data using PCA and save the transformed data to disk.

## 2_generate_attacks.py
---

This code performs adversarial attacks and visualizations on the MNIST dataset. First, it loads the MNIST dataset using the loadData function, which applies some transformations to the data, and creates data loaders for the training and testing sets. Then, it loads a pre-trained CNN model from a saved state dict using the buildCNN function and the load_state_dict method.

Next, the code applies the Fast Gradient Sign Method (FGSM) adversarial attack on a randomly selected image from the test set with varying levels of epsilon (the attack strength) and saves the perturbed images and their confidence scores in separate folders using the fgsm_attack and saliency_map functions. Finally, it generates a saliency map for the original image and saves it to a separate folder. The code uses various visualization techniques and settings to display the original and perturbed images, confidence scores, and saliency maps using Matplotlib.

## 3_swarm_attack.py
---
This code implements the Particle Swarm Optimization (PSO) algorithm to optimize a column of a pre-trained Convolutional Neural Network (CNN). The SwarmPSO() function initializes the swarm, consisting of multiple particles, with random initial positions. The swarm then updates its position by considering its previous position, its previous best position, and the global best position. The global best position is updated every time a particle finds a better position than the previous global best. In each epoch, the function evaluates the cost function on each particle's position, finds the best position, and updates the global best position. Finally, it visualizes the best position after the last epoch using imshow().

The SwarmPSOVisualize() function is similar to SwarmPSO() but includes visualizations of the swarm in each epoch using visualizeSwarm() and plotInfo() functions. visualizeSwarm() plots the swarm's positions in the 50-dimensional PCA space reduced from the original 784-dimensional space of MNIST images. plotInfo() plots the average and the best position of the swarm, which are saved in the ./artifacts folder along with the GIF of the swarm's movement throughout the epochs. The swarm's data is also saved in a CSV file named swarm.csv for later analysis. 