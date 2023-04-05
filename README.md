# Adversarial Observation
---

The Adversarial Observation package contains different codes for Explainable techniques that can be used on neural network models.

## Whats New?
This now exists



## Requirements
---
1. Python (3.0 and above)
2. Docker

## Installation (From pip)
TODO

## Installation (From Source)
---
1. Clone the repository 
    
    ```
    git clone git@gitlab.com:JamilGafur_Work/professional_works/Adversarial_Observation.git

    cd  Adversarial_Objservation
    ```

2. Build Docker environment:
    ```
    docker build -t your_project:latest .
    ```

3. Start environment:
    ```
    sh interactiveStudy.sh your_project
    ```

# License

This project is licensed under the [insert license here]. For more information, see the LICENSE.md file.



# Structure
```
.
├── README.md
├── attack_random_visualize.py
├── attack_targeted_visualize.py
├── generate_FGSM.py
├── generate_saliency_map.py
├── interactiveStudy.sh
├── Dockerfile
├── Adversarial_Observation
│   ├── Attacks.py
│   ├── Swarm_Observer
│   │   ├── BirdParticle.py
│   │   └── Swarm.py
│   ├── utils.py
│   └── visualize.py
├── Create
│   ├── artifacts
│   │   ├── labels.npy
│   │   ├── pca.pkl
│   │   └── transform.npy
│   ├── buildModels
│   │   ├── MNIST_CNN.py
│   └── buildViusals
│       └── MNISTPCA.py
└── saved_models
    └── MNIST_CNN.pt
```

# Examples:
* attack_random_visualize.py: generates random data and attacks the model using the APSO method
* attack_targeted_visualize.py: initializes the swarm to training data and attacks the model using the APSO method
* generate_FGSM.py: Given a single image, generate an adversarial attack using the FGSM
* generate_saliency_map.py: Given a single image, generate a saliency map
* Dockerfile: a dockerfile for this code
* interactiveStudy.sh: starts an interactive docker container (cmd line argument is container name) to run this code


# Adversarial_Observation

* Attacks: Contains different adversarial attacks
* util.py: Contains helper functions
* visualize.py: Contains functions for visualization
* Swarm_Observer/BirdParticle.py: A single particle for the PSO optimizer
* Swarm_Observer/Swarm.py: A Swarm for the APSO


# Create Folders:

* saved_models: contains a pretrained MNIST CNN to use for example scripts
* Create/buildModels.py: Contains code for reproducing the MNIST CNN trained Weights
* Create/buildVisuals.py: Contains code for reproducing the MNIST PCA
* Artifacts/transform.npy: The PCA reduced MNIST training Data
* Artifacts/labels.npy: The labels for the PCA reduced MNIST training Data
* Artifacts/pca.pkl: the pickled file of the MNIST PCA used to generate transform.npy