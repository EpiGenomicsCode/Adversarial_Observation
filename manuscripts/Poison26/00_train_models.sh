#!/bin/bash
#SBATCH -A bbse-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60g
#SBATCH --time=8:00:00

SIF=/work/hdd/bbse/wklai/AdversarialData/APSO_Poison/manuscripts/POISON25/pytorch-captum.sif

cd /work/hdd/bbse/wklai/AdversarialData/APSO_Poison/manuscripts/POISON25
mkdir -p models

# Train MNIST prediction models
# # Basic 2D CNN
# TRAIN=bin/train/MNIST/train_MNIST_basic-model_standard.py
# singularity exec $SIF python $TRAIN --output MNIST_basic-standard.pt
# 
# # Basic 2D CNN with augementation
# TRAIN=bin/train/MNIST/train_MNIST_basic-model_aug.py
# singularity exec $SIF python $TRAIN --output MNIST_basic-aug.pt
# 
# # Basic 2D CNN with FGSM hardening
# TRAIN=bin/train/MNIST/train_MNIST_basic-model_FGSM.py
# singularity exec $SIF python $TRAIN --output MNIST_basic-FGSM.pt
# 
# # Basic 2D CNN with PGD hardening
# TRAIN=bin/train/MNIST/train_MNIST_basic-model_PGD.py
# singularity exec $SIF python $TRAIN --output MNIST_basic-PGD.pt
# 
# 2D CNN model with additional layers and dropout
# TRAIN=bin/train/MNIST/train_MNIST_adv-model_standard.py 
# singularity exec $SIF python $TRAIN --output MNIST_adv-standard.pt
# 
# # 2D CNN model with additional layers and dropout with augmentation
# TRAIN=bin/train/MNIST/train_MNIST_adv-model_aug.py 
# singularity exec $SIF python $TRAIN --output MNIST_adv-aug.pt
# 
# # 2D CNN model with additional layers and dropout with FGSM hardening
# TRAIN=bin/train/MNIST/train_MNIST_adv-model_FGSM.py
# singularity exec $SIF python $TRAIN --output MNIST_adv-FGSM.pt
# 
# # 2D CNN model with additional layers and dropout with PGD hardening
# TRAIN=bin/train/MNIST/train_MNIST_adv-model_PGD.py
# singularity exec $SIF python $TRAIN --output MNIST_adv-PGD.pt

mv MNIST_*pt models/

# # Train CIFAR10 prediction models
# # Simple 2D CNN
# TRAIN1=bin/train/train_CIFAR10_model1.py
# # 2D CNN model with additional layers and dropout
# TRAIN2=bin/train/train_CIFAR10_model2.py
# # Same architecture as model 2 but with augmented input
# TRAIN3=bin/train/train_CIFAR10_model3.py
# # Same architecture as model 2 but with FGSM hardening
# TRAIN4=bin/train/train_CIFAR10_model4.py
# # Same architecture as model 2 but with PGD hardening
# TRAIN5=bin/train/train_CIFAR10_model5.py
# 
# python $TRAIN1 --output CIFAR10_model1.pt --epochs 20
# python $TRAIN2 --output CIFAR10_model2.pt --epochs 20
# python $TRAIN3 --output CIFAR10_model3.pt --epochs 20
# python $TRAIN4 --output CIFAR10_model4.pt --epochs 20
# python $TRAIN5 --output CIFAR10_model5.pt --epochs 20
# 
# mv CIFAR10_model*pt models/
# 
# TRAIN1=bin/train/train_audioMNIST_model1.py
# # 2D CNN model with additional layers and dropout
# TRAIN2=bin/train/train_audioMNIST_model2.py
# # Same architecture as model 2 but with augmented input
# TRAIN3=bin/train/train_audioMNIST_model3.py
# # Same architecture as model 2 but with FGSM hardening
# TRAIN4=bin/train/train_audioMNIST_model4.py
# # Same architecture as model 2 but with PGD hardening
# TRAIN5=bin/train/train_audioMNIST_model5.py
# 
# singularity exec $SIF python $TRAIN1 --output audioMNIST_model1.pt --epochs 10
# singularity exec $SIF python $TRAIN2 --output audioMNIST_model2.pt --epochs 10
# singularity exec $SIF python $TRAIN3 --output audioMNIST_model3.pt --epochs 10
# singularity exec $SIF python $TRAIN4 --output audioMNIST_model4.pt --epochs 10
# singularity exec $SIF python $TRAIN5 --output audioMNIST_model5.pt --epochs 10
# 
#mv audioMNIST_model* models/
