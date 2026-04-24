#!/bin/bash
#SBATCH -A bbse-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60g
#SBATCH --time=8:00:00

SIF=/work/hdd/bbse/wklai/AdversarialData/Adversarial_Observation/manuscripts/POISON26/singularity/pytorch-captum.sif
cd /work/hdd/bbse/wklai/AdversarialData/Adversarial_Observation/manuscripts/POISON26

# ==========================================
# 1. Train MNIST Prediction Models
# ==========================================
echo "Starting MNIST Training..."
mkdir -p models/MNIST

# Basic
singularity exec $SIF python bin/train/MNIST/train_MNIST.py --arch basic --training standard      --output mnist_basic_standard.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST.py --arch basic --training standard --aug --output mnist_basic_standard_aug.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST.py --arch basic --training fgsm          --output mnist_basic_fgsm.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST.py --arch basic --training fgsm     --aug --output mnist_basic_fgsm_aug.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST.py --arch basic --training pgd           --output mnist_basic_pgd.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST.py --arch basic --training pgd      --aug --output mnist_basic_pgd_aug.pt

# Adv
singularity exec $SIF python bin/train/MNIST/train_MNIST.py --arch adv --training standard      --output mnist_adv_standard.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST.py --arch adv --training standard --aug --output mnist_adv_standard_aug.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST.py --arch adv --training fgsm          --output mnist_adv_fgsm.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST.py --arch adv --training fgsm     --aug --output mnist_adv_fgsm_aug.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST.py --arch adv --training pgd           --output mnist_adv_pgd.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST.py --arch adv --training pgd      --aug --output mnist_adv_pgd_aug.pt

# MobileNet
singularity exec $SIF python bin/train/MNIST/train_MNIST_MobileNet.py --training standard      --output mnist_mobilenet_standard.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST_MobileNet.py --training standard --aug --output mnist_mobilenet_standard_aug.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST_MobileNet.py --training fgsm          --output mnist_mobilenet_fgsm.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST_MobileNet.py --training fgsm     --aug --output mnist_mobilenet_fgsm_aug.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST_MobileNet.py --training pgd           --output mnist_mobilenet_pgd.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST_MobileNet.py --training pgd      --aug --output mnist_mobilenet_pgd_aug.pt

# RegNetX
singularity exec $SIF python bin/train/MNIST/train_MNIST_RegNetX.py --training standard      --output mnist_regnetx_standard.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST_RegNetX.py --training standard --aug --output mnist_regnetx_standard_aug.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST_RegNetX.py --training fgsm          --output mnist_regnetx_fgsm.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST_RegNetX.py --training fgsm     --aug --output mnist_regnetx_fgsm_aug.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST_RegNetX.py --training pgd           --output mnist_regnetx_pgd.pt
singularity exec $SIF python bin/train/MNIST/train_MNIST_RegNetX.py --training pgd      --aug --output mnist_regnetx_pgd_aug.pt

mv mnist_*.pt models/MNIST/ 2>/dev/null
exit

# ==========================================
# 2. Train CIFAR-10 Prediction Models (E=20)
# ==========================================
echo "Starting CIFAR-10 Training..."
E=20

# Basic
singularity exec $SIF python bin/train/CIFAR10/basic/model_standard_CIFAR10_basic.py --output cifar10_basic_standard.pt --epochs $E
singularity exec $SIF python bin/train/CIFAR10/basic/model_aug_CIFAR10_basic.py      --output cifar10_basic_standard_aug.pt --epochs $E
singularity exec $SIF python bin/train/CIFAR10/basic/model_fgsm_CIFAR10_basic.py     --output cifar10_basic_fgsm.pt     --epochs $E --adv-train
singularity exec $SIF python bin/train/CIFAR10/basic/model_pgd_CIFAR10_basic.py      --output cifar10_basic_pgd.pt      --epochs $E --adv-train

# MobileNet
singularity exec $SIF python bin/train/CIFAR10/MobileNet/model_standard_CIFAR10_MobileNet.py --output cifar10_mobilenet_standard.pt --epochs $E
singularity exec $SIF python bin/train/CIFAR10/MobileNet/model_aug_CIFAR10_MobileNet.py      --output cifar10_mobilenet_standard_aug.pt --epochs $E
singularity exec $SIF python bin/train/CIFAR10/MobileNet/model_fgsm_CIFAR10_MobileNet.py     --output cifar10_mobilenet_fgsm.pt     --epochs $E --adv-train
singularity exec $SIF python bin/train/CIFAR10/MobileNet/model_pgd_CIFAR10_MobileNet.py      --output cifar10_mobilenet_pgd.pt      --epochs $E --adv-train

# RegNetX
singularity exec $SIF python bin/train/CIFAR10/RegNetX/model_standard_CIFAR10_RegNetX.py --output cifar10_regnetx_standard.pt --epochs $E
singularity exec $SIF python bin/train/CIFAR10/RegNetX/model_aug_CIFAR10_RegNetX.py      --output cifar10_regnetx_standard_aug.pt --epochs $E
singularity exec $SIF python bin/train/CIFAR10/RegNetX/model_fgsm_CIFAR10_RegNetX.py     --output cifar10_regnetx_fgsm.pt     --epochs $E --adv-train
singularity exec $SIF python bin/train/CIFAR10/RegNetX/model_pgd_CIFAR10_RegNetX.py      --output cifar10_regnetx_pgd.pt      --epochs $E --adv-train

mv cifar10_*.pt models/ 2>/dev/null

# ==========================================
# 3. Train AudioMNIST Prediction Models (E=10)
# ==========================================
echo "Starting AudioMNIST Training..."
E=10

# Basic 
singularity exec $SIF python bin/train/AUDIOMNIST/basic/model_standard_AUDIOMNIST_basic.py --output audiomnist_basic_standard.pt --epochs $E
singularity exec $SIF python bin/train/AUDIOMNIST/basic/model_aug_AUDIOMNIST_basic.py      --output audiomnist_basic_standard_aug.pt --epochs $E
singularity exec $SIF python bin/train/AUDIOMNIST/basic/model_fgsm_AUDIOMNIST_basic.py     --output audiomnist_basic_fgsm.pt     --epochs $E --adv-train
singularity exec $SIF python bin/train/AUDIOMNIST/basic/model_pgd_AUDIOMNIST_basic.py      --output audiomnist_basic_pgd.pt      --epochs $E --adv-train

# MobileNet 
singularity exec $SIF python bin/train/AUDIOMNIST/MobileNet/model_standard_AUDIOMNIST_MobileNet.py --output audiomnist_mobilenet_standard.pt --epochs $E
singularity exec $SIF python bin/train/AUDIOMNIST/MobileNet/model_aug_AUDIOMNIST_MobileNet.py      --output audiomnist_mobilenet_standard_aug.pt --epochs $E
singularity exec $SIF python bin/train/AUDIOMNIST/MobileNet/model_fgsm_AUDIOMNIST_MobileNet.py     --output audiomnist_mobilenet_fgsm.pt     --epochs $E --adv-train
singularity exec $SIF python bin/train/AUDIOMNIST/MobileNet/model_pgd_AUDIOMNIST_MobileNet.py      --output audiomnist_mobilenet_pgd.pt      --epochs $E --adv-train

# RegNetX
singularity exec $SIF python bin/train/AUDIOMNIST/RegNetX/model_standard_AUDIOMNIST_RegNetX.py --output audiomnist_regnetx_standard.pt --epochs $E
singularity exec $SIF python bin/train/AUDIOMNIST/RegNetX/model_aug_AUDIOMNIST_RegNetX.py      --output audiomnist_regnetx_standard_aug.pt --epochs $E
singularity exec $SIF python bin/train/AUDIOMNIST/RegNetX/model_fgsm_AUDIOMNIST_RegNetX.py     --output audiomnist_regnetx_fgsm.pt     --epochs $E --adv-train
singularity exec $SIF python bin/train/AUDIOMNIST/RegNetX/model_pgd_AUDIOMNIST_RegNetX.py      --output audiomnist_regnetx_pgd.pt      --epochs $E --adv-train

mv audiomnist_*.pt models/ 2>/dev/null

echo "All training complete!"
