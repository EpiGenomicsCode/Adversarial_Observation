#!/bin/bash

SIF=/storage/group/bfp2/default/wkl2-WillLai/Adversarial_Project/Adversarial_Observation/manuscripts/POISON25/pytorch-captum.sif
WORKINGDIR=/storage/group/bfp2/default/wkl2-WillLai/Adversarial_Project/Adversarial_Observation/manuscripts/POISON25

mkdir -p $WORKINGDIR/MNIST_stats

MERGE=$WORKINGDIR/bin/infer/merge_MNIST-CSV.py
INFER=$WORKINGDIR/bin/infer/infer_MNIST.py

# ==========================================
# CONFIGURATION
# ==========================================
DATASET="MNIST"

# The model that the adversarial examples were originally generated to fool
SOURCE_MODEL="mnist_basic_standard" 

# The list of models you want to test against the SOURCE_MODEL's adversarial examples
TARGET_MODELS=(
    "mnist_basic_standard"
    "mnist_basic_aug"
    "mnist_adv_standard"
    "mnist_adv_fgsm"
    "mnist_adv_pgd"
)

# Paths based on the SOURCE_MODEL
# (Assuming your pipeline groups the attacks into an _ALL folder before merging)
MERGE_INPUT_DIR="$WORKINGDIR/${DATASET}_test_${SOURCE_MODEL}_ALL/"
MERGED_CSV="$WORKINGDIR/${DATASET}_stats/${SOURCE_MODEL}-poison_ALL_best_particle_image_denoise.csv"

# ==========================================
# 1. Merge CSVs
# ==========================================
echo "Merging CSVs for attacks generated against $SOURCE_MODEL..."
python $MERGE $MERGE_INPUT_DIR --output_file $MERGED_CSV


# ==========================================
# 2. Cross-Model Inference Loop
# ==========================================
echo "Starting Inference..."

for TARGET in "${TARGET_MODELS[@]}"; do
    echo "Evaluating $TARGET against $SOURCE_MODEL attacks..."
    
    MODEL_FILE="${TARGET}.pt"
    OUTFILE="$WORKINGDIR/${DATASET}_stats/${SOURCE_MODEL}-Poison_${TARGET}-Performance.tsv"
    
    singularity exec -B $WORKINGDIR/models:/models $SIF python $INFER \
        --model /models/$MODEL_FILE \
        --csv $MERGED_CSV \
        --outfile $OUTFILE

done

echo "Inference scoring complete!"