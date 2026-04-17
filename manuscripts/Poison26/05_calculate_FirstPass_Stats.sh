#!/bin/bash

SIF=/storage/group/bfp2/default/wkl2-WillLai/Adversarial_Project/Adversarial_Observation/manuscripts/POISON25/pytorch-captum.sif
WORKINGDIR=/storage/group/bfp2/default/wkl2-WillLai/Adversarial_Project/Adversarial_Observation/manuscripts/POISON25

# ==========================================
# CONFIGURATION
# ==========================================
DATASET="MNIST" # Change to CIFAR10 or audioMNIST as needed
LABELS="labels/${DATASET}_test_labels-misclassify.tsv"

mkdir -p $WORKINGDIR/${DATASET}_stats

EVAL=$WORKINGDIR/bin/eval/evaluate_poisoning.py
PARSE=$WORKINGDIR/bin/eval/convert_results.py

# List all the models you attacked in script 02/03/04 that you want to evaluate
MODELS=(
    "mnist_basic_standard"
    "mnist_basic_aug"
    "mnist_adv_fgsm"
    "mnist_mobilenet_pgd"
)

# ==========================================
# EVALUATION LOOP
# ==========================================
for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo "Processing First-Pass Stats for: $MODEL"
    echo "========================================"
    
    MAIN_FOLDER="$WORKINGDIR/${DATASET}_test_${MODEL}"
    OUTPUT_PREFIX="$WORKINGDIR/${DATASET}_stats/${DATASET}_stats-${MODEL}"
    
    # 1. Evaluate Poisoning Success
    singularity exec $SIF python $EVAL \
        --main_folder $MAIN_FOLDER \
        --labels_file $LABELS \
        --output_prefix $OUTPUT_PREFIX
        
    # 2. Parse Failed/Resilient labels
    singularity exec $SIF python $PARSE \
        --input_file ${OUTPUT_PREFIX}_fail.tsv \
        --output_file $WORKINGDIR/${DATASET}_stats/${DATASET}_test_labels_${MODEL}-misclassify_RESILIENT.tsv

done

echo "First pass stats calculation complete!"