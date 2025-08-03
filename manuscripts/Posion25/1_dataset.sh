#!/bin/bash

# Configuration
SEED=42
BATCH_SIZE=32
DATASETS=("mnist")
SPLITS=("train" "test")

# Step 1: Generate label TSVs
echo "Generating label TSVs..."
for dataset in "${DATASETS[@]}"; do
    echo "Processing $dataset..."
    python 1_dataset_label_tool.py --generate --dataset $dataset --batch_size $BATCH_SIZE
done

# Step 2: Generate false labels
echo "Generating false labels..."
for dataset in "${DATASETS[@]}"; do
    for split in "${SPLITS[@]}"; do
        input_file="${dataset^^}_${split}_labels.tsv"         # e.g., MNIST_train_labels.tsv
        output_file="${dataset^^}_${split}_false_labels.tsv"  # e.g., MNIST_train_false_labels.tsv

        echo "Generating false labels for $input_file..."
        python 1_dataset_label_tool.py --input "$input_file" --output "$output_file" --seed $SEED
    done
done

echo "All label and false label files generated."
