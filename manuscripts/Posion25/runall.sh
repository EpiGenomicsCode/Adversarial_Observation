#!/bin/bash

# Define available datasets and model types
datasets=("MNIST")
#  "MNIST_Audio")
model_types=("normal" "complex")
#  "complex_augmented")
save_dir="results"

# Iterate over all combinations of datasets and model types
for dataset in "${datasets[@]}"; do
    for model_type in "${model_types[@]}"; do
        echo "Training and attacking model for dataset: $dataset, model_type: $model_type"
        
        # Step 1: Train the model
        python 1_trainModel.py --data $dataset --model_type $model_type --save_dir $save_dir
        
        # Step 2: Attack the model
        python 2_attackModel.py --data $dataset --model_type $model_type --save_dir $save_dir
        
        echo "Finished training and attacking for $dataset - $model_type"
    done
        # python 3_stats.py --data $dataset --model_type $model_type --save_dir $save_dir
        # echo "Statistics collected for $dataset - $model_type"
done

