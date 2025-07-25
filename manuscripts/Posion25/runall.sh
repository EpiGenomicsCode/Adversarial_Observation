#!/bin/bash

# Define available datasets and model types
datasets=("MNIST" "MNIST_Audio")
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
done

# Step 3: Collect statistics after all attacks have been run
echo "Collecting statistics..."
python 3_stats.py --data MNIST --iterations 10 --particles 100 --save_dir $save_dir
python 3_stats.py --data MNIST_Audio --iterations 10 --particles 100 --save_dir $save_dir

echo "Statistics collection complete."
