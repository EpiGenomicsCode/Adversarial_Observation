#!/usr/bin/env python3
import os
import glob
import hashlib
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/AudioMNIST", help="Path to AudioMNIST dataset")
    parser.add_argument("--output", type=str, default="audioMNIST_test_labels.tsv", help="Output file name")
    args = parser.parse_args()

    wav_files = glob.glob(os.path.join(args.data, '*', '*.wav'))
    if len(wav_files) == 0:
        print(f"Warning: No .wav files found in {args.data}. Are you sure the dataset is downloaded?")
        return

    # Deterministic shuffle to match training pipeline
    wav_files = sorted(wav_files, key=lambda x: hashlib.md5(x.encode()).hexdigest())

    # 80/20 Split
    train_size = int(0.8 * len(wav_files))
    test_files = wav_files[train_size:]
    
    # In AudioMNIST, the test indices naturally continue from train_size
    test_indices = list(range(train_size, len(wav_files)))

    # Extract labels (AudioMNIST labels are the first character of the filename)
    labels = [int(os.path.basename(f)[0]) for f in test_files]

    # Save to TSV
    df = pd.DataFrame({
        "Index": test_indices,
        "Label": labels
    })
    
    df.to_csv(args.output, sep="\t", index=False)
    print(f"Successfully generated {len(test_files)} test labels at {args.output}")

if __name__ == "__main__":
    main()