#!/usr/bin/env python3
"""
Generate a violin plot of |attack-vector_best_particle_denoise.csv| sums
grouped by original image classification labels for a single model directory.

Usage:
    python attack_vector_violin_by_label.py \
        --dataset MNIST \
        --model_dir MNIST-rand_model1 \
        --label_file index_label.tsv \
        --output_svg MNIST-rand_model1_attack_vector_by_label.svg
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate violin plot of attack-vector sums grouped by label."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["MNIST", "CIFAR10"],
        help="Dataset name (used for labeling and optional behavior).",
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to the model directory containing subfolders with attack-vector_best_particle_denoise.csv files.",
    )
    parser.add_argument(
        "--label_file",
        required=True,
        help="TSV file containing 'Index' and 'Label' columns.",
    )
    parser.add_argument(
        "--output_svg",
        required=True,
        help="Output SVG filename for the violin plot.",
    )
    return parser.parse_args()


def extract_index_from_folder(folder_name: str):
    """
    Extract trailing integer index from folder names like:
      MNIST-rand_test_model1_5476 → 5476
    """
    match = re.search(r"_([0-9]+)$", folder_name)
    return int(match.group(1)) if match else None


def main():
    args = parse_args()

    dataset = args.dataset
    model_dir = args.model_dir
    filename = "attack-vector_best_particle_denoise.csv"
    label_file = args.label_file
    output_svg = args.output_svg

    # --- LOAD LABELS ---
    labels_df = pd.read_csv(label_file, sep="\t")
    label_dict = dict(zip(labels_df["Index"], labels_df["Label"]))

    # --- COLLECT DATA ---
    label_groups = {}
    pattern = os.path.join(model_dir, "**", filename)

    print(f"📂 Scanning {model_dir} for {filename} files...")
    for filepath in glob.iglob(pattern, recursive=True):
        try:
            parent_dir = os.path.basename(os.path.dirname(filepath))
            index = extract_index_from_folder(parent_dir)

            if index is None or index not in label_dict:
                print(f"Skipping {filepath}: could not determine index or not in label file")
                continue

            label = label_dict[index]
            data = np.loadtxt(filepath, delimiter=",")
            total = np.sum(np.abs(data))
            label_groups.setdefault(label, []).append(total)

        except Exception as e:
            print(f"Skipping {filepath}: {e}")

    if not label_groups:
        raise RuntimeError(f"No valid data found in {model_dir}")

    # --- SUMMARY ---
    summary_rows = [
        {"label": label, "num_files": len(vals), "mean_abs_sum": np.mean(vals)}
        for label, vals in sorted(label_groups.items())
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_out = os.path.splitext(output_svg)[0] + "_summary.tsv"
    summary_df.to_csv(summary_out, sep="\t", index=False)
    print(f"Summary written to {summary_out}")

    # --- PLOT VIOLIN ---
    plt.figure(figsize=(8, 6))
    sorted_labels = sorted(label_groups.keys())
    plt.violinplot(
        [label_groups[l] for l in sorted_labels],
        showmeans=True,
        showextrema=True,
    )
    plt.xticks(range(1, len(sorted_labels) + 1), sorted_labels)
    plt.ylabel("Sum of |values|")
    plt.xlabel("Original Label")

    plt.title(
        f"{dataset} - Distribution of Absolute Sums by Label\n({os.path.basename(model_dir)})"
    )
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(output_svg, format="svg")
    print(f"SVG plot saved to {output_svg}")

if __name__ == "__main__":
    main()

