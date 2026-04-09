#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate violin plots of absolute sums per model directory.")
    parser.add_argument("--dataset", choices=["MNIST", "CIFAR10", "audioMNIST"], required=True)
    parser.add_argument("--models", nargs='+', required=True, help="List of model folders (e.g., mnist_basic_standard mnist_adv_pgd)")
    parser.add_argument("--output", required=True, help="Output SVG filename")
    parser.add_argument("--summary", default="attack_vector_stats.tsv")
    parser.add_argument("--filename", default="attack-vector_best_particle_denoise.csv")
    args = parser.parse_args()

    # Reconstruct folder names based on the standard output format from the attack scripts
    model_dirs = [f"{args.dataset}_test_{m}" for m in args.models]

    data_groups = {}
    summary_rows = []
    total_valid = total_invalid = 0

    for model_dir in model_dirs:
        pattern = os.path.join(model_dir, "**", args.filename)
        abs_sums = []
        filepaths = []
        invalid_count = 0

        for filepath in glob.iglob(pattern, recursive=True):
            try:
                data = np.loadtxt(filepath, delimiter=",")
                total = np.sum(np.abs(data))
                abs_sums.append(total)
                filepaths.append(filepath)
            except Exception as e:
                invalid_count += 1

        data_groups[model_dir] = abs_sums

        if abs_sums:
            abs_sums_np = np.array(abs_sums)
            stats = {
                "model": model_dir,
                "num_valid_files": len(abs_sums),
                "mean_abs_sum": np.mean(abs_sums_np),
                "median_abs_sum": np.median(abs_sums_np),
                "std_abs_sum": np.std(abs_sums_np),
                "min_abs_sum": np.min(abs_sums_np),
                "max_abs_sum": np.max(abs_sums_np),
            }
        else:
            stats = {"model": model_dir, "num_valid_files": 0, "mean_abs_sum": None}
            
        summary_rows.append(stats)
        total_valid += len(abs_sums)

    df = pd.DataFrame(summary_rows)
    df.to_csv(args.summary, sep="\t", index=False)
    print(f"Summary written to {args.summary}")

    # Plot
    plt.figure(figsize=(10, 6))
    plot_data = [data_groups[m] for m in model_dirs if data_groups[m]]
    if plot_data:
        plt.violinplot(plot_data, showmeans=True, showextrema=True, showmedians=True)
        # Use just the underlying model name for cleaner x-axis labels
        plt.xticks(range(1, len(model_dirs) + 1), args.models, rotation=25, ha="right")
        plt.ylabel("Sum of |values|")
        plt.title(f"{args.dataset}: Attack Magnitude per Model")
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(args.output, format="svg")
        print(f"Plot saved to {args.output}")
    else:
        print("No valid data found to plot.")

if __name__ == "__main__":
    main()