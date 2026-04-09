#!/usr/bin/env python3
"""
Generate violin plots of |attack-vector_best_particle_denoise.csv| sums
and report comprehensive summary statistics per model directory.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Generate violin plots of absolute sums per model directory and detailed summary stats."
    )
    parser.add_argument(
        "--dataset",
        choices=["MNIST", "CIFAR10"],
        default="MNIST",
        help="Dataset name (determines model directories). Choices: MNIST or CIFAR10. Default: MNIST",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output SVG filename for the violin plot (e.g., results.svg)",
    )
    parser.add_argument(
        "--summary",
        default="attack_vector_abs_sum_stats_per_model.tsv",
        help="Output TSV file to save summary (default: attack_vector_abs_sum_stats_per_model.tsv)",
    )
    parser.add_argument(
        "--filename",
        default="attack-vector_best_particle_denoise.csv",
        help="Target CSV filename to search in model directories (default: attack-vector_best_particle_denoise.csv)",
    )
    args = parser.parse_args()

    # --- CONFIG: choose model directories based on dataset ---
    if args.dataset == "MNIST":
        model_dirs = [f"MNIST-rand_model{i}" for i in range(1, 6)]
    else:  # CIFAR10
        model_dirs = [f"CIFAR10-rand_model{i}" for i in range(1, 6)]

    filename = args.filename

    # --- COLLECT DATA ---
    data_groups = {}
    summary_rows = []
    total_valid = total_invalid = 0

    for model_dir in model_dirs:
        pattern = os.path.join(model_dir, "**", filename)
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
                print(f"Skipping {filepath}: {e}")

        data_groups[model_dir] = abs_sums

        # Compute summary stats
        if abs_sums:
            abs_sums_np = np.array(abs_sums)
            max_idx = int(np.argmax(abs_sums_np))
            stats = {
                "model": model_dir,
                "num_valid_files": len(abs_sums),
                "num_invalid_files": invalid_count,
                "max_abs_sum": np.max(abs_sums_np),
                "max_folder": os.path.dirname(filepaths[max_idx]),
                "mean_abs_sum": np.mean(abs_sums_np),
                "median_abs_sum": np.median(abs_sums_np),
                "std_abs_sum": np.std(abs_sums_np),
                "min_abs_sum": np.min(abs_sums_np),
                "iqr_abs_sum": np.percentile(abs_sums_np, 75) - np.percentile(abs_sums_np, 25),
                "cv_abs_sum": np.std(abs_sums_np) / np.mean(abs_sums_np) if np.mean(abs_sums_np) != 0 else np.nan,
            }
            print(f"{model_dir}: {len(abs_sums)} files | mean={stats['mean_abs_sum']:.6f} | max={stats['max_abs_sum']:.6f} | folder={stats['max_folder']}")
        else:
            stats = {
                "model": model_dir,
                "num_valid_files": 0,
                "num_invalid_files": invalid_count,
                "max_abs_sum": None,
                "max_folder": None,
                "mean_abs_sum": None,
                "median_abs_sum": None,
                "std_abs_sum": None,
                "min_abs_sum": None,
                "iqr_abs_sum": None,
                "cv_abs_sum": None,
            }
            print(f"{model_dir}: no valid files found (invalid={invalid_count}).")

        summary_rows.append(stats)
        total_valid += len(abs_sums)
        total_invalid += invalid_count

    # --- SAVE SUMMARY ---
    df = pd.DataFrame(summary_rows)
    df.to_csv(args.summary, sep="\t", index=False)
    print(f"\nSummary written to {args.summary}")

    # --- GLOBAL SUMMARY ---
    valid_df = df.dropna(subset=["mean_abs_sum"])
    if not valid_df.empty:
        global_summary = {
            "models_analyzed": len(model_dirs),
            "total_valid_files": total_valid,
            "total_invalid_files": total_invalid,
            "global_mean_of_means": valid_df["mean_abs_sum"].mean(),
            "global_max": valid_df["max_abs_sum"].max(),
            "global_min": valid_df["min_abs_sum"].min(),
            "global_std_of_means": valid_df["mean_abs_sum"].std(),
        }
        print("\n=== Global Summary Across Models ===")
        for k, v in global_summary.items():
            print(f"{k}: {v}")
    else:
        print("\nNo valid data found across models.")

    # --- PLOT VIOLIN ---
    plt.figure(figsize=(8, 6))
    plot_data = [data_groups[m] for m in model_dirs if data_groups[m]]
    parts = plt.violinplot(plot_data, showmeans=True, showextrema=True, showmedians=True)

    plt.xticks(range(1, len(model_dirs) + 1), model_dirs, rotation=20)
    plt.ylabel("Sum of |values|")
    plt.title(f"{args.dataset}: Distribution of Absolute Sums per Model")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # --- SAVE AS SVG ---
    plt.savefig(args.output, format="svg")
    print(f"SVG plot saved to {args.output}")
#    plt.show()


if __name__ == "__main__":
    main()

