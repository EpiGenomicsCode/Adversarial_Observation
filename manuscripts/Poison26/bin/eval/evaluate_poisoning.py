import os
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser(description="Check poisoning success from experiment results.")
    parser.add_argument("--main_folder", type=str, required=True,
                        help="Path to the main folder containing MNIST_test_* subfolders.")
    parser.add_argument("--labels_file", type=str, required=True,
                        help="Path to TSV file with columns: index, trueLabel, falseLabel.")
    parser.add_argument("--output_prefix", type=str, default="poison",
                        help="Prefix for output files (default: poison).")
    return parser.parse_args()


def process_subfolder(subfolder_name, main_folder):
    file_path = os.path.join(main_folder, subfolder_name, "best_particle_stats_denoise.tsv")
    if not os.path.isfile(file_path):
        print(f"Warning: {file_path} not found.")
        return subfolder_name, None, None

    best_class = None
    target_class = None
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if best_class is None and line.startswith("Best Class"):
                parts = line.split()
                if len(parts) >= 3:
                    best_class = parts[2]
            elif target_class is None and line.startswith("Target Class"):
                parts = line.split()
                if len(parts) >= 3:
                    target_class = parts[2]
            if best_class is not None and target_class is not None:
                break

    return subfolder_name, best_class, target_class


def main():
    args = parse_args()

    # Load label mapping file
    labels_df = pd.read_csv(args.labels_file, sep="\t")
    index_to_labels = {
        str(row["index"]): (str(row["trueLabel"]), str(row["falseLabel"]))
        for _, row in labels_df.iterrows()
    }

    results = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        with os.scandir(args.main_folder) as entries:
            for entry in entries:
                if entry.is_dir() and entry.name.startswith("MNIST_test_"):
                    futures.append(executor.submit(process_subfolder, entry.name, args.main_folder))

        for future in as_completed(futures):
            subfolder, best_class, target_class = future.result()

            if best_class is None or target_class is None:
                continue

            # Extract the index from folder name (assumes format: MNIST_test_<index>)
            index = subfolder.split("_")[-1]
            if index not in index_to_labels:
                print(f"Warning: No label mapping found for {subfolder}")
                continue

            true_label, false_label = index_to_labels[index]

            success = (best_class == false_label)
            results.append((index, subfolder, true_label, false_label, best_class, target_class, success))

    # Save results
    results_df = pd.DataFrame(results, columns=[
        "index", "folder", "trueLabel", "falseLabel", "bestClass", "targetClass", "success"
    ])

    success_df = results_df[results_df["success"]]
    fail_df = results_df[~results_df["success"]]

    success_out = f"{args.output_prefix}_success.tsv"
    fail_out = f"{args.output_prefix}_fail.tsv"

    success_df.to_csv(success_out, sep="\t", index=False)
    fail_df.to_csv(fail_out, sep="\t", index=False)

    print(f"Poisoning successful cases saved to {success_out}")
    print(f"Poisoning failed cases saved to {fail_out}")


if __name__ == "__main__":
    main()

