import os
import argparse
import csv
import re

def combine_csvs(parent_dir, output_file=None):
    if output_file is None:
        output_file = os.path.join(parent_dir, "combined_best_particle_image_denoise.csv")

    pattern = re.compile(r'^MNIST-rand_test_[^_]+_(\d+)$')

    folders = []
    for name in os.listdir(parent_dir):
        match = pattern.match(name)
        if match:
            folders.append((int(match.group(1)), name))
    folders.sort()

    with open(output_file, 'w', newline='') as out_csv:
        writer = None

        for folder_num, folder_name in folders:
            file_path = os.path.join(parent_dir, folder_name, "best_particle_image_denoise.csv")

            if not os.path.isfile(file_path):
                print(f"Skipping missing: {file_path}")
                continue

            try:
                with open(file_path, 'r') as in_csv:
                    reader = csv.reader(in_csv)
                    for row in reader:
                        row_with_tag = [folder_num] + row
                        if writer is None:
                            writer = csv.writer(out_csv)
                            print(f"Writing headerless CSV with folder_num prefix to: {output_file}")
                        writer.writerow(row_with_tag)

#                print(f"Processed: {folder_name}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine clean particle CSVs from subfolders into one file.")
    parser.add_argument("parent_dir", help="Path to the parent directory containing MNIST_test_* folders")
    parser.add_argument("--output_file", help="Optional path to the output combined CSV")

    args = parser.parse_args()
    combine_csvs(args.parent_dir, args.output_file)

