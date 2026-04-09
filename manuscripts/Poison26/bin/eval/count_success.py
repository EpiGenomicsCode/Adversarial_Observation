import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_subfolder(main_folder, subfolder_name):
    file_path = os.path.join(main_folder, subfolder_name, 'best_particle_stats_denoise.tsv')
    if not os.path.isfile(file_path):
        return subfolder_name, None, None
    best_class = None
    target_class = None
    with open(file_path, 'r') as f:
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
    parser = argparse.ArgumentParser(description="Count successful attacks in a results folder.")
    parser.add_argument("--main_folder", type=str, required=True, help="Path to evaluation folder (e.g., CIFAR10_test_basic_standard)")
    args = parser.parse_args()

    matched_count = 0
    mismatched_folders = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        with os.scandir(args.main_folder) as entries:
            for entry in entries:
                # Check for "test_" to cleanly support MNIST_test_, CIFAR10_test_, audioMNIST_test_
                if entry.is_dir() and "_test_" in entry.name:
                    futures.append(executor.submit(process_subfolder, args.main_folder, entry.name))
        
        for future in as_completed(futures):
            subfolder, best_class, target_class = future.result()
            if best_class is None or target_class is None:
                continue
            if best_class == target_class:
                matched_count += 1
            else:
                mismatched_folders.append(subfolder)
            
            if matched_count > 0 and matched_count % 500 == 0:
                print(f"Matched count: {matched_count}")

    print(f"Number of folders with same Best Class and Target Class: {matched_count}")
    print(f"Number of folders with mismatched values: {len(mismatched_folders)}")

if __name__ == "__main__":
    main()