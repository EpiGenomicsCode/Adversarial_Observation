import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Path to your main folder
#main_folder = '/work/hdd/bbse/wklai/AdversarialData/APSO_Poison/manuscripts/POISON25/MNIST_train_model1'
main_folder = '/work/hdd/bbse/wklai/AdversarialData/APSO_Poison/manuscripts/POISON25/MNIST_train_model2'

def process_subfolder(subfolder_name):
    file_path = os.path.join(main_folder, subfolder_name, 'best_particle-clean_stats.tsv')
    if not os.path.isfile(file_path):
        print(f"Warning: {file_path} not found.")
        return subfolder_name, None, None
    best_class = None
    target_class = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # If not yet found, check each relevant line and break early if possible.
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

matched_count = 0
mismatched_folders = []

# Adjust max_workers based on your system (8 is a common starting point).
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = []
    with os.scandir(main_folder) as entries:
        for entry in entries:
            if entry.is_dir() and entry.name.startswith("MNIST_test_"):
                futures.append(executor.submit(process_subfolder, entry.name))
    
    for future in as_completed(futures):
        subfolder, best_class, target_class = future.result()
        if best_class is None or target_class is None:
            print(f"Warning: Could not extract classes from {os.path.join(main_folder, subfolder, 'best_particle-clean_stats.tsv')}")
            continue
        if best_class == target_class:
            matched_count += 1
        else:
            mismatched_folders.append(subfolder)
        # Optionally, print progress every 100 matches
        if matched_count % 500 == 0:
            print(f"Matched count: {matched_count}")

print(f"Number of folders with same Best Class and Target Class: {matched_count}")
print(f"Number of folders with mismatched values: {len(mismatched_folders)}")
print("Folders with mismatched values:")
for folder in mismatched_folders:
    print(folder)
