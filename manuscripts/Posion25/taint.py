import os
import numpy as np
import tensorflow as tf
import pickle
from manuscripts.Posion25.analysis import *
from Adversarial_Observation.Swarm import ParticleSwarm
from analysis import *

def adversarial_attack_blackbox(model, dataset, image_index, output_dir='results', num_iterations=30, num_particles=100):
    
    pickle_path = os.path.join(output_dir, 'attacker.pkl')

    dataset_list = list(dataset.as_numpy_iterator())
    all_images, all_labels = zip(*dataset_list)
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if image_index < 0 or image_index >= len(all_images):
        raise ValueError(f"Image index {image_index} out of range")

    single_input = all_images[image_index]
    single_target = np.argmax(all_labels[image_index])
    target_class = (single_target + 1) % 10

    input_set = np.stack([
        single_input + (np.random.uniform(0, 1, single_input.shape) * (np.random.rand(*single_input.shape) < 0.9))
        for _ in range(num_particles)
    ])

    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            attacker = pickle.load(f)
        print(f"Loaded attacker from {pickle_path}")
    else:

        attacker = ParticleSwarm(
            model=model, input_set=input_set, starting_class=single_target,
            target_class=target_class, num_iterations=num_iterations,
            save_dir=output_dir, inertia_weight=0.01
        )
        attacker.optimize()
        # save the attacker as a pickle
        with open(pickle_path, 'wb') as f:
            pickle.dump(attacker, f)
        print(f"Saved attacker to {pickle_path}")
    print("Adversarial attack completed. Analyzing results...")
    analyze_attack(attacker, single_input, target_class)

def best_analysis(attacker, original_data, target):
    adv = attacker.global_best_position.numpy()
    save_dir = attacker.save_dir
    ensure_dir(save_dir)

    # save the original data
    save_array_csv(os.path.join(save_dir, "original_data.csv"), original_data)
    save_ndarray_visualization(os.path.join(save_dir, "original_data.png"), original_data)
    save_softmax_stats(os.path.join(save_dir, "original_data_stats.tsv"),
                       *get_softmax_stats(attacker.model, original_data), target)
    
    # Save best particle
    save_array_csv(os.path.join(save_dir, "best_particle.csv"), adv)
    save_ndarray_visualization(os.path.join(save_dir, "best_particle.png"), adv)
    save_softmax_stats(os.path.join(save_dir, "best_particle_stats.tsv"),
                       *get_softmax_stats(attacker.model, adv), target)

    # Save difference
    diff = original_data - adv
    save_array_csv(os.path.join(save_dir, "attack_vector_best_particle.csv"), diff)
    save_ndarray_visualization(
        os.path.join(save_dir, "attack_vector_best_particle.png"),
        diff, mode="auto", cmap="seismic", vmin=-1, vmax=1
    )

    # Save stats
    softmax_output, max_val, max_class = get_softmax_stats(attacker.model, adv)
    save_softmax_stats(os.path.join(save_dir, "best_particle_stats.tsv"), softmax_output, max_class, max_val, target)

def denoise_analysis(attacker, original_data, denoised_data, target):
    save_dir = attacker.save_dir
    ensure_dir(save_dir)

    save_array_csv(os.path.join(save_dir, "best_particle-clean.csv"), denoised_data)
    save_ndarray_visualization(os.path.join(save_dir, "best_particle-clean.png"), denoised_data)

    diff = original_data - denoised_data
    save_array_csv(os.path.join(save_dir, "attack_vector_best_particle-clean.csv"), diff)
    save_ndarray_visualization(
        os.path.join(save_dir, "attack_vector_best_particle-clean.png"),
        diff, mode="auto", cmap="seismic", vmin=-1, vmax=1
    )

    softmax_output, max_val, max_class = get_softmax_stats(attacker.model, denoised_data)
    save_softmax_stats(os.path.join(save_dir, "best_particle-clean_stats.tsv"), softmax_output, max_class, max_val, target)

def reduce_excess_perturbations(attacker, original_data, adv_data, target_label):
    """
    Reduce unnecessary perturbations in adversarial data while maintaining misclassification.
    Works for data of any shape.
    """
    original_data = np.squeeze(original_data)
    adv_data = np.squeeze(adv_data)

    if original_data.shape != adv_data.shape:
        raise ValueError("Original and adversarial data must have the same shape after squeezing.")

    adv_data = adv_data.copy()
    changed = True

    # Wrap the iteration with tqdm to monitor progress
    while changed:
        changed = False
        indices = list(np.ndindex(original_data.shape))
        for idx in tqdm(indices, desc="Reducing perturbations"):
            if np.isclose(original_data[idx], adv_data[idx]):
                continue

            original_val = original_data[idx]
            current_val = adv_data[idx]

            # Try reverting completely
            adv_data[idx] = original_val
            pred = predict_class(attacker.model, adv_data)

            if pred != target_label:
                # Try partial revert
                adv_data[idx] = current_val + 0.5 * (original_val - current_val)
                pred = predict_class(attacker.model, adv_data)
                if pred != target_label:
                    adv_data[idx] = current_val
                else:
                    changed = True
            else:
                changed = True

    return adv_data

def reduce_excess_perturbations_scale(attacker, original_data, adv_data, target_label, tol=1e-4, max_iter=20):
    original_data = np.squeeze(original_data)
    adv_data = np.squeeze(adv_data)

    if original_data.shape != adv_data.shape:
        raise ValueError("Original and adversarial data must have the same shape after squeezing.")

    perturbation = adv_data - original_data
    low, high = 0.0, 1.0
    best_scaled = adv_data

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        candidate = original_data + mid * perturbation
        pred = predict_class(attacker.model, candidate)

        if pred == target_label:
            best_scaled = candidate
            high = mid
        else:
            low = mid

        if abs(high - low) < tol:
            break

    return best_scaled

def full_analysis(attacker, input_data, target):
    """
    Save full analysis of all particles' histories and confidences.
    """
    analysis = {
        "original_misclassification_input": input_data.tolist(),
        "original_misclassification_target": int(target),
        "particles": []
    }

    for i, particle in tqdm(enumerate(attacker.particles), total=len(attacker.particles), desc="Full Analysis"):
        pdata = {
            "particle_index": i,
            "positions": [],
            "confidence_values": [],
            "max_output_values": [],
            "max_output_classes": [],
            "differences_from_original": []
        }

        for pos in tqdm(particle.history, desc=f"Particle {i} history", leave=False):
            pos_np = pos.numpy() if isinstance(pos, tf.Tensor) else np.array(pos)
            softmax, max_val, max_class = get_softmax_stats(attacker.model, pos_np)
            diff = float(np.linalg.norm(pos_np - input_data))

            pdata["positions"].append(pos_np.tolist())
            pdata["confidence_values"].append(softmax.tolist())
            pdata["max_output_values"].append(max_val)
            pdata["max_output_classes"].append(max_class)
            pdata["differences_from_original"].append(diff)

        analysis["particles"].append(pdata)

    path = os.path.join(attacker.save_dir, "attack_analysis.json")
    with open(path, "w") as f:
        json.dump(analysis, f, indent=4)

    print(f"Full analysis saved to {path}")

def analyze_attack(attacker, original_img, target):
    print("Starting analysis of the adversarial attack...")
    best_analysis(attacker, original_img, target)
    print("Reducing excess perturbations...")
    reduced_img = reduce_excess_perturbations(attacker, original_img, attacker.global_best_position.numpy(), target)
    denoise_analysis(attacker, original_img, reduced_img, target)
    print("Performing full analysis of the attack...")
    full_analysis(attacker, original_img, target)
