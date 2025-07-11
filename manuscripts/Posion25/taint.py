import os
import numpy as np
import tensorflow as tf
from manuscripts.Posion25.analysis import best_analysis, denoise_analysis, reduce_excess_perturbations, full_analysis
from Adversarial_Observation.Swarm import ParticleSwarm

def adversarial_attack_blackbox(model, dataset, image_index, output_dir='results', num_iterations=30, num_particles=100):
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

    attacker = ParticleSwarm(
        model=model, input_set=input_set, starting_class=single_target,
        target_class=target_class, num_iterations=num_iterations,
        save_dir=output_dir, inertia_weight=0.01
    )
    attacker.optimize()
    analyze_attack(attacker, single_input, target_class)


def analyze_attack(attacker, original_img, target):
    best_analysis(attacker, original_img, target)
    reduced_img = reduce_excess_perturbations(attacker, original_img, attacker.global_best_position.numpy(), target)
    denoise_analysis(attacker, original_img, reduced_img, target)
    full_analysis(attacker, original_img, target)
