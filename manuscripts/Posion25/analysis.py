import numpy as np
import json
from tqdm import tqdm
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import wavfile


def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)


def save_ndarray_visualization(path, array, mode="auto", sample_rate=16000, **kwargs):
    """
    Try to save an ndarray visualization depending on its shape.
    For 2D/3D (image-like), use imsave.
    For 1D or other, use line plot or imshow fallback.
    Additionally, if the data looks like audio waveform (1D),
    save it as a WAV audio file (AudioMNIST style).
    
    Params:
      path: base path (extension used for image, wav saved alongside)
      array: ndarray data
      mode: "auto" or "image" (forces imsave)
      sample_rate: used for wav saving (default 16kHz)
      kwargs: extra params for imsave
    """
    array = np.squeeze(array)

    try:
        if mode == "image" or (mode == "auto" and array.ndim in [2, 3]):
            # Image-like data: save as image
            plt.imsave(path, array, **kwargs)
        elif array.ndim == 1:
            # 1D data - audio waveform?
            # Save waveform plot
            plt.figure()
            plt.plot(array)
            plt.title(os.path.basename(path))
            plt.savefig(path)
            plt.close()

            # Also save as wav audio file (normalize to int16)
            wav_path = os.path.splitext(path)[0] + ".wav"
            audio = array
            # Normalize audio to int16 range
            max_abs = np.max(np.abs(audio))
            if max_abs > 0:
                audio_norm = audio / max_abs
            else:
                audio_norm = audio
            audio_int16 = (audio_norm * 32767).astype(np.int16)
            wavfile.write(wav_path, sample_rate, audio_int16)
        else:
            # fallback for other dims: show as imshow
            plt.figure()
            plt.imshow(array, aspect='auto')
            plt.title(os.path.basename(path))
            plt.savefig(path)
            plt.close()
    except Exception as e:
        print(f"Failed to save visual for {path}: {e}")


def save_array_csv(path, array):
    np.savetxt(path, [array.flatten()], delimiter=',')


def get_softmax_stats(model, x):
    """
    Computes softmax stats for a single input (no batch).
    Automatically reshapes to expected 4D (for CNNs) or leaves as-is.
    """
    x = np.array(x)
    x = np.squeeze(x)

    if x.ndim == 3:  # likely [H, W, C]
        x = x[np.newaxis, ...]  # Add batch dim → [1, H, W, C]
    elif x.ndim == 2:  # likely [H, W]
        x = x[np.newaxis, ..., np.newaxis]  # Add batch and channel → [1, H, W, 1]
    elif x.ndim == 1:
        x = x[np.newaxis, :]  # Just batch → [1, features]
    elif x.ndim == 4:
        pass  # already [batch, H, W, C]
    else:
        raise ValueError(f"Unsupported input shape: {x.shape}")

    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    logits = model(x_tensor)
    softmax_output = tf.nn.softmax(tf.squeeze(logits)).numpy()
    max_val = float(np.max(softmax_output))
    max_class = int(np.argmax(softmax_output))
    return softmax_output, max_val, max_class


def predict_class(model, x):
    """
    Predicts class index. Same auto-shape handling as get_softmax_stats.
    """
    x = np.array(x)
    x = np.squeeze(x)

    if x.ndim == 3:
        x = x[np.newaxis, ...]
    elif x.ndim == 2:
        x = x[np.newaxis, ..., np.newaxis]
    elif x.ndim == 1:
        x = x[np.newaxis, :]
    elif x.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported input shape: {x.shape}")

    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    logits = model(x_tensor)
    return tf.argmax(logits[0]).numpy()


def save_softmax_stats(path, softmax_output, max_class, max_val, target):
    with open(path, 'w') as f:
        f.write("Class\tConfidence\n")
        for i, val in enumerate(softmax_output):
            f.write(f"{i}\t{val}\n")
        f.write(f"\nBest Class\t{max_class}\nMax Confidence\t{max_val}\nTarget Class\t{target}\n")


def best_analysis(attacker, original_data, target):
    adv = attacker.global_best_position.numpy()
    save_dir = attacker.save_dir
    ensure_dir(save_dir)

    # save the original data
    save_array_csv(os.path.join(save_dir, "original_data.csv"), original_data)
    save_ndarray_visualization(os.path.join(save_dir, "original_data.png"), original_data)
    
    # Save best particle
    save_array_csv(os.path.join(save_dir, "best_particle.csv"), adv)
    save_ndarray_visualization(os.path.join(save_dir, "best_particle.png"), adv)

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
