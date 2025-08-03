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

