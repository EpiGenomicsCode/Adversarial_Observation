import os
import glob
import time
import tqdm
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from models import *
SAMPLING_RATE = 16000
NUM_CLASSES = 10
RANDOM_SEED = 42
BATCH_SIZE = 32

def normalize_mnist(x):
    return x / 255.0

def pad_audio(audio, max_len):
    return audio[:max_len] if len(audio) > max_len else np.pad(audio, (0, max_len - len(audio)), 'constant')

def load_dataset(data_path):
    data, labels = [], []
    max_len = 0
    wav_files = glob.glob(os.path.join(data_path, '*', '*.wav'))

    for audio_path in tqdm.tqdm(wav_files, desc="Loading audio files"):
        audio, sr = librosa.load(audio_path, sr=SAMPLING_RATE)
        data.append(audio)
        try:
            label = int(audio_path.split('/')[-1][0])
            labels.append(label)
        except ValueError:
            print(f"Skipping file {audio_path} due to invalid class format")
        max_len = max(max_len, len(audio))

    data = [pad_audio(audio, max_len) for audio in data]
    return np.array(data), np.array(labels), max_len

def prepare_datasets(data, labels, max_len, test_size=0.2):
    data = np.array([pad_audio(audio, max_len) for audio in data])[..., np.newaxis]
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=RANDOM_SEED)
    x_train = x_train.astype(np.float32) / np.max(np.abs(x_train))
    x_test = x_test.astype(np.float32) / np.max(np.abs(x_test))
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)
    return train_ds, test_ds, (x_test, y_test)

def load_audio_mnist_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist.")
    data, labels, max_len = load_dataset(data_path)
    train_ds, test_ds, _ = prepare_datasets(data, labels, max_len)
    return train_ds, test_ds, max_len


def load_data(batch_size=32, dataset_type="MNIST", use_augmentation=False):
    if dataset_type == "MNIST":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = normalize_mnist(x_train.reshape(-1, 28, 28, 1).astype('float32'))
        x_test = normalize_mnist(x_test.reshape(-1, 28, 28, 1).astype('float32'))
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        if use_augmentation:
            # Data Augmentation with ImageDataGenerator
            datagen = ImageDataGenerator(
                rotation_range=10,
                zoom_range=0.10,
                width_shift_range=0.1,
                height_shift_range=0.1
            )
            datagen.fit(x_train)
            train_datagen = datagen.flow(x_train, y_train, batch_size=batch_size)
            train_dataset = tf.data.Dataset.from_generator(
                lambda: train_datagen,
                output_signature=(
                    tf.TensorSpec(shape=(batch_size, 28, 28, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(batch_size, 10), dtype=tf.float32)
                )
            )
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
        return train_dataset, test_dataset

    elif dataset_type == "MNIST_Audio":
        data_path = "./AudioMNIST/data"
        train_ds, test_ds, max_len = load_audio_mnist_data(data_path)
        return train_ds, test_ds

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def train_model(model, train_dataset, epochs=10):
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}:")
        start = time.time()
        loss, acc, batches = 0, 0, 0
        for x, y in train_dataset:
            l, a = model.train_on_batch(x, y)
            loss += l
            acc += a
            batches += 1
        print(f"Loss: {loss/batches:.4f}, Accuracy: {acc/batches:.4f}, Time: {time.time()-start:.2f}s")
    return model

def evaluate_model(model, test_dataset):
    y_true = []
    loss, acc = model.evaluate(test_dataset, verbose=0)
    for _, y in test_dataset:
        y_true.extend(np.argmax(y.numpy(), axis=1))

    y_pred = model.predict(test_dataset, verbose=0)
    auroc = roc_auc_score(to_categorical(y_true, NUM_CLASSES), y_pred, multi_class='ovr')
    auprc = average_precision_score(to_categorical(y_true, NUM_CLASSES), y_pred)
    print(f"Test Loss: {loss:.4f}, Accuracy: {acc:.4f}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")

def train_model_and_save(args):
    # Create folder name based on dataset and model type
    folder_name = f"{args.data}_{args.model_type}"
    save_dir = os.path.join(args.save_dir, folder_name)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'{folder_name}.keras')

    # Load or train the model based on the data and model type
    if args.data == 'MNIST':
        if args.model_type == 'normal':
            model = load_MNIST_model(model_path)
            train_ds, test_ds = load_data(dataset_type="MNIST", use_augmentation=False)
        elif args.model_type == 'complex':
            model = load_complex_MNIST_model(model_path)
            train_ds, test_ds = load_data(dataset_type="MNIST", use_augmentation=False)
        elif args.model_type == 'complex_augmented':
            model = load_complex_MNIST_model(model_path)
            train_ds, test_ds = load_data(dataset_type="MNIST", use_augmentation=True)
    elif args.data == 'MNIST_Audio':
        if args.model_type == 'normal':
            model = load_AudioMNIST_model(model_path)
            train_ds, test_ds = load_data(dataset_type="MNIST_Audio", use_augmentation=False)
        elif args.model_type == 'complex':
            model = load_complex_AudioMNIST_model(model_path)
            train_ds, test_ds = load_data(dataset_type="MNIST_Audio", use_augmentation=False)
        elif args.model_type == 'complex_augmented':
            model = load_complex_AudioMNIST_model(model_path)
            train_ds, test_ds = load_data(dataset_type="MNIST_Audio", use_augmentation=True)

    # Check if model weights exist, if not, train and save
    if not os.path.exists(model_path):
        print("Training the model...")
        model = train_model(model, train_ds, epochs=args.epochs)
        model.save(model_path)
        print(f"Model saved to {model_path}")
    else:
        print(f"Model found. Loading weights from {model_path}")
        model.load_weights(model_path)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, test_ds)

    return model, test_ds, save_dir, model_path
