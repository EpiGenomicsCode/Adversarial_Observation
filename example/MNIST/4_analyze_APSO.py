import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from tqdm import tqdm
from multiprocessing import Pool

def main():
    images = glob.glob('APSO/*/*/*.npy')
    filter_size = 5
    filter_sigma = 1

    with Pool() as pool:
        results = list(tqdm(pool.imap(smooth_and_save, images), total=len(images)))
    
    print("Smoothing completed.")

def smooth_and_save(img_path):
    convl_filename = img_path.split('.')[0] + '_convl.png'
    img = np.load(img_path)
    smoothed_img = smooth_array(img)
    
    # Set the figure size to match the image size
    fig, ax = plt.subplots(figsize=(img.shape[1] / 80, img.shape[0] / 80), dpi=80)
    ax.imshow(smoothed_img, cmap='gray')
    
    # Remove the axis labels and ticks
    ax.axis('off')
    
    plt.savefig(convl_filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def smooth_array(img, filter_size=5, filter_sigma=1):
    # Define a smoothing filter
    filter_values = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * filter_sigma**2)) * np.exp(
            -((x - filter_size//2)**2 + (y - filter_size//2)**2) / (2 * filter_sigma**2)
        ),
        (filter_size, filter_size)
    )
    filter_values /= np.sum(filter_values)  # Normalize the filter

    # Perform convolution to smooth the array
    smoothed_array = convolve2d(img, filter_values, mode='same')

    return smoothed_array
