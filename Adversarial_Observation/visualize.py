import imageio
from typing import List
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualizeGIF(filenames: List[str], output_file: str = 'output.gif') -> None:
    """
    Visualizes the images in the folder as a GIF.
    :param foldername: The folder where the images are stored.
    :param epochs: A list of filenames of the images to include in the GIF.
    :param output_file: The name of the output GIF file (default: 'output.gif').
    """
    # Load the images from the file names.
    images = [imageio.imread(f'{filename}') for filename in filenames]

    # Save the images as a GIF.
    imageio.mimsave(output_file, images, duration=.5)
    #close the images
    plt.close('all')
    
    
