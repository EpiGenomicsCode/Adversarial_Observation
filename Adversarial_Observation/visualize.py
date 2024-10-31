import imageio
from typing import List
import matplotlib.pyplot as plt

def visualize_gif(filenames: List[str], output_file: str = 'output.gif') -> None:
    """
    Create a GIF from a list of image filenames.

    Args:
        filenames (List[str]): List of image filenames.
        output_file (str): Output filename for the GIF (default: 'output.gif').

    Returns:
        None
    """
    images = [imageio.imread(filename) for filename in filenames]
    imageio.mimsave(output_file, images, duration=0.5)
    plt.close('all')
