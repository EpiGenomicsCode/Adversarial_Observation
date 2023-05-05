import imageio
from typing import List
import matplotlib.pyplot as plt


def visualizeGIF(filenames: List[str], output_file: str = 'output.gif') -> None:
    """
    Given a list of filenames, generate a GIF file in that order and save it to the specified output file.

    Args:
        filenames (List[str]): A list of filenames of the images to include in the GIF.
        output_file (str): The name of the output GIF file (default: 'output.gif').

    Returns:
        None
    
    """
    # Load the images from the file names.
    images = [imageio.imread(f'{filename}') for filename in filenames]

    # Save the images as a GIF.
    imageio.mimsave(output_file, images, duration=.5)
    #close the images
    plt.close('all')

    return None
    

