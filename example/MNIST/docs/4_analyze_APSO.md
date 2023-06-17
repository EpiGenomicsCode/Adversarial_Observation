# Image Smoothing with Convolution Description

The given code is a Python script that performs image smoothing using convolution. It applies a Gaussian smoothing filter to a set of images and saves the smoothed images.

## Functions

### `main()`
This function is the main entry point of the script. It starts by defining variables for image file paths, filter size, and filter sigma. Then, it creates a pool of processes for parallel execution. The `smooth_and_save` function is applied to each image file path using the `imap` method of the pool. The progress is tracked using a progress bar from `tqdm`. Once the smoothing is completed for all images, a message is printed.

### `smooth_and_save(img_path)`
This function takes an image file path as input. It loads the image from the file using `np.load` and applies the `smooth_array` function to smooth the image. The smoothed image is then plotted using `matplotlib.pyplot` and saved to a file with the same name but appended with "_convl.png". The figure is closed to free up resources.

### `smooth_array(img, filter_size=5, filter_sigma=1)`
This function performs image smoothing using convolution. It takes an input image and optional parameters for the filter size and filter sigma. It defines a Gaussian smoothing filter based on the filter size and sigma values. The filter is then applied to the image using `scipy.signal.convolve2d` with mode set to 'same', which ensures the output has the same shape as the input. The smoothed array is returned.

## Execution
The script starts by importing the required libraries and modules. Then, it defines the main entry point function `main()` and the supporting functions `smooth_and_save()` and `smooth_array()`. Finally, the `main()` function is called to execute the script.
