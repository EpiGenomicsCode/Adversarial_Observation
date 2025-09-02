
# Image Smoothing with Convolution

## Description

This Python script applies Gaussian smoothing (blur) to a set of images using convolution. It loads `.npy` image files, applies a smoothing filter to each image, and then saves the smoothed images as `.png` files. The script uses parallel processing to handle large sets of images efficiently.


## Functions

### `main()`

The entry point of the script. It:

1. Gathers paths to all `.npy` image files from the directory `./APSO/images/*/*npy`.
2. Sets default values for the Gaussian filter size and sigma.
3. Creates a pool of processes to apply the `smooth_and_save` function to each image in parallel.
4. Tracks progress with a `tqdm` progress bar.
5. Once all images are processed, it prints a completion message.

### `smooth_and_save(img_path)`

This function processes each image:

1. Loads the image from the file specified by `img_path` using `np.load()`.
2. Applies a Gaussian smoothing filter via the `smooth_array()` function.
3. Saves the smoothed image as a `.png` file with the same base name but appended with `_convl.png`.
4. The figure is saved with no axis labels or ticks for a clean result.
5. The figure is closed after saving to free resources.

### `smooth_array(img, filter_size=5, filter_sigma=1)`

This function performs the actual image smoothing:

1. Defines a Gaussian kernel based on the given `filter_size` and `filter_sigma`.
2. Applies the kernel to the image using 2D convolution via `scipy.signal.convolve2d()`.
3. The convolution is performed with the `mode='same'` option, ensuring the output image has the same size as the input image.
4. The smoothed image array is returned.

## Execution

1. Clone or download the script to your local machine.
2. Place your `.npy` image files in the `./APSO/images/` directory or modify the script to point to your own directory containing `.npy` files.
3. Run the script:

   ```bash
   python <script_name>.py
   ```

The script will process each image in parallel, smooth it with a Gaussian filter, and save the results as `.png` images in the same directory as the original images.

### Example Run:

```bash
['./APSO/images/folder1/image1.npy', './APSO/images/folder2/image2.npy', ... ]
Smoothing completed.
Saved ./APSO/images/folder1/image1_convl.png
Saved ./APSO/images/folder2/image2_convl.png
...
```

## Customization

* **Filter Size**: The default filter size is set to 5. You can adjust this in the `smooth_array()` function to control the level of smoothing. Larger values result in a stronger blur effect.

* **Sigma**: The default sigma is set to 1. Increasing the sigma will make the Gaussian filter more spread out, which can result in a more pronounced smoothing effect.

* **Image File Paths**: The script assumes images are stored in `.npy` format. Ensure your images are in this format or modify the script to handle other formats if needed.

## Notes

* The script uses parallel processing (`multiprocessing.Pool`) to speed up the smoothing of large sets of images.
* The images are displayed using `matplotlib` and saved without axis labels for a cleaner output.
* By default, the smoothed images are saved in the same directory as the original images with a `_convl.png` suffix. Modify the file-saving path in the `smooth_and_save()` function if needed.

