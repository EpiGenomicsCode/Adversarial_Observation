# SHAP Values Generation and Visualization

This Python script generates and visualizes SHAP (SHapley Additive exPlanations) values for a given model and dataset. SHAP values help explain the predictions made by a model, showing which parts of an input image were most important for the model's prediction. The script visualizes the original images alongside their SHAP value explanations and saves both the images and the visualizations for further analysis.

## Functions

### `getData(dataloader)`

This function retrieves one sample of each class from the provided dataloader. It:

* Iterates over the dataloader to collect unique samples based on their target labels.
* Sorts the data and target tensors by the target labels.
* Stacks the data samples along the 0th dimension and converts the target list into a PyTorch tensor.

**Returns:**

* A tuple `(data, target)`, where:

  * `data`: A tensor containing the samples, stacked along the 0th dimension.
  * `target`: A tensor containing the corresponding class labels.

### `save_and_plot_shap_values(dataloader, model)`

This function generates SHAP values for the provided model and dataloader, then saves and visualizes the results. It:

1. Checks the availability of a GPU device and moves the model and data to the appropriate device.
2. Retrieves the data and target tensors using the `getData` function.
3. Initializes a `shap.DeepExplainer` to compute SHAP values for each image in the batch.
4. Creates a directory named `SHAP` to store the results (if it doesn't already exist).
5. Creates a 10x11 grid of subplots:

   * The first column shows the original image.
   * The following columns show SHAP values for each class, with a maximum of 10 SHAP values per image.
6. Saves each image and SHAP visualization as `.npy` (for raw data) and `.png` (for visualizations) files.
7. Adds a colorbar to the figure and ensures the figure is properly saved and closed to free up resources.
8. If there are fewer than 10 samples in the dataloader, the empty subplot cells are removed.

**Parameters:**

* `dataloader`: The PyTorch dataloader providing the dataset (e.g., the test set).
* `model`: The trained model to generate SHAP values for.

### `main()`

The main entry point of the script:

1. Loads the MNIST dataset using the `load_MNIST_data` function.
2. Loads the pre-trained MNIST model and its weights from the file `'MNIST_cnn.pt'`.
3. Calls the `save_and_plot_shap_values` function to generate and save SHAP values for the test set.

## Execution

To run the script, follow these steps:

1. **Install dependencies**: Ensure that you have the following Python packages installed:

   * `torch`
   * `numpy`
   * `matplotlib`
   * `shap`
   * `Adversarial_Observation` (custom module)

2. **Run the script**:

   * The script automatically loads the MNIST dataset and the pre-trained model, then generates and saves SHAP values visualizations.
   * You can run the script using the command:

     ```bash
     python shap_values_generation.py
     ```

3. **Output**:

   * SHAP visualizations are saved in the `SHAP` directory.
   * Each sample's image and its corresponding SHAP values for each class are saved as `.npy` files.
   * Visualizations are saved as `.png` images, with a colorbar indicating the magnitude of the SHAP values.

### Example Output:

* A directory structure like:

  ```
  SHAP/
    0_original.npy
    0_shap_0.npy
    0_shap_1.npy
    ...
    shap_values.png
    row_0.png
    row_1.png
    ...
  ```

## Notes

* The script assumes the MNIST images are 28x28 pixels, and the model is a CNN (Convolutional Neural Network) trained on MNIST data. If using a different dataset or model, adjustments to the code (such as image reshaping) may be required.
* The SHAP values for each image are reshaped and visualized for each class in a 28x28 grid. Ensure that the model outputs correspond to a classification task where each class has a separate SHAP explanation.

## Requirements

* **Python**: Version 3.x
* **Libraries**:

  * `torch` (for PyTorch)
  * `shap` (for SHAP explanations)
  * `matplotlib` (for visualization)
  * `numpy` (for numerical operations)
  * `Adversarial_Observation` (custom module, used for loading MNIST data and model)

