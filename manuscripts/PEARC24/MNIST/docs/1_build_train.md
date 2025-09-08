## MNIST Training and Evaluation

This code demonstrates the training and evaluation process for a Convolutional Neural Network (CNN) model on the MNIST dataset, which contains grayscale images of handwritten digits (0-9).

The script, `1_build_train.py`, trains a CNN using the MNIST dataset, evaluates its performance, and saves the results and the trained model.

### Functions

The code includes the following functions:

#### 1. `trainModel(model, train_loader, optimizer, loss, epoch, filename)`

This function trains the model for one epoch using the provided data loader, optimizer, and loss function. It saves the training loss to the specified file.

* **`model`**: The CNN model to be trained.
* **`train_loader`**: The data loader for the training data.
* **`optimizer`**: The optimizer used for training (e.g., Adam).
* **`loss`**: The loss function (e.g., CrossEntropyLoss).
* **`epoch`**: The current epoch number.
* **`filename`**: The file path to save the training loss.

#### 2. `testModel(model, test_loader, filename)`

This function evaluates the model on the test dataset, calculating the test loss and accuracy. It saves the test loss to the specified file.

* **`model`**: The CNN model to be evaluated.
* **`test_loader`**: The data loader for the test data.
* **`filename`**: The file path to save the test loss.

#### 3. `seedEverything(seed)`

This function ensures reproducibility by seeding all random number generators across the framework.

* **`seed`**: The seed value for random number generators.

#### 4. `main()`

This is the main function that orchestrates the training and evaluation process. It performs the following steps:

* Seeds the random number generators for reproducibility using `seedEverything()`.
* Loads the MNIST dataset with `load_MNIST_data()` from the `AO.utils` module.
* Builds the CNN model using `load_MNIST_model()` from the `AO.utils` module.
* Sets up the Adam optimizer and CrossEntropy loss function.
* Trains the model for a specified number of epochs, calling the `trainModel()` and `testModel()` functions for each epoch.
* Saves the trained model’s parameters to the file `MNIST_cnn.pt`.

### Usage

To use this code:

1. **Install Dependencies**
   Ensure that you have the following Python libraries installed:

   * `torch`
   * `torchvision`
   * `numpy`
   * `tqdm`

   You can install them using `pip`:

   ```bash
   pip install torch torchvision numpy tqdm
   ```

2. **Run the Script**
   Execute the Python script `1_build_train.py`:

   ```bash
   python 1_build_train.py
   ```

3. **Monitor Training and Testing**
   The script will output the training loss and test loss for each epoch in the terminal. The progress bar (from `tqdm`) will also show the current status of training.

4. **Review the Log File**
   The training loss and test loss will be appended to a file called `log.csv`. Each line in the file will record the training and test loss for each epoch.

5. **Saved Model**
   After training completes, the final model’s parameters (weights) will be saved to the file `MNIST_cnn.pt`. You can load this file later for inference or further training.

6. **Modify Configuration**
   If necessary, modify the paths, filenames, or number of epochs in the script to suit your needs.

### Example Output

In the `log.csv` file, you will see entries like:

```
Train Epoch: 1 Loss: 0.350100
Test set: Average loss: 0.0512, Accuracy: 97%
Train Epoch: 2 Loss: 0.090200
Test set: Average loss: 0.0451, Accuracy: 98%
```

### Notes

* **Device**: The model will be trained on a GPU if available, otherwise it will fall back to CPU. This is automatically detected in the code.

* **Reproducibility**: The random seed is fixed (`42`), so the results should be consistent across multiple runs.

* **Custom `AO` Module**: This script depends on a custom module `Adversarial_Observation` (imported as `AO`) for utilities like `load_MNIST_data()` and `load_MNIST_model()`. Make sure this module is available and correctly implemented.

