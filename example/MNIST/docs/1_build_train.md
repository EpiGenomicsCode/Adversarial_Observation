## MNIST Training and Evaluation

This code, implemented in `1_build_train.py`, demonstrates the training and evaluation process for a Convolutional Neural Network (CNN) model on the MNIST dataset. The MNIST dataset consists of grayscale images of handwritten digits from 0 to 9.

### Functions

The code includes the following functions:

#### 1. `trainModel(model, train_loader, optimizer, loss, epoch, filename)`

This function trains the model for one epoch using the provided data loader, optimizer, and loss function. It saves the training loss to the specified file.

- `model`: The model to be trained.
- `train_loader`: The data loader for the training data.
- `optimizer`: The optimizer used for training.
- `loss`: The loss function.
- `epoch`: The current epoch number.
- `filename`: The name of the file to save the training loss.

#### 2. `testModel(model, test_loader, filename)`

This function evaluates the model using the provided data loader and calculates the test loss and accuracy. It saves the test loss to the specified file.

- `model`: The model to be evaluated.
- `test_loader`: The data loader for the test data.
- `filename`: The name of the file to save the test loss.

#### 3. `seedEverything(seed)`

This function seeds all the random number generators to ensure reproducibility.

- `seed`: The seed value for random number generators.

#### 4. `main()`

This is the main function that orchestrates the training and evaluation process. It performs the following steps:

- Seeds the random number generators for reproducibility.
- Loads the MNIST dataset using the `load_MNIST_data()` function from the `util` module.
- Builds the CNN model using the `build_MNIST_Model()` function from the `util` module.
- Sets up the optimizer and loss function.
- Trains the model for a specified number of epochs, calling the `trainModel()` and `testModel()` functions.
- Saves the trained model to a file.

### Usage

To use this code:

1. Make sure you have the necessary dependencies installed.
2. Run the Python script named `1_build_train.py`.
3. Execute the script to train the model and evaluate its performance on the MNIST dataset.
4. The training loss will be saved in the `log.csv` file.
5. The trained model will be saved in the `MNIST_cnn.pt` file.

Note: Before running the code, you may need to modify the paths and filenames to match your desired configuration.
