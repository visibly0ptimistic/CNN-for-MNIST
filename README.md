# MNIST CNN Classifier

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset using TensorFlow and Keras.

![MNIST CNN Preview](/CNN-for-MNIST.png)

## Project Structure
- `cnn_mnist_classifier.ipynb`: Jupyter notebook containing the complete code with visualizations.
- `cnn_mnist_classifier.py`: Python script version of the classifier.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

You can install the required packages using:
`pip install tensorflow numpy matplotlib seaborn scikit-learn`

## Usage

### Jupyter Notebook

1. Ensure you have Jupyter installed: `pip install jupyter`
2. Navigate to the project directory and run: `jupyter notebook`
3. Open `cnn_mnist_classifier.ipynb` in the Jupyter interface.
4. Run the cells in order to train the model and view the results.

### Python Script

1. Navigate to the project directory.
2. Run the script: `python cnn_mnist_classifier.py`

## Features

- Loads and preprocesses the MNIST dataset.
- Defines and trains a CNN model for digit classification.
- Evaluates the model's performance on a test set.
- Provides various visualizations:
  - Training history (accuracy and loss)
  - Sample predictions
  - Confusion matrix
  - Model architecture
  - Convolutional layer feature maps

## Model Architecture

The CNN model consists of:
- 3 Convolutional layers with ReLU activation
- 2 MaxPooling layers
- Flatten layer
- 2 Dense layers (including the output layer)

## Results

After training, the model typically achieves over 98% accuracy on the test set. Detailed results and visualizations are provided in the notebook/script output.
