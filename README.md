# MNIST Detection

MNISTDetection is a simple code that utilizes least squares and singular value decomposition (SVD) to distinguish handwritten digits from the MNIST dataset.

## Table of Contents

- [About MNISTDetection](#about-mnistdetection)
- [Code Description](#code-description)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## About MNISTDetection

The MNIST dataset is a widely used benchmark dataset for handwritten digit recognition. MNISTDetection is a Python script that applies the concept of least squares and SVD to predict the digit represented by an input image from the MNIST dataset.

## Code Description

The code performs the following steps:

1. **Loading MNIST dataset**: The code loads the MNIST dataset, which includes handwritten digit images and their corresponding labels.

2. **Separating digits on train data**: The code organizes the training data by separating the images based on their labels, creating a dictionary `train_data` that maps each digit to the indices of its corresponding images.

3. **Creating H matrices**: The code creates a set of matrices `H` for each digit, where each matrix is of size `28x28`. Each row in `H[i]` represents a flattened image associated with digit `i`.

4. **Solving least squares using SVD**: The code defines the `predict_digit` function, which predicts the digit for an input image index `i`. It performs SVD on the corresponding `H[i]` matrix, computes the alpha values, and calculates the residual error between the projected image and the input image. The digit with the minimum error is considered the predicted digit.

5. **Testing the sample**: The code selects a random sample of images from the test data and calls the `predict_digit` function to obtain the predicted digit. It compares the predicted digit with the actual label, displays the image, and calculates the percentage of accurate predictions.

## Installation

To use the MNISTDetection script, follow these steps:

1. Clone this repository to your local machine.
2. Ensure that you have Python 3.x installed.
3. Install the required dependencies using `pip` or another package manager. You can find the necessary dependencies listed in the `requirements.txt` file.

## Contributing

Contributions are welcome! If you'd like to contribute to this repository, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your contribution.
3. Implement your solution or make the necessary changes.
4. Add documentation and comments to explain the solution.
5. Commit and push your changes to your forked repository.
6. Submit a pull request to the main repository.

Please make sure to adhere to the existing coding style and include appropriate tests for your solution.

## License

This project is licensed under the [Apache License 2.0](./LICENSE). Feel free to use the code provided here for your own projects.
