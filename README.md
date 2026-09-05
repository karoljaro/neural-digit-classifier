# Neural Digit Classifier

A learning project focused on building a neural network from scratch with Python and NumPy.

The goal is to understand how a neural network actually works without using machine learning frameworks such as PyTorch, TensorFlow, or Keras.

The model will learn to recognize handwritten digits from `0` to `9`.

## Initial scope

The first version focuses on simple input images:

- one digit per image
- uniform background
- high contrast
- grayscale input
- no shadows
- no perspective distortion
- no rotation

Images will be converted into numerical matrices, normalized, flattened, and passed into a neural network implemented manually with NumPy.

## Learning goals

The project will cover:

- image representation as matrices
- input normalization
- neurons and weights
- bias
- matrix multiplication
- activation functions
- forward propagation
- loss functions
- backpropagation
- gradient descent
- training and evaluation
- prediction on custom images

## Constraints

The neural network itself will be implemented from scratch.

No high-level machine learning frameworks will be used for the model implementation.

## Tech stack

- Python 3.14
- NumPy
- Black
- Flake8

## Status

Work in progress.
