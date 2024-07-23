# MNIST Digit Recognizer in C

This project implements a simple neural network to recognize handwritten digits from the MNIST dataset using C. It uses ReLU activation in the hidden layer and softmax in the output layer.

## Features

- Single hidden layer neural network
- ReLU activation function in the hidden layer
- Softmax activation function in the output layer

## Dataset

The project uses the MNIST dataset from Kaggle. Place the following files in the `dataset` folder:
- train.csv
- test.csv

## Compilation

To compile the project, use the following command:
## Running the Program
gcc main.c neuralNetwork.c utils.c -o mnist_recognizer -lm

After compilation, run the program with:
./mnist_recognizer

The program will train on the training data and then generate a `sample_submission.csv` file with predictions for the test set, which you can submit to Kaggle.

## Note

This is a basic implementation and may take a while to run on the full dataset. For better performance, consider techniques like mini-batch training or using a more optimized linear algebra library.
