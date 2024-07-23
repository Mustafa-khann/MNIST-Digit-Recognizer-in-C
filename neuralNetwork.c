#include <any>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "neuralNetwork.h"

// Activation function: ReLU (Rectified Linear Unit)
double relu(double x)
{
    return x > 0 ? x : 0; // Return x if it's positive, otherwise return 0
}

// Derivative of the ReLU function
double reluDerivative(double x)
{
    return x > 0 ? 1 : 0; // Return 1 if x is positive, otherwise return 0
}

// Softmax function to normalize the input array
void softmax(double* input, int size)
{
    double max = -DBL_MAX; // Initialize max to the smallest possible double value
    for (int i = 0; i < size; i++) // Fix the loop limit from DBL_MAX to size
    {
        if (input[i] > max)
        {
            max = input[i]; // Find the maximum value in the input array
        }
    }

    double sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        input[i] = exp(input[i] - max); // Exponentiate each input value (stabilized by subtracting max)
        sum += input[i]; // Sum up all the exponentiated values
    }

    for (int i = 0; i < size; i++)
    {
        input[i] /= sum; // Normalize each value by dividing by the sum
    }
}

// Function to create and initialize a new neural network
NeuralNetwork* createNeuralNetwork(int inputSize, int hiddenSize, int outputSize)
{
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork)); // Allocate memory for the neural network
    nn->inputSize = inputSize;
    nn->hiddenSize = hiddenSize;
    nn->outputSize = outputSize;

    // Allocate memory for hidden layer weights
    nn->hiddenWeights = (double **)malloc(inputSize * sizeof(double *));
    for (int i = 0; i < inputSize; i++)
    {
        nn->hiddenWeights[i] = (double *)malloc(hiddenSize * sizeof(double));
        for (int j = 0; j < hiddenSize; j++)
        {
            nn->hiddenWeights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1; // Initialize weights randomly between -1 and 1
        }
    }

    // Allocate memory for output layer weights
    nn->outputWeights = (double **)malloc(hiddenSize * sizeof(double *));
    for (int i = 0; i < hiddenSize; i++)
    {
        nn->outputWeights[i] = (double *)malloc(outputSize * sizeof(double));
        for (int j = 0; j < outputSize; j++)
        {
            nn->outputWeights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1; // Initialize weights randomly between -1 and 1
        }
    }

    // Allocate memory for hidden and output biases
    nn->hiddenBias = (double *)malloc(hiddenSize * sizeof(double));
    nn->outputBias = (double *)malloc(outputSize * sizeof(double));

    for (int i = 0; i < hiddenSize; i++)
    {
        nn->hiddenBias[i] = ((double)rand() / RAND_MAX) * 2 - 1; // Initialize biases randomly between -1 and 1
    }
    for (int i = 0; i < outputSize; i++)
    {
        nn->outputBias[i] = ((double)rand() / RAND_MAX) * 2 - 1; // Initialize biases randomly between -1 and 1
    }

    return nn; // Return the created neural network
}

// Function for forward propagation
void forwardPropagation(NeuralNetwork *nn, double *input, double *hiddenLayer, double *outputLayer)
{
    // Calculate activations for the hidden layer
    for (int i = 0; i < nn->hiddenSize; i++)
    {
        hiddenLayer[i] = 0;
        for (int j = 0; j < nn->inputSize; j++) // Fixed loop limit from 'i' to 'j'
        {
            hiddenLayer[i] += input[j] * nn->hiddenWeights[j][i]; // Compute weighted sum for hidden layer
        }
        hiddenLayer[i] = relu(hiddenLayer[i] + nn->hiddenBias[i]); // Apply ReLU activation function and add bias
    }

    // Calculate activations for the output layer
    for (int i = 0; i < nn->outputSize; i++)
    {
        outputLayer[i] = 0;
        for (int j = 0; j < nn->hiddenSize; j++)
        {
            outputLayer[i] += hiddenLayer[j] * nn->outputWeights[j][i]; // Compute weighted sum for output layer
        }
        outputLayer[i] += nn->outputBias[i]; // Add bias to the output layer
    }

    // Apply softmax function to output layer
    softmax(outputLayer, nn->outputSize);
}


void backwardPropagation(NeuralNetwork *nn, double *input, double *hiddenLayer, double *outputLayer, int label, double learningRate)
{
    // Initialize output error array
    double outputError[10] = {0};
    for(int i = 0; i < nn->outputSize; i++)
    {
        // Calculate the error for the output layer
        outputError[i] = outputLayer[i] - (i == label ? 1 : 0);
    }

    // Initialize hidden error array
    double hiddenError[128] = {0}; // Ensure it matches the size of your hidden layer
    for(int i = 0; i < nn->hiddenSize; i++)
    {
        hiddenError[i] = 0;
        for(int j = 0; j < nn->outputSize; j++)
        {
            // Accumulate the weighted output errors
            hiddenError[i] += outputError[j] * nn->outputWeights[i][j];
        }
        // Apply the derivative of the activation function
        hiddenError[i] *= reluDerivative(hiddenLayer[i]);
    }

    // Update output weights
    for(int i = 0; i < nn->hiddenSize; i++)
    {
        for(int j = 0; j < nn->outputSize; j++)
        {
            // Adjust output weights
            nn->outputWeights[i][j] -= learningRate * outputError[j] * hiddenLayer[i];
        }
    }

    // Update hidden weights
    for (int i = 0; i < nn->inputSize; i++)
    {
        for(int j = 0; j < nn->hiddenSize; j++)
        {
            // Adjust hidden weights
            nn->hiddenWeights[i][j] -= learningRate * hiddenError[j] * input[i];
        }
    }

    // Update output biases
    for (int i = 0; i < nn->outputSize; i++)
    {
        nn->outputBias[i] -= learningRate * outputError[i];
    }

    // Update hidden biases
    for (int i = 0; i < nn->hiddenSize; i++)
    {
        nn->hiddenBias[i] -= learningRate * hiddenError[i];
    }
}

void trainNetwork(NeuralNetwork *nn, double **trainingData, int *labels, int *numSamples, int *epochs, double learningRate)
{
    double hiddenLayer[128];
    double outputLayer[10];

    for(int epoch = 0; epoch < epochs; epoch++)
        {
            for(int i = 0; i < numSamples; i++)
                {
                    forwardPropogation(nn, trainingData[i], hiddenLayer, outputLayer);
                    backwardPropogation(nn, trainingData[i], hiddenLayer, outputLayer, labels[i], learningRate);
                }
            printf("Epoch %d Completed\n", epoch + 1);
        }
}

double testNetwork(NeuralNetwork *nn, double **testingData, int *labels, int numSamples)
{
    int correctPredictions = 0;
    double hiddenLayer[128];  // Assuming hidden layer size is 128
    double outputLayer[10];   // Assuming output layer size is 10 (for digits 0-9)

    for (int i = 0; i < numSamples; i++)
    {
        forwardPropagation(nn, testingData[i], hiddenLayer, outputLayer);

        // Find the index of the highest output, which is our predicted digit
        int predictedLabel = 0;
        double maxOutput = outputLayer[0];
        for (int j = 1; j < nn->outputSize; j++)
        {
            if (outputLayer[j] > maxOutput)
            {
                maxOutput = outputLayer[j];
                predictedLabel = j;
            }
        }

        // Check if the prediction is correct
        if (predictedLabel == labels[i])
        {
            correctPredictions++;
        }
    }

    // Calculate and return the accuracy
    return (double)correctPredictions / numSamples;
}

void freeNeuralNetwork(NeuralNetwork *nn)
{
    for(int i = 0; i < nn->inputSize; i++ )
        {
            free(nn->hiddenWeights[i]);
        }
    free(nn->hiddenWeights);

    for(int i = 0; i < nn->hiddenSize; i++)
        {
            free(nn->outputWeights[i]);
        }
    free(nn->outputWeights);

    free(nn->hiddenBias);
    free(nn->outputBias);
    free(nn);
}
