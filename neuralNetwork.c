#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "neuralNetwork.h"

// Activation function: ReLU (Rectified Linear Unit)
double relu(double x) {
    return (x > 0) ? x : 0;
}

// Derivative of ReLU function
double reluDerivative(double x) {
    return (x > 0) ? 1 : 0;
}

// Softmax activation function for output layer
void softmax(double* input, int size) {
    double max = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        input[i] = exp(input[i] - max);  // Subtract max for numerical stability
        sum += input[i];
    }
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

// Create and initialize a new neural network
NeuralNetwork* createNeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    if (nn == NULL) return NULL;

    nn->inputSize = inputSize;
    nn->hiddenSize = hiddenSize;
    nn->outputSize = outputSize;

    // Allocate and initialize hidden layer weights
    nn->hiddenWeights = (double **)malloc(inputSize * sizeof(double *));
    if (nn->hiddenWeights == NULL) {
        free(nn);
        return NULL;
    }

    for (int i = 0; i < inputSize; i++) {
        nn->hiddenWeights[i] = (double *)malloc(hiddenSize * sizeof(double));
        if (nn->hiddenWeights[i] == NULL) {
            for (int j = 0; j < i; j++) free(nn->hiddenWeights[j]);
            free(nn->hiddenWeights);
            free(nn);
            return NULL;
        }
        for (int j = 0; j < hiddenSize; j++) {
            nn->hiddenWeights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;  // Random initialization between -1 and 1
        }
    }

    // Allocate and initialize output layer weights
    nn->outputWeights = (double **)malloc(hiddenSize * sizeof(double *));
    if (nn->outputWeights == NULL) {
        for (int i = 0; i < inputSize; i++) free(nn->hiddenWeights[i]);
        free(nn->hiddenWeights);
        free(nn);
        return NULL;
    }

    for (int i = 0; i < hiddenSize; i++) {
        nn->outputWeights[i] = (double *)malloc(outputSize * sizeof(double));
        if (nn->outputWeights[i] == NULL) {
            for (int j = 0; j < i; j++) free(nn->outputWeights[j]);
            for (int j = 0; j < inputSize; j++) free(nn->hiddenWeights[j]);
            free(nn->outputWeights);
            free(nn->hiddenWeights);
            free(nn);
            return NULL;
        }
        for (int j = 0; j < outputSize; j++) {
            nn->outputWeights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;  // Random initialization between -1 and 1
        }
    }

    // Allocate and initialize biases
    nn->hiddenBias = (double *)malloc(hiddenSize * sizeof(double));
    nn->outputBias = (double *)malloc(outputSize * sizeof(double));

    if (nn->hiddenBias == NULL || nn->outputBias == NULL) {
        for (int i = 0; i < hiddenSize; i++) free(nn->outputWeights[i]);
        for (int i = 0; i < inputSize; i++) free(nn->hiddenWeights[i]);
        free(nn->outputWeights);
        free(nn->hiddenWeights);
        free(nn->hiddenBias);
        free(nn->outputBias);
        free(nn);
        return NULL;
    }

    for (int i = 0; i < hiddenSize; i++) {
        nn->hiddenBias[i] = ((double)rand() / RAND_MAX) * 2 - 1;  // Random initialization between -1 and 1
    }
    for (int i = 0; i < outputSize; i++) {
        nn->outputBias[i] = ((double)rand() / RAND_MAX) * 2 - 1;  // Random initialization between -1 and 1
    }

    return nn;
}

// Perform forward propagation through the network
void forwardPropagation(NeuralNetwork *nn, double *input, double *hiddenLayer, double *outputLayer) {
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < nn->hiddenSize; i++) {
            hiddenLayer[i] = 0;
            for (int j = 0; j < nn->inputSize; j++) {
                hiddenLayer[i] += input[j] * nn->hiddenWeights[j][i];
            }
            hiddenLayer[i] = relu(hiddenLayer[i] + nn->hiddenBias[i]);
        }

        #pragma omp for
        for (int i = 0; i < nn->outputSize; i++) {
            outputLayer[i] = 0;
            for (int j = 0; j < nn->hiddenSize; j++) {
                outputLayer[i] += hiddenLayer[j] * nn->outputWeights[j][i];
            }
            outputLayer[i] += nn->outputBias[i];
        }
    }

    softmax(outputLayer, nn->outputSize);
}

// Perform backward propagation to update weights and biases
void backwardPropagation(NeuralNetwork *nn, double *input, double *hiddenLayer, double *outputLayer, int label, double learningRate) {
    double outputError[10] = {0};
    double hiddenError[128];

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < nn->outputSize; i++) {
            outputError[i] = (i == label) ? outputLayer[i] - 1 : outputLayer[i];
        }

        #pragma omp for
        for (int i = 0; i < nn->hiddenSize; i++) {
            hiddenError[i] = 0;
            for (int j = 0; j < nn->outputSize; j++) {
                hiddenError[i] += outputError[j] * nn->outputWeights[i][j];
            }
            hiddenError[i] *= reluDerivative(hiddenLayer[i]);
        }

        #pragma omp for collapse(2)
        for (int i = 0; i < nn->hiddenSize; i++) {
            for (int j = 0; j < nn->outputSize; j++) {
                nn->outputWeights[i][j] -= learningRate * outputError[j] * hiddenLayer[i];
            }
        }

        #pragma omp for collapse(2)
        for (int i = 0; i < nn->inputSize; i++) {
            for (int j = 0; j < nn->hiddenSize; j++) {
                nn->hiddenWeights[i][j] -= learningRate * hiddenError[j] * input[i];
            }
        }

        #pragma omp for
        for (int i = 0; i < nn->outputSize; i++) {
            nn->outputBias[i] -= learningRate * outputError[i];
        }

        #pragma omp for
        for (int i = 0; i < nn->hiddenSize; i++) {
            nn->hiddenBias[i] -= learningRate * hiddenError[i];
        }
    }
}

// Train the neural network
void trainNetwork(NeuralNetwork *nn, double **trainingData, int *labels, int numSamples, int epochs, double learningRate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalLoss = 0.0;

        #pragma omp parallel reduction(+:totalLoss)
        {
            double hiddenLayer[128];
            double outputLayer[10];

            #pragma omp for
            for (int i = 0; i < numSamples; i++) {
                if (trainingData[i] == NULL) {
                    continue;
                }
                forwardPropagation(nn, trainingData[i], hiddenLayer, outputLayer);

                if (labels[i] < 0 || labels[i] >= nn->outputSize) {
                    continue;
                }

                totalLoss -= log(outputLayer[labels[i]]);

                backwardPropagation(nn, trainingData[i], hiddenLayer, outputLayer, labels[i], learningRate);
            }
        }

        printf("Epoch %d/%d completed, Average Loss: %f\n", epoch + 1, epochs, totalLoss / numSamples);
    }
}

// Freeing memory allocated for the neural network
void freeNeuralNetwork(NeuralNetwork *nn) {
    if (nn == NULL) return;

    for (int i = 0; i < nn->inputSize; i++) {
        free(nn->hiddenWeights[i]);
    }
    free(nn->hiddenWeights);

    for (int i = 0; i < nn->hiddenSize; i++) {
        free(nn->outputWeights[i]);
    }
    free(nn->outputWeights);

    free(nn->hiddenBias);
    free(nn->outputBias);
    free(nn);
}
