#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include "neuralNetwork.h"

// Xavier/Glorot initialization
double xavier_init(int inputs, int outputs) {
    double limit = sqrt(6.0 / (inputs + outputs));
    return ((double)rand() / RAND_MAX) * 2 * limit - limit;
}

// Activation function: ReLU (Rectified Linear Unit)
double relu(double x) {
    return fmax(0, x);
}

// Derivative of ReLU function
double reluDerivative(double x) {
    return x > 0 ? 1 : 0;
}

// Clip gradients
double clip_gradient(double grad, double threshold) {
    return fmax(fmin(grad, threshold), -threshold);
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
            nn->hiddenWeights[i][j] = xavier_init(inputSize, hiddenSize);
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
            nn->outputWeights[i][j] = xavier_init(hiddenSize, outputSize);
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
        nn->hiddenBias[i] = 0;  // Initialize biases to zero
    }
    for (int i = 0; i < outputSize; i++) {
        nn->outputBias[i] = 0;  // Initialize biases to zero
    }

    return nn;
}

// Perform forward propagation through the network with batch normalization
void forwardPropagation(NeuralNetwork *nn, double *input, double *hiddenLayer, double *outputLayer) {
    // Hidden layer with batch normalization
    double hiddenSum[128] = {0};
    double hiddenMean = 0, hiddenVar = 0;
    for (int i = 0; i < nn->hiddenSize; i++) {
        for (int j = 0; j < nn->inputSize; j++) {
            hiddenSum[i] += input[j] * nn->hiddenWeights[j][i];
        }
        hiddenSum[i] += nn->hiddenBias[i];
        hiddenMean += hiddenSum[i];
    }
    hiddenMean /= nn->hiddenSize;
    for (int i = 0; i < nn->hiddenSize; i++) {
        hiddenVar += (hiddenSum[i] - hiddenMean) * (hiddenSum[i] - hiddenMean);
    }
    hiddenVar = sqrt(hiddenVar / nn->hiddenSize + 1e-8);
    for (int i = 0; i < nn->hiddenSize; i++) {
        hiddenLayer[i] = relu((hiddenSum[i] - hiddenMean) / hiddenVar);
    }

    // Output layer
    for (int i = 0; i < nn->outputSize; i++) {
        outputLayer[i] = 0;
        for (int j = 0; j < nn->hiddenSize; j++) {
            outputLayer[i] += hiddenLayer[j] * nn->outputWeights[j][i];
        }
        outputLayer[i] += nn->outputBias[i];
    }
    softmax(outputLayer, nn->outputSize);
}

// Perform backward propagation to update weights and biases with gradient clipping
void backwardPropagation(NeuralNetwork *nn, double *input, double *hiddenLayer, double *outputLayer, int label, double learningRate) {
    // Calculate output layer error
    double outputError[10] = {0};
    for (int i = 0; i < nn->outputSize; i++) {
        outputError[i] = (i == label) ? outputLayer[i] - 1 : outputLayer[i];
    }

    // Calculate hidden layer error
    double hiddenError[128];
    for (int i = 0; i < nn->hiddenSize; i++) {
        hiddenError[i] = 0;
        for (int j = 0; j < nn->outputSize; j++) {
            hiddenError[i] += outputError[j] * nn->outputWeights[i][j];
        }
        hiddenError[i] *= reluDerivative(hiddenLayer[i]);
    }

    // Update output weights with gradient clipping
    double clipThreshold = 1.0;
    for (int i = 0; i < nn->hiddenSize; i++) {
        for (int j = 0; j < nn->outputSize; j++) {
            double grad = clip_gradient(outputError[j] * hiddenLayer[i], clipThreshold);
            nn->outputWeights[i][j] -= learningRate * grad;
        }
    }

    // Update hidden weights with gradient clipping
    for (int i = 0; i < nn->inputSize; i++) {
        for (int j = 0; j < nn->hiddenSize; j++) {
            double grad = clip_gradient(hiddenError[j] * input[i], clipThreshold);
            nn->hiddenWeights[i][j] -= learningRate * grad;
        }
    }

    // Update biases
    for (int i = 0; i < nn->outputSize; i++) {
        nn->outputBias[i] -= learningRate * outputError[i];
    }
    for (int i = 0; i < nn->hiddenSize; i++) {
        nn->hiddenBias[i] -= learningRate * hiddenError[i];
    }
}

// Train the neural network with learning rate decay
void trainNetwork(NeuralNetwork *nn, double **trainingData, int *labels, int numSamples, int epochs, double initialLearningRate) {
    int num_threads = omp_get_max_threads();
    double *thread_losses = (double *)calloc(num_threads, sizeof(double));
    double learningRate = initialLearningRate;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalLoss = 0.0;

        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            double hiddenLayer[128];
            double outputLayer[10];
            double localLoss = 0.0;

            #pragma omp for
            for (int i = 0; i < numSamples; i++) {
                if (trainingData[i] == NULL || labels[i] < 0 || labels[i] >= nn->outputSize) {
                    continue;
                }

                forwardPropagation(nn, trainingData[i], hiddenLayer, outputLayer);

                localLoss -= log(fmax(outputLayer[labels[i]], 1e-10));

                #pragma omp critical
                {
                    backwardPropagation(nn, trainingData[i], hiddenLayer, outputLayer, labels[i], learningRate);
                }
            }

            thread_losses[thread_id] = localLoss;
        }

        for (int i = 0; i < num_threads; i++) {
            totalLoss += thread_losses[i];
            thread_losses[i] = 0.0;
        }

        printf("Epoch %d/%d completed, Average Loss: %f, Learning Rate: %f\n",
               epoch + 1, epochs, totalLoss / numSamples, learningRate);

        // Learning rate decay
        learningRate *= 0.99;
    }

    free(thread_losses);
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
