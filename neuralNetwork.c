#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "neuralNetwork.h"

#define DEBUG_PRINT(fmt, ...) printf("%s:%d: " fmt "\n", __func__, __LINE__, ##__VA_ARGS__)

double relu(double x)
{
    return x > 0 ? x : 0;
}

double reluDerivative(double x)
{
    return x > 0 ? 1 : 0;
}

void softmax(double* input, int size)
{
    double max = input[0];
    for(int i = 1; i < size; i++)
    {
        if (input[i] > max)
        {
            max = input[i];
        }
    }

    double sum = 0.0;
    for(int i = 0; i < size; i++)
    {
        input[i] = exp(input[i] - max);
        sum += input[i];
    }
    for(int i = 0; i < size; i++)
    {
        input[i] /= sum;
    }
}

NeuralNetwork* createNeuralNetwork(int inputSize, int hiddenSize, int outputSize)
{
    DEBUG_PRINT("Creating neural network with input: %d, hidden: %d, output: %d", inputSize, hiddenSize, outputSize);

    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    if (nn == NULL) {
        DEBUG_PRINT("Failed to allocate memory for neural network");
        return NULL;
    }

    nn->inputSize = inputSize;
    nn->hiddenSize = hiddenSize;
    nn->outputSize = outputSize;

    nn->hiddenWeights = (double **)malloc(inputSize * sizeof(double *));
    if (nn->hiddenWeights == NULL) {
        DEBUG_PRINT("Failed to allocate memory for hidden weights");
        free(nn);
        return NULL;
    }

    for(int i = 0; i < inputSize; i++)
    {
        nn->hiddenWeights[i] = (double *)malloc(hiddenSize * sizeof(double));
        if (nn->hiddenWeights[i] == NULL) {
            DEBUG_PRINT("Failed to allocate memory for hidden weights row %d", i);
            for (int j = 0; j < i; j++) free(nn->hiddenWeights[j]);
            free(nn->hiddenWeights);
            free(nn);
            return NULL;
        }
        for(int j = 0; j < hiddenSize; j++)
        {
            nn->hiddenWeights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }

    nn->outputWeights = (double **)malloc(hiddenSize * sizeof(double *));
    if (nn->outputWeights == NULL) {
        DEBUG_PRINT("Failed to allocate memory for output weights");
        for (int i = 0; i < inputSize; i++) free(nn->hiddenWeights[i]);
        free(nn->hiddenWeights);
        free(nn);
        return NULL;
    }

    for(int i = 0; i < hiddenSize; i++)
    {
        nn->outputWeights[i] = (double *)malloc(outputSize * sizeof(double));
        if (nn->outputWeights[i] == NULL) {
            DEBUG_PRINT("Failed to allocate memory for output weights row %d", i);
            for (int j = 0; j < i; j++) free(nn->outputWeights[j]);
            for (int j = 0; j < inputSize; j++) free(nn->hiddenWeights[j]);
            free(nn->outputWeights);
            free(nn->hiddenWeights);
            free(nn);
            return NULL;
        }
        for(int j = 0; j < outputSize; j++)
        {
            nn->outputWeights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }

    nn->hiddenBias = (double *)malloc(hiddenSize * sizeof(double));
    nn->outputBias = (double *)malloc(outputSize * sizeof(double));

    if (nn->hiddenBias == NULL || nn->outputBias == NULL) {
        DEBUG_PRINT("Failed to allocate memory for biases");
        for (int i = 0; i < hiddenSize; i++) free(nn->outputWeights[i]);
        for (int i = 0; i < inputSize; i++) free(nn->hiddenWeights[i]);
        free(nn->outputWeights);
        free(nn->hiddenWeights);
        free(nn->hiddenBias);
        free(nn->outputBias);
        free(nn);
        return NULL;
    }

    for(int i = 0; i < hiddenSize; i++)
    {
        nn->hiddenBias[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
    for(int i = 0; i < outputSize; i++)
    {
        nn->outputBias[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }

    DEBUG_PRINT("Neural network created successfully");
    return nn;
}

void forwardPropagation(NeuralNetwork *nn, double *input, double *hiddenLayer, double *outputLayer)
{
    DEBUG_PRINT("Entering forwardPropagation");

    for(int i = 0; i < nn->hiddenSize; i++)
    {
        hiddenLayer[i] = 0;
        for(int j = 0; j < nn->inputSize; j++)
        {
            hiddenLayer[i] += input[j] * nn->hiddenWeights[j][i];
        }
        hiddenLayer[i] = relu(hiddenLayer[i] + nn->hiddenBias[i]);
    }
    DEBUG_PRINT("Hidden layer computed");

    for(int i = 0; i < nn->outputSize; i++)
    {
        outputLayer[i] = 0;
        for(int j = 0; j < nn->hiddenSize; j++)
        {
            outputLayer[i] += hiddenLayer[j] * nn->outputWeights[j][i];
        }
        outputLayer[i] += nn->outputBias[i];
    }
    DEBUG_PRINT("Output layer computed");

    softmax(outputLayer, nn->outputSize);
    DEBUG_PRINT("Softmax applied");
}

void backwardPropagation(NeuralNetwork *nn, double *input, double *hiddenLayer, double *outputLayer, int label, double learningRate)
{
    DEBUG_PRINT("Entering backwardPropagation");

    double outputError[10] = {0};
    for(int i = 0; i < nn->outputSize; i++)
    {
        outputError[i] = (i == label) ? outputLayer[i] - 1 : outputLayer[i];
    }

    double hiddenError[128];
    for(int i = 0; i < nn->hiddenSize; i++)
    {
        hiddenError[i] = 0;
        for(int j = 0; j < nn->outputSize; j++)
        {
            hiddenError[i] += outputError[j] * nn->outputWeights[i][j];
        }
        hiddenError[i] *= reluDerivative(hiddenLayer[i]);
    }

    for(int i = 0; i < nn->hiddenSize; i++)
    {
        for(int j = 0; j < nn->outputSize; j++)
        {
            nn->outputWeights[i][j] -= learningRate * outputError[j] * hiddenLayer[i];
        }
    }

    for(int i = 0; i < nn->inputSize; i++)
    {
        for(int j = 0; j < nn->hiddenSize; j++)
        {
            nn->hiddenWeights[i][j] -= learningRate * hiddenError[j] * input[i];
        }
    }

    for(int i = 0; i < nn->outputSize; i++)
    {
        nn->outputBias[i] -= learningRate * outputError[i];
    }
    for(int i = 0; i < nn->hiddenSize; i++)
    {
        nn->hiddenBias[i] -= learningRate * hiddenError[i];
    }

    DEBUG_PRINT("Backward propagation completed");
}

void trainNetwork(NeuralNetwork *nn, double **trainingData, int *labels, int numSamples, int epochs, double learningRate) {
    DEBUG_PRINT("Entering trainNetwork function");
    double hiddenLayer[128];
    double outputLayer[10];

    for (int epoch = 0; epoch < epochs; epoch++) {
        DEBUG_PRINT("Starting epoch %d of %d", epoch + 1, epochs);
        double totalLoss = 0.0;
        for (int i = 0; i < numSamples; i++) {
            DEBUG_PRINT("Processing sample %d of %d in epoch %d", i + 1, numSamples, epoch + 1);
            if (trainingData[i] == NULL) {
                DEBUG_PRINT("Error: NULL input data at sample %d", i);
                return;
            }
            forwardPropagation(nn, trainingData[i], hiddenLayer, outputLayer);

            if (labels[i] < 0 || labels[i] >= nn->outputSize) {
                DEBUG_PRINT("Error: Invalid label %d at sample %d", labels[i], i);
                return;
            }

            totalLoss -= log(outputLayer[labels[i]]);

            backwardPropagation(nn, trainingData[i], hiddenLayer, outputLayer, labels[i], learningRate);

            if (i % 1000 == 0) {
                DEBUG_PRINT("Completed %d samples in epoch %d", i + 1, epoch + 1);
            }
        }
        DEBUG_PRINT("Epoch %d completed, Average Loss: %f", epoch + 1, totalLoss / numSamples);
    }
    DEBUG_PRINT("Training completed");
}

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

    DEBUG_PRINT("Neural network freed");
}
