#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

typedef struct {
    int inputSize;
    int hiddenSize;
    int outputSize;
    double **hiddenWeights;
    double **outputWeights;
    double *hiddenBias;
    double *outputBias;
} NeuralNetwork;

NeuralNetwork* createNeuralNetwork(int inputSize, int hiddenSize, int outputSize);

void forwardPropagation(NeuralNetwork *nn, double *input, double *hiddenLayer, double *outputLayer);

void backwardPropagation(NeuralNetwork *nn, double *input, double *hiddenLayer, double *outputLayer, int label, double learningRate);

void trainNetwork(NeuralNetwork *nn, double **trainingData, int *labels, int numSamples, int epochs, double learningRate);

double testNetwork(NeuralNetwork *nn, double **testingData, int *labels, int numSamples);

void freeNeuralNetwork(NeuralNetwork *nn);

#endif
