#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "neuralNetwork.h"
#include "utils.h"

#define TRAINING_SAMPLES 60000
#define TEST_SAMPLES 10000

int main()
{
    srand(time(NULL));
    // Loading MNIST Data
    double **trainingImages = (double **)malloc(TRAINING_SAMPLES * sizeof(double *));
    int *trainingLabels = (int *)malloc(TRAINING_SAMPLES * sizeof(int));
    double **testImages = (double **)malloc(TEST_SAMPLES * sizeof(double *));
    int *testLabels = (int*)malloc(TEST_SAMPLES * sizeof(int));

    for(int i = 0; i<TRAINING_SAMPLES; i++)
        {
            trainingImages[i] = (double *)malloc(784 * sizeof(double));
        }
    for(int i = 0; i<TEST_SAMPLES; i++)
        {
            testImages[i] = (double *)malloc(784 * sizeof(double));
        }

    loadMNISTData(trainingImages, trainingLabels, testImages, testLabels);

    NeuralNetwork *nn = createNeuralNetwork(784, 128, 10);

    trainNetwork(nn, trainingImages, trainingLabels, 10, 0.1);

    // Test the network
    double accuracy = testNetwork(nn, testImages, testLabels, TEST_SAMPLES );
    printf("Test Accuracy : %.2f%%", accuracy * 100);

    freeNeuralNetwork(nn);

    for(int i = 0; i<TRAINING_SAMPLES; i++)
        {
            free(trainingImages[i]);
        }
    for(int i = 0; i<TEST_SAMPLES; i++)
        {
            free(testImages[i]);
        }
    free(trainingImages);
    free(trainingLabels);
    free(testImages);
    free(testLabels);

    return 0;
}
