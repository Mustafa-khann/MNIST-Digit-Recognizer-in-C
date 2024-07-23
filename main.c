#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "neuralNetwork.h"
#include "utils.h"

#define TRAINING_SAMPLES 42000
#define TEST_SAMPLES 28000
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define EPOCHS 50
#define LEARNING_RATE 0.1

double calculateAccuracy(NeuralNetwork *nn, double **images, int *labels, int numSamples) {
    int correct = 0;
    double hiddenLayer[HIDDEN_SIZE];
    double outputLayer[OUTPUT_SIZE];

    for (int i = 0; i < numSamples; i++) {
        forwardPropagation(nn, images[i], hiddenLayer, outputLayer);
        int predictedLabel = 0;
        double maxOutput = outputLayer[0];
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (outputLayer[j] > maxOutput) {
                maxOutput = outputLayer[j];
                predictedLabel = j;
            }
        }
        if (predictedLabel == labels[i]) {
            correct++;
        }
    }

    return (double)correct / numSamples;
}

int main() {
    srand(time(NULL));

    // Allocate memory
    double **trainingImages = (double **)malloc(TRAINING_SAMPLES * sizeof(double *));
    int *trainingLabels = (int *)malloc(TRAINING_SAMPLES * sizeof(int));
    double **testImages = (double **)malloc(TEST_SAMPLES * sizeof(double *));
    int *testLabels = (int *)malloc(TEST_SAMPLES * sizeof(int));

    for (int i = 0; i < TRAINING_SAMPLES; i++) {
        trainingImages[i] = (double *)malloc(INPUT_SIZE * sizeof(double));
    }
    for (int i = 0; i < TEST_SAMPLES; i++) {
        testImages[i] = (double *)malloc(INPUT_SIZE * sizeof(double));
    }

    printf("Memory allocated...\n");

    // Load data
    loadMNISTData("dataset/train.csv", "dataset/test.csv", trainingImages, trainingLabels, testImages, testLabels, TRAINING_SAMPLES, TEST_SAMPLES);
    printf("Data loaded, creating neural network...\n");

    // Create and train neural network
    NeuralNetwork *nn = createNeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("Neural network created, starting training...\n");
    trainNetwork(nn, trainingImages, trainingLabels, TRAINING_SAMPLES, EPOCHS, LEARNING_RATE);
    printf("Training complete.\n");

    // Calculate and print accuracies
    double trainingAccuracy = calculateAccuracy(nn, trainingImages, trainingLabels, TRAINING_SAMPLES);
    printf("Training accuracy: %.2f%%\n", trainingAccuracy * 100);

    double testAccuracy = calculateAccuracy(nn, testImages, testLabels, TEST_SAMPLES);
    printf("Test accuracy: %.2f%%\n", testAccuracy * 100);

    // Free memory
    freeNeuralNetwork(nn);
    for (int i = 0; i < TRAINING_SAMPLES; i++) {
        free(trainingImages[i]);
    }
    for (int i = 0; i < TEST_SAMPLES; i++) {
        free(testImages[i]);
    }
    free(trainingImages);
    free(trainingLabels);
    free(testImages);
    free(testLabels);

    return 0;
}
