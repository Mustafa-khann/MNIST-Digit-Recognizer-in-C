#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "neuralNetwork.h"
#include "utils.h"

#define TRAINING_SAMPLES 42000
#define TEST_SAMPLES 28000
#define INPUT_SIZE 784
#define HIDDEN_SIZE 28
#define OUTPUT_SIZE 10

int main()
{
    srand(time(NULL));

    double **trainingImages = (double **)malloc(TRAINING_SAMPLES * sizeof(double *));
    int *trainingLabels = (int *) malloc(TRAINING_SAMPLES * sizeof(int ));
    double **testImages = (double **)malloc(TEST_SAMPLES * sizeof(double *));
    int *testLabels = (int *)malloc(TEST_SAMPLES * sizeof(int));


    for(int i = 0; i<TRAINING_SAMPLES; i++)
        {
            trainingImages[i] = (double *)malloc(INPUT_SIZE * sizeof(double));
        }
    for(int i = 0; i<TEST_SAMPLES; i++)
        {
            testImages[i] = (double *)malloc(INPUT_SIZE * sizeof(double));
        }

    printf("Memory allocated...\n");

    loadMNISTData("dataset/train.csv", "dataset/test.csv", trainingImages, trainingLabels, testImages, testLabels, TRAINING_SAMPLES, TEST_SAMPLES);

    printf("Data loaded, creating neural network...\n");

    NeuralNetwork *nn = createNeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    printf("Neural network created, starting training...\n");

    printf("Starting training with %d samples for %d epochs\n", TRAINING_SAMPLES, 50);
    trainNetwork(nn, trainingImages, trainingLabels, TRAINING_SAMPLES, 50, 0.1);

    printf("Training complete, starting predictions...\n");

    FILE *predictionFile = fopen("sample_submission.csv", "w");
    fprintf(predictionFile, "ImageId,Label\n");


    double hiddenLayer[HIDDEN_SIZE];
    double outputLayer[OUTPUT_SIZE];

    for(int i = 0; i<TEST_SAMPLES; i++)
        {
            forwardPropagation(nn, testImages[i], hiddenLayer, outputLayer);
            int predictedLabel = 0;
            double maxOutput = outputLayer[0];
            for(int j = 0; j<OUTPUT_SIZE; j++)
                {
                    if(outputLayer[j] > maxOutput)
                        {
                            maxOutput = outputLayer[j];
                            predictedLabel = j;
                        }
                }
            fprintf(predictionFile, "%d,%d\n", i+1, predictedLabel);
        }

    fclose(predictionFile);
    printf("Predictions saved to sample_submission.csv");

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
