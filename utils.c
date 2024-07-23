#include <linux/limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

#define MAX_LINE_LENGTH 10000

void loadMNISTData(const char *trainFile, const char *testFile, double **trainingImages, int *trainingLabels, double **testImages, int *testLabels, int trainingSize, int testSize)
{
    FILE *file;
    char line[MAX_LINE_LENGTH];
    char *token;
    int i , j;

    file = fopen(trainFile, "r");
    if(file == NULL)
        {
            printf("Error opening training file\n");
            exit(1);
        }

    fgets(line, MAX_LINE_LENGTH, file);

    for(int i; i<trainingSize; i++)
        {
            fgets(line, MAX_LINE_LENGTH, file);

            token = strtok(line, ",");
            trainingLabels [i] = atoi(token);

            for(int j = 0; j < 784; j++)
                {
                    token = strtok(NULL, ",");
                    trainingImages[i][j] = atof(token) / 255.0;
                }
        }

    fclose(file);

    file = fopen(testFile, "r");
    if(file == NULL)
        {
            printf("Error opening test file\n");
            exit(1);
        }

    fgets(line, MAX_LINE_LENGTH, file);

    for(int i = 0; i < testSize; i++)
        {
            fgets(line, MAX_LINE_LENGTH, file);

            for(int j = 0; j < 784; j++)
                {
                    token = strtok(j == 0 ? line : NULL, ",");
                    testImages[i][j] = atof(token) / 255.0;
                }
            testLabels[i] = -1;
        }
    fclose(file);


    printf("MNIST data loaded from CSV files\n");

}
