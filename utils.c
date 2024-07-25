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

    file = fopen(trainFile, "r");
    if(file == NULL)
    {
        fprintf(stderr, "Error opening training file\n");
        exit(1);
    }

    if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
        fprintf(stderr, "Error reading header from training file\n");
        fclose(file);
        exit(1);
    }

    for(int i = 0; i < trainingSize; i++)
    {
        if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
            fprintf(stderr, "Error reading line %d from training file\n", i+1);
            fclose(file);
            exit(1);
        }

        token = strtok(line, ",");
        if (token == NULL) {
            fprintf(stderr, "Error parsing label in line %d of training file\n", i+1);
            fclose(file);
            exit(1);
        }
        trainingLabels[i] = atoi(token);

        for(int j = 0; j < 784; j++)
        {
            token = strtok(NULL, ",");
            if (token == NULL) {
                fprintf(stderr, "Error parsing pixel %d in line %d of training file\n", j+1, i+1);
                fclose(file);
                exit(1);
            }
            trainingImages[i][j] = atof(token) / 255.0;
        }
    }
    fclose(file);

    file = fopen(testFile, "r");
        if(file == NULL)
        {
            fprintf(stderr, "Error opening test file\n");
            exit(1);
        }

        if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
            fprintf(stderr, "Error reading header from test file\n");
            fclose(file);
            exit(1);
        }

        for(int i = 0; i < testSize; i++)
        {
            if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
                fprintf(stderr, "Error reading line %d from test file\n", i+1);
                fclose(file);
                exit(1);
            }

            for(int j = 0; j < 784; j++)
            {
                token = strtok(j == 0 ? line : NULL, ",");
                if (token == NULL) {
                    fprintf(stderr, "Error parsing pixel %d in line %d of test file\n", j+1, i+1);
                    fclose(file);
                    exit(1);
                }
                testImages[i][j] = atof(token) / 255.0;
            }
        }
        fclose(file);

        printf("MNIST data loaded from CSV files\n");
}
