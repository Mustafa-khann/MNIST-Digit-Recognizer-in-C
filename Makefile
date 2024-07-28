CC := gcc
CFLAGS := -O3 -march=native -fopenmp -Wall -Wextra
LDFLAGS := -lm -fopenmp

SRCS := main.c neuralNetwork.c utils.c
TARGET := mnist_recognizer

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

build:
	gcc main.c neuralNetwork.c utils.c -o mnist_recognizer -fopenmp -O3 -march=native -lm

clean:
	rm -f $(TARGET) sample_submission.csv
