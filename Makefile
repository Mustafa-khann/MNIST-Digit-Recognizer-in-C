build:
	gcc main.c neuralNetwork.c utils.c -o mnist_recognizer -lm

clean:
	rm mnist_recognizer sample_submission.csv
