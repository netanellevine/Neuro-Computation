import numpy as np
import random


class Adaline:
    def __init__(self):
        self._num_of_iterations = 10
        self._alpha = 0.1
        self._weights = []
        self._error = []

    def getAlpha(self):
        return self._alpha

    def getNumOfIterations(self):
        return self._num_of_iterations

    def getWeights(self):
        return self._weights

    def getErrors(self):
        return self._error

    def setAlpha(self, alpha):
        self._alpha = alpha

    def setNumOfIterations(self, num_of_iterations):
        self._num_of_iterations = num_of_iterations

    def setWeights(self, weights):
        self._weights = weights

    def fit(self, X, Y, num_of_iterations=10, alpha=0.001):
        self.setNumOfIterations(num_of_iterations)
        self.setAlpha(alpha)
        self._weights = [0.5, 0.5]
        bias = random.uniform(0.005, 1)
        bias = 0.2
        train_length = len(X)
        for j in range(num_of_iterations):
            E = 0
            for i in range(train_length):
                y_in = bias + self._weights[0] * X[i].getX() + self._weights[1] * X[i].getY()
                diff = Y[i] - y_in
                if diff != 0:
                    self._weights[0] += alpha * diff * X[i].getX()
                    self._weights[1] += alpha * diff * X[i].getY()
                    bias += alpha * diff
                    E += pow(diff, 2)
            MSE = E / train_length
            if MSE < 0.000000000000001:
                print(j, f'last MSE: {MSE:.14f}')
                break
            # print(f'{MSE:.14f}')
        self._weights.append(bias)

    def predict(self, X_test):
        # print(self._weights)
        predictions = []
        for i in range(len(X_test)):
            res = self._weights[0] * X_test[i].getX() + self._weights[1] * X_test[i].getY() + self._weights[2]
            ans = 1 if 4 <= (pow(X_test[i].getY(), 2) + pow(X_test[i].getX(), 2)) <= 9 else -1
            if res > 0.0:
                predictions.append(1)
            else:
                predictions.append(-1)
        return predictions


    def score(self, predictions, answers):
        correct = sum(predictions[i] == answers[i] for i in range(len(predictions)))
        return correct / len(predictions)

