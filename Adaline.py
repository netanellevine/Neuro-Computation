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
        train_length = len(X)
        for j in range(num_of_iterations):
            E = 0
            for i in range(train_length):
                y_in = bias + self._weights[0] * X[i].x + self._weights[1] * X[i].y
                t = Y
                diff = (t - y_in)
                if (t - y_in) > 0.0000001:
                    self._weights[0] += alpha * diff * X[i].x
                    self._weights[1] += alpha * diff * X[i].y
                    bias += alpha * diff
                    E += diff ** 2
            MSE = E / train_length
            if MSE < 0.0000001:
                print(j, f'last MSE: {MSE:.14f}')
                break
            # print(f'{MSE:.14f}')
        return self._weights

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            res = self._weights[0] * X[i].x + self._weights[1] * X[i].y
            if res > 0:
                predictions.append(1)
            else:
                predictions.append(-1)
        return predictions


def score(predictions, answers):
    correct = sum(i == answers[i] for i in predictions)
    return correct / len(predictions)