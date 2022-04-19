import random
import numpy as np
from Adaline import Adaline
from Point import Point
import warnings
warnings.filterwarnings("ignore")


def create_random_points():
    return round(random.randint(-10000, 10000) / 100, 15)


def create_X(num):
    pass


def main():
    data_size = 1000
    num_of_iterations = 1000
    learning_rate = 0.00001

    # Part A:
    X_train = []
    X_test = []
    for _ in range(data_size):
        X_train.append(Point(create_random_points(), create_random_points()))
        X_test.append(Point(create_random_points(), create_random_points()))
    adalineA = Adaline()
    Y_train = [(1 if point.getY() > 1 else -1) for point in X_train]
    adalineA.fit(X_train, Y_train, num_of_iterations, learning_rate)
    Y_test = [(1 if point.getY() > 1 else -1) for point in X_test]
    predictions = adalineA.predict(X_test)
    score = adalineA.score(predictions, Y_test)
    # print("res: ", res, "real value: ", test[i].value)
    print("____________________________________Part A____________________________________")
    print(f'| Data size: {data_size} | Number Of Iterations: {num_of_iterations} | Learning Rate: {learning_rate:.10f} |')
    print(
        f'| Weight of X: {adalineA.getWeights()[0]:.10f} | Weight of Y: {adalineA.getWeights()[1]:.10f}')#| Weight of Bias: {adalineA.getWeights()[2]:.10f} |')
    print(f'Part A score: {score * 100}%')

    print()
    print("__________________________________________________________________________________")
    print("__________________________________________________________________________________")
    print("__________________________________________________________________________________")
    print()

    # Part B:
    X_train = []
    X_test = []
    for _ in range(data_size):
        X_train.append(Point(create_random_points(), create_random_points()))
        X_test.append(Point(create_random_points(), create_random_points()))
    adalineB = Adaline()
    Y_train = [(1 if 4 <= (pow(point.getY(), 2) + pow(point.getX(), 2)) <= 9 else -1) for point in X_train]
    adalineB.fit(X_train, Y_train, num_of_iterations, 0.0001)
    Y_test = [(1 if 4 <= (pow(point.getY(), 2) + pow(point.getX(), 2)) <= 9 else -1) for point in X_test]
    predictions = adalineB.predict(X_test)
    score = adalineB.score(predictions, Y_test)
    # print("res: ", res, "real value: ", test[i].value)
    print("____________________________________Part B____________________________________")
    print(f'| Data size: {data_size} | Number Of Iterations: {num_of_iterations} | Learning Rate: {learning_rate:.10f} |')
    print(
        f'| Weight of X: {adalineB.getWeights()[0]:.10f} | Weight of Y: {adalineB.getWeights()[1]:.10f}')# | Weight of Bias: {adalineB.getWeights()[2]:.10f} |')
    print(f'Part B score: {score * 100}%')


if __name__ == '__main__':
    main()
