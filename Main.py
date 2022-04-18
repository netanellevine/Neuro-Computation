import random
import numpy as np
from Adaline import Adaline
from Point import Point
import warnings
warnings.filterwarnings("ignore")


def create_random_points():
    return round(random.randint(-10000, 10000) / 100, 8)


def main():
    X_train = []
    X_test = []
    for _ in range(1000):
        X_train.append(Point(create_random_points(), create_random_points()))
        X_test.append(Point(create_random_points(), create_random_points()))
    adaline = Adaline()
    Y_train = [(1 if point.getY() > 1 else -1) for point in X_train]
    adaline.fit(X_train, Y_train, 1000, 0.00001)
    Y_test = [(1 if point.getY() > 1 else -1) for point in X_test]
    predictions = adaline.predict(X_test)
    score = adaline.score(predictions, Y_test)
    # print("res: ", res, "real value: ", test[i].value)

    print(score)


if __name__ == '__main__':
    main()
