import random
import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.value = 1 if y > 1 else -1


def adaline(train) -> tuple:  # w1, w2
    w = np.array([0.5, 0.5])
    b = random.uniform(0.005, 1)
    alpha = random.uniform(0.005, 1)
    alpha = 0.0001
    train_length = len(train)
    for j in range(5000):
        E = 0
        for i in range(train_length):
            y_in = b + w[0] * train[i].x + w[1] * train[i].y
            t = train[i].value
            diff = (t - y_in)
            if (t - y_in) > 0.0000001:
                w[0] += alpha * diff * train[i].x
                w[1] += alpha * diff * train[i].y
                # print("x weight: ", w[0])
                # print("y weight: ", w[1])
                b += alpha * diff
                E += diff ** 2
        MSE = E / train_length
        if MSE < 0.0000001:
            print(j, f'last MSE: {MSE:.14f}')
            break

        print(f'{MSE:.14f}')
    return w[0], w[1]


def create_random_points():
    return round(random.randint(-10000, 10000) / 100, 8)


def main():
    train = []
    test = []
    for _ in range(1000):
        x = create_random_points()
        y = create_random_points()
        train.append(Point(x, y))
        x = create_random_points()
        y = create_random_points()
        test.append(Point(x, y))
    w = adaline(train)
    count_success = 0
    # print(w)

    for i in range(100):
        res = w[0] * test[i].x + w[1] * test[i].y
        if res > 0 and test[i].value == 1:
            count_success += 1
        elif res < 0 and test[i].value == -1:
            count_success += 1
        # print("res: ", res, "real value: ", test[i].value)

    print(count_success / 100)


if __name__ == '__main__':
    main()
