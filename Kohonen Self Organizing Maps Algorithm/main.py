import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from kohonen import Kohonen
import cv2 as cv
def Part_A_1():
    points = []
    for i in range(1000):
        points.append([random.uniform(0, 1), random.uniform(0, 1)])
    layers = (np.ones(10) * 10).astype(int)
    ko = Kohonen()
    ko.fit(points)
    ko2 = Kohonen(neurons_amount=layers)
    ko2.fit(points)
    return

def Part_A_2():
    points = []
    for i in range(1000):
        points.append([random.gauss(0.5, 0.15), random.uniform(0, 1)])
    layers = (np.ones(10) * 10).astype(int)
    ko = Kohonen(neurons_amount=layers)
    ko.fit(points)

    points = []
    for i in range(1000):
        points.append([random.gauss(0.5, 0.15), random.gauss(0.5, 0.15)])
    ko = Kohonen(neurons_amount=layers)
    ko.fit(points)
    return

def Part_A_3():
    points = []
    for i in range(7000):
        points.append([random.uniform(0, 1), random.uniform(0, 1)])
    circle = []
    for p in points:
        if 0.15**2 <= (p[0] - 0.5)**2 + (p[1]-0.5)**2 <= 0.3**2:
            circle.append(p)
    layers = (np.ones(10) * 10).astype(int)
    ko = Kohonen(neurons_amount=[30], learning_rate=0.05)
    ko.fit(circle, iteration=10000)
    ko2 = Kohonen(neurons_amount=layers, learning_rate=0.5)
    ko2.fit(circle)
    pass

def Part_B():
    hand = cv.imread("hand.jpg")
    hand = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
    hand = cv2.resize(hand, (0,0), fx=0.5, fy=0.5)

    points = np.argwhere(hand != 255).astype(np.float32)
    plt.show()
    max = points.max(axis=0)
    max = max*1.0
    points[:, 0] = points[:, 0] / max[0]
    points[:, 1] = points[:, 1] / max[1]
    print(len(points))
    shape = hand.shape
    print(points.max(axis=0), shape)
    layers = (np.ones(15) * 15).astype(int)
    ko = Kohonen(neurons_amount=layers,learning_rate=0.4)
    ko.fit(points, iteration=10000)

    hand2 = cv.imread("80%_hand.jpg")
    hand2 = cv2.cvtColor(hand2, cv2.COLOR_BGR2GRAY)
    hand2 = cv2.resize(hand2, (0, 0), fx=0.5, fy=0.5)
    points2 = np.argwhere(hand2 != 255).astype(np.float32)
    max = points2.max(axis=0)
    max = max * 1.0
    points2[:, 0] = points2[:, 0] / max[0]
    points2[:, 1] = points2[:, 1] / max[1]
    ko.refit(points2)

    plt.imshow(hand2)
    pass

if __name__ == '__main__':
    # Part_A_1()
    # Part_A_2()
    # Part_A_3()
    Part_B()