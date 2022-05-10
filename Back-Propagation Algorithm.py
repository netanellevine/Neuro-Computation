import numpy as np
import seaborn as sns
from numpy import random
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._base import ACTIVATIONS
from Point import Point
from Adaline import Adaline
import math
data_size = 10000

def partC(X_train, y_train, X_test, y_test, model):
    print("\nPart C\n")
    print("Score of correct prediction: ", model.score(X_test, y_test) * 100, "%")
    #diagram(model, X_train)
    show_area(X_test, y_test, model)
    get_cm(X_test, y_test, model)


def create_random_points(data_size, n):
    points = np.empty((data_size, 2), dtype=object)
    random.seed(7)
    for i in range(data_size // 2):
        points[i, 0] = random.randint(-300, 300) / 100
        points[i, 1] = random.randint(-300, 300) / 100

    for i in range(data_size // 2, data_size):
        points[i, 0] = random.randint(-n, n) / 100
        points[i, 1] = random.randint(-n, n) / 100

    targets = np.array([(1 if 4 <= (pow(points[i][0], 2) + pow(points[i][1], 2)) <= 9 else -1) for i in range(data_size)])
    features = points.astype(np.float64)
    targets = targets.astype(np.float64)

    return features, targets


def diagram(model, X_train):
    for layer_index in range(1, model.n_layers_):
        layer = get_layer(model, X_train, layer_index)
        neuron_index = 1
        for neuron in layer:
            plt.scatter(x=X_train[neuron == -1, 1], y=X_train[neuron == -1, 0], c='red', label=-1.0)
            plt.scatter(x=X_train[neuron == 1, 1], y=X_train[neuron == 1, 0], c='green', label=1.0)
            plt.title("Layer index: " + str(layer_index) + " Neuron index: " + str(neuron_index))
            plt.show()
            neuron_index += 1


def get_layer(model, X, layer_index):
    if layer_index == 0:
        layer_index = model.n_layers_
    features = X
    act = ACTIVATIONS[model.activation]
    for i in range(layer_index - 1):
        weight_i, bias_i = model.coefs_[i], model.intercepts_[i]
        features = np.dot(features, weight_i) + bias_i
        if i != layer_index - 2:
            act(features)

    # if the layer has more than one neuron
    if features.shape[1] > 1:
        neurons = []
        for j in range(features.shape[1]):
            neurons.append(model._label_binarizer.inverse_transform(features[:, j]))
        return neurons
    act(features)
    # if the layer has only one neuron
    neuron = model._label_binarizer.inverse_transform(features)
    return neuron


def show_area(X_test, y_test, model):
    x_min, y_min = X_test[:, 0].min() - 1, X_test[:, 1].min() - 1
    x_max, y_max = X_test[:, 0].max() + 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    pred = model.predict(np.array([xx.flatten(), yy.flatten()]).T)
    pred = pred.reshape(xx.shape)
    colors = ListedColormap(('red', 'green'))
    plt.contourf(xx, yy, pred, cmap=colors, alpha=0.5)
    # limits of the data plotted
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.scatter(x=X_test[y_test == -1, 1], y=X_test[y_test == -1, 0], c='red', alpha=0.5, label=-1.0)
    plt.scatter(x=X_test[y_test == 1, 1], y=X_test[y_test == 1, 0], c='green', alpha=0.5, label=1.0)
    plt.title("Part C: Back propagation using MLP Algorithm")
    plt.show()


def get_cm(X_test, y_test, model):
    c_m = confusion_matrix(model.predict(X_test), y_test)
    plt.subplots()
    sns.heatmap(c_m, annot=True, fmt=".0f", cmap="Blues")
    plt.title("Confusion matrix")
    plt.xlabel("Actual")
    plt.ylabel("Predict")
    plt.show()


def partD(X_train, y_train, X_test, y_test, model):
    print("\nPart D\n")
    adaline = Adaline()
    Adaline_with_MLP(adaline, model, X_train, y_train, X_test, y_test)



def Adaline_with_MLP(adaline, MLP, X_train, y_train, X_test, y_test):
    last_hidden_layer = get_layer(MLP, X_train, MLP.n_layers_ - 1)
    X_train = []
    for i in range(len(last_hidden_layer[0])):
        X_train.append(Point(last_hidden_layer[0][i], last_hidden_layer[1][i]))

    temp = X_test.copy()
    X_test = []
    print(temp.shape)
    for i in range(temp.shape[0]):
        X_test.append(Point(temp[i][0], temp[i][1]))

    # adaline.fit(X_train, y_train)

    # avg_bias = (MLP.coefs_[-1][0] + MLP.coefs_[-1][1])/2
    # weights = np.zeros(3)
    # weights[0] = MLP.intercepts_[1][0]
    # weights[1] = MLP.intercepts_[1][1]
    # weights[2] = (avg_bias)
    # adaline.setWeights(weights)

    predictions = adaline.predict(X_test)
    score = adaline.score(predictions, y_test)
    print(f'Adaline with MLP score: {score * 100}%')
    # confusion matrix
    cm = confusion_matrix(predictions, y_test)
    plt.subplots()
    sns.heatmap(cm, fmt=".0f", annot=True, cmap="Blues")
    plt.title("Confusion matrix - Adaline")
    plt.xlabel("Actual")
    plt.ylabel("Predict")
    plt.show()

if __name__ == '__main__':
    # part C
    X_train, y_train = create_random_points(data_size, 10000)
    X_test, y_test = create_random_points(data_size, 10000)
    classifier_mlp = MLPClassifier(activation='logistic', learning_rate_init=0.1,
                                   hidden_layer_sizes=(8, 2), random_state=7)
    classifier_mlp.fit(X_train, y_train)
    #partC(X_train, y_train, X_test, y_test, classifier_mlp)

    partD(X_train, y_train, X_test, y_test, classifier_mlp)