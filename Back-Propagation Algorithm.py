import numpy as np
import seaborn as sb
from numpy import random
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from matplotlib.colors import ListedColormap as LCM
from sklearn.neural_network._base import ACTIVATIONS as Act
from mlxtend.classifier import Adaline
data_size = 1000


def partC(X_train, y_train, X_test, y_test, model):
    print("\nPart C\n")
    print("Score of correct prediction: ", model.score(X_test, y_test) * 100, "%")
    diagram(model, X_train)  # present the geometric diagrams
    show_area(X_test, y_test, model)  # present the final test neurons distribution
    get_cm(model.predict(X_test), y_test)  # present the confusion matrix of the test part


# Creating random data
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


# This function will present the geometric diagram of the model
def diagram(model, X_train):
    for layer_index in range(1, model.n_layers_):  # walking through all of the neurons layers
        layer = get_layer(model, X_train, layer_index)  # getting the current layer
        neuron_index = 1
        for neuron in layer:  # walking through every neuron
            # plotting the neuron data prediction
            plt.scatter(x=X_train[neuron == -1, 1], y=X_train[neuron == -1, 0], c='red', label=-1.0)
            plt.scatter(x=X_train[neuron == 1, 1], y=X_train[neuron == 1, 0], c='green', label=1.0)
            plt.title("Layer index: " + str(layer_index) + " Neuron index: " + str(neuron_index))
            plt.show()
            neuron_index += 1


# This function will return the neurons in the current layer_index
def get_layer(model, X, layer_index):
    features = X
    act = Act[model.activation]  # getting the model activation function
    for i in range(layer_index - 1):  # walking through each layer until the layer_index-1
        weight_i, bias_i = model.coefs_[i], model.intercepts_[i]  # getting the weights and bias of each layer
        features = np.dot(features, weight_i) + bias_i  # multiplying each point by the weights using dot product and adding the bias
        if i != layer_index - 2:
            act(features)  # calculating the activation function to everything but the last layer_index
    if features.shape[1] > 1:  # if the layer has more than one neuron
        neurons = []
        for j in range(features.shape[1]):
            curr_neuron = model._label_binarizer.inverse_transform(features[:, j])  # getting the current neuron
            neurons.append(curr_neuron)
        return neurons
    # if the layer has only one neuron
    act(features)  # calculating the last layer activation function
    neuron = model._label_binarizer.inverse_transform(features)  # getting the only level neuron
    return neuron


# Presenting the final test neurons distribution for part C
def show_area(X_test, y_test, model):
    x_min, y_min = X_test[:, 0].min() - 1, X_test[:, 1].min() - 1
    x_max, y_max = X_test[:, 0].max() + 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))  # getting the axis coordinates
    pred = model.predict(np.array([xx.flatten(), yy.flatten()]).T)
    pred = pred.reshape(xx.shape)
    colors = LCM(('red', 'green'))
    plt.contourf(xx, yy, pred, cmap=colors, alpha=0.5)  # draw contour lines and filled contours, respectively
    # limits of the data plotted
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.scatter(x=X_test[y_test == -1, 1], y=X_test[y_test == -1, 0], c='red', alpha=0.5, label=-1.0)
    plt.scatter(x=X_test[y_test == 1, 1], y=X_test[y_test == 1, 0], c='green', alpha=0.5, label=1.0)
    plt.title("Part C: Back propagation using MLP Algorithm")
    plt.show()


# Presenting the confusion matrix
def get_cm(predictions, y_test):
    c_m = confusion_matrix(predictions, y_test)
    plt.subplots()
    sb.heatmap(c_m, annot=True, fmt=".0f", cmap="Blues")
    plt.title("Confusion matrix")
    plt.xlabel("Actual")
    plt.ylabel("Predict")
    plt.show()


def partD(X_train, y_train, X_test, y_test, model):
    print("\nPart D\n")
    adaline = Adaline(epochs=2)
    Adaline_with_MLP(adaline, model, X_train, y_train, X_test, y_test)


def Adaline_with_MLP(adaline, MLP, X_train, y_train, X_test, y_test):
    X_train = get_last_bp_layer_for_adaline(MLP, X_train)  # calculating the last layer output with the training data
    y_train[y_train == -1] = 0  # changing to 0 because the packaged adaline can have 0 and 1
    y_train = y_train.astype(int)
    adaline.fit(X_train, y_train)  # fitting the training data
    X_test_copy = X_test
    X_test = get_last_bp_layer_for_adaline(MLP, X_test)
    y_test[y_test == -1] = 0  # changing to 0 because the packaged adaline can have 0 and 1
    y_test = y_test.astype(int)
    score = adaline.score(X_test, y_test)
    print(f'Adaline with MLP test score: {score * 100}%')
    predictions = adaline.predict(X_test)
    get_cm(predictions, y_test)  # presenting the confusion matrix
    show_areaD(X_test_copy, X_test, y_test, adaline, MLP)


# Calculating the last layer output with the training data
def get_last_bp_layer_for_adaline(model, X):
    features = X
    act = Act[model.activation]  # getting the model activation function
    for i in range(model.n_layers_ - 1):  # walking through each layer until the layer_index-1
        weights_i, bias_i = model.coefs_[i], model.intercepts_[i]  # getting the weights and bias of each layer
        features = np.dot(features, weights_i) + bias_i  # multiplying each point by the weights using dot product and adding the bias
        act(features)  # calculating the activation function to everything but the last layer_index
    return features


# Presenting the final test neurons distribution for part D
def show_areaD(X_test_orig, X_test, y_test, model, mlp):
    print(X_test_orig)
    x_min, y_min = X_test_orig[:, 0].min() - 1, X_test_orig[:, 1].min() - 1
    x_max, y_max = X_test_orig[:, 0].max() + 1, X_test_orig[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))  # getting the axis coordinates
    lss = get_last_bp_layer_for_adaline(mlp, np.array([xx.flatten(), yy.flatten()]).T)  # calculating the last layer output with the training data
    pred = model.predict(lss)
    pred = pred.reshape(xx.shape)
    colors = LCM(('red', 'green'))
    plt.contourf(xx, yy, pred, cmap=colors, alpha=0.5)  # draw contour lines and filled contours, respectively
    # limits of the data plotted
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.scatter(x=X_test_orig[y_test == 0, 1], y=X_test_orig[y_test == 0, 0], c='red', alpha=0.5, label=-1.0)
    plt.scatter(x=X_test_orig[y_test == 1, 1], y=X_test_orig[y_test == 1, 0], c='green', alpha=0.5, label=1.0)
    plt.title("Part D: Back propagation using MLP Algorithm")
    plt.show()


if __name__ == '__main__':
    X_train, y_train = create_random_points(data_size, 10000)
    X_test, y_test = create_random_points(data_size, 10000)
    classifier_mlp = MLPClassifier(activation='logistic', learning_rate_init=0.1, hidden_layer_sizes=(8, 2), random_state=7)
    classifier_mlp.fit(X_train, y_train)
    partC(X_train, y_train, X_test, y_test, classifier_mlp)
    partD(X_train, y_train, X_test, y_test, classifier_mlp)
