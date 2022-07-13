import random
import numpy as np
import matplotlib.pyplot as plt


class Kohonen:
    """
    This class represent Self Organizing Map that implements the Kohonen algorithm.
    """

    def __init__(self, learning_rate=0.1, neurons_amount=[100]):
        """

        :param learning_rate: According to this parameter the network calculates how much the neurons should change
                             (the radius is getting learning_rate in each iteration of the fit function).
        :param neurons_amount: An array that represent the layers of the network such that the length of it is the
                                number of layers and the i'th value is the number of neurons in the i'th level.
        :param data: (2D array) The data that the network will fit to.
        :param neurons: (3D array) The  first dimension contains n arrays the represent layers, each layer contains
                        m arrays that represent neurons and each neurone contains the weights of this neurone.
                        (the neurons length is equal to the length of the data instances)
        :param radius: A tuning parameter, according to it the network calculates which neurons should change and
                      how much.(the radius is getting smaller in each iteration of the fit function).
        :param lamda: A time constant that used to the determine how to decrease the radius and the learning rate
                        in each iteration.
        """
        self.learning_rate = learning_rate
        self.neurons_amount = neurons_amount
        self.data = None
        self.neurons = []
        self.radius = max(self.neurons_amount[0], len(self.neurons_amount)) / 2
        self.lamda = None

    def fit(self, data_set, iteration=10000):
        """
        This function fits the network to a given data. at first the functions initialize the neurons in random weights.
        then in each iteration the function choose a random vector from the data and check which neuron is the closest
        to the vector using Euclidean distance. according to the closest neuron, the learning rate and the radius the
        function compute how much each neuron should move towards the vector.
        :param data_set: (2D array) Array that contain all the data vectors.
        :param iteration: The number of the iteration to repeat the steps that mentioned above.
        :return:
        """
        # initializing the data and time constant.
        self.data = np.array(data_set)
        self.lamda = iteration / np.log(self.radius)

        # initializing the neurons with random weights.
        for layer in range(len(self.neurons_amount)):
            self.neurons.append([])
            for n in range(self.neurons_amount[layer]):
                weights = []
                for t in range(len(data_set[0])):
                    weights.append(random.uniform(0, 1))
                self.neurons[layer].append(weights)
        self.neurons = np.array(self.neurons)
        # starting the fitting process.
        for i in range(iteration):
            # selecting random vector from the given data.
            vec = self.data[int(random.uniform(0, len(self.data)))]
            # find which neuron is the closest to the vector.
            nn = self.nearest_neuron(vec)
            # updating the learning rate and the radius.
            curr_learning_rate = self.learning_rate * np.exp(-i / self.lamda)
            curr_radius = self.radius * np.exp(-i / self.lamda)
            # going over the neurons in each layer and compute how to change each neuron.
            for j in range(len(self.neurons)):
                for n in range(len(self.neurons[j])):
                    curr_neuron = self.neurons[j][n]
                    d = np.linalg.norm(np.array(nn) - np.array([j, n]))
                    neighbourhood = np.exp(- (d ** 2) / (2 * (curr_radius ** 2)))
                    self.neurons[j][n] += curr_learning_rate * neighbourhood * (vec - curr_neuron)
            # plotting the network to track progress.
            if (i % 1000 == 0) or i == iteration - 1:
                if len(self.neurons_amount) == 1:
                    self.plot1D(i)
                else:
                    self.plot2D(i)

    def refit(self, data, iteration=1000):
        """
        This function do the same as the fit function without initializing the neurons with random weights.
        :param data:
        :param iteration:
        :return:
        """
        self.data = np.array(data)
        self.lamda = iteration / np.log(self.radius)
        for i in range(iteration):
            vec = self.data[int(random.uniform(0, len(self.data)))]
            nn = self.nearest_neuron(vec)
            # nearest_n = self.neurons[nn[0]][nn[1]]
            curr_learning_rate = self.learning_rate * np.exp(-i / self.lamda)
            curr_radius = self.radius * np.exp(-i / self.lamda)

            for j in range(len(self.neurons)):
                for n in range(len(self.neurons[j])):
                    curr_neuron = self.neurons[j][n]
                    d = np.linalg.norm(np.array(nn) - np.array([j, n]))
                    neighbourhood = np.exp(- (d ** 2) / (2 * (curr_radius ** 2)))
                    self.neurons[j][n] += curr_learning_rate * neighbourhood * (
                                vec - curr_neuron)  # dist(curr_neuron, vec)
            if (i % 1000 == 0) or i == iteration - 1:
                if len(self.neurons_amount) == 1:
                    self.plot1D(i)
                else:
                    self.plot2D(i)

    def nearest_neuron(self, vec):
        """
        This function checks find which neuron is the closest to a given vector using Euclidean distance.
        :param vec: Input vector
        :return: A tuple that contains the index of the closest neuron to the vector in self.neurons array.
        """
        min_dist = np.inf
        loc = None
        for i in range(len(self.neurons)):
            for n in range(len(self.neurons[i])):
                curr_neuron = self.neurons[i][n]
                curr_dist = dist(curr_neuron, vec)
                if min_dist > curr_dist:
                    loc = (i, n)
                    min_dist = curr_dist
        return loc

    def plot1D(self, t):
        """
        this function plot a network with 1 layer in the t'th iteration.
        :param t: the current iteration
        :return:
        """
        xs = []
        ys = []
        for i in range(self.neurons.shape[0]):
            for j in range(self.neurons.shape[1]):
                xs.append(self.neurons[i, j, 0])
                ys.append(self.neurons[i, j, 1])

        fig, ax = plt.subplots()
        ax.scatter([xs], [ys], c='r')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.plot(xs, ys, 'b-')
        ax.scatter(self.data[:, 0], self.data[:, 1], alpha=0.3)
        plt.title("Plot1D Iteration No. " + str(t))
        # plt.savefig("plot1D iteration " + str(t) + ".png")
        plt.show()

    def plot2D(self, t):
        """
        this function plot a network with 1 layer in the t'th iteration.
        :param t: the current iteration
        :return:
        """
        neurons_x = self.neurons[:, :, 0]
        neurons_y = self.neurons[:, :, 1]
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for i in range(neurons_x.shape[0]):
            xh = []
            yh = []
            xs = []
            ys = []
            for j in range(neurons_x.shape[1]):
                xs.append(neurons_x[i, j])
                ys.append(neurons_y[i, j])
                xh.append(neurons_x[j, i])
                yh.append(neurons_y[j, i])
            ax.plot(xs, ys, 'r-', markersize=0, linewidth=1)
            ax.plot(xh, yh, 'r-', markersize=0, linewidth=1)
        ax.plot(neurons_x, neurons_y, color='b', marker='o', linewidth=0, markersize=3)
        ax.scatter(self.data[:, 0], self.data[:, 1], c="g", alpha=0.05, s=5)
        plt.title("Plot2D Iteration No. " + str(t))
        # plt.savefig("plot2D iteration " + str(t) + ".png")
        plt.show()


def dist(vec, weights):
    return np.sqrt(((vec - weights) ** 2).sum())
