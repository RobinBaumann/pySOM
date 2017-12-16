import numpy as np
import itertools
from collections import Counter
import time


class SOM:
    """2D Self organizing map (SOM)."""

    def inv_alpha_decay(self, step):
        return self.init_alpha * (1 - (step / self.max_iterations))

    def linear_alpha_decay(self, step):
        return self.init_alpha * (1 / step)

    def power_alpha_decay(self, step):
        return self.init_alpha * np.exp(step / self.max_iterations)

    def gaussian_neighborhood_update(self, r_i, r_c):
        return np.exp(-(np.linalg.norm((np.subtract(r_i, r_c))) ** 2) / (2 * (self.sigma ** 2)))

    def bubble_neighborhood_update(self, r_i, r_c):
        if ((r_i[0] - r_c[0]) ** 2 + (r_i[1] - r_c[1]) ** 2) < self.sigma ** 2:
            return self.alpha
        else:
            return 0

    LEARNING_RATE_ALGOS = {'inv': inv_alpha_decay,
                           'linear': linear_alpha_decay,
                           'power': power_alpha_decay}
    NEIGHBORHOOD_FUNCTIONS = {'gaussian': gaussian_neighborhood_update,
                              'bubble': bubble_neighborhood_update}

    def __init__(self, map_shape,
                 input_dimensions,
                 init_alpha,
                 init_sigma,
                 max_iterations=500,
                 learning='inv',
                 neighborhood='gaussian'):
        """ Initialize the SOM with the given parameters.

        Parameters
        ----------
        map_shape : tuple
            Size of the emerging SOM.
        input_dimensions : int
            dimensionality of the training input.
        init_alpha : float
            initial value of the learning rate.
        init_sigma : float
            initial value of the neighborhood radius around the BMU
        max_iterations : int, optional
            number of training iterations
        learning : str, optional
            used learning rate schedule ('inv', 'linear', 'power')
        neighborhood : str, optional
            neighborhood function ('gaussian', 'bubble').

        """
        self.map_shape = map_shape
        self.input_dimensions = input_dimensions
        self.alpha = self.init_alpha = init_alpha
        self.sigma = self.init_sigma = init_sigma
        self.max_iterations = max_iterations
        self.labels = np.empty(shape=map_shape, dtype=np.object)
        self.learning = learning
        self.neighborhood = neighborhood
        self.weights = np.random.rand(map_shape[0], map_shape[1], input_dimensions)

    def decrease_sigma(self, step):
        """Decrease the Sigma value."""
        return self.init_sigma * (1 - step / self.max_iterations)

    def adjust_weights(self, input_vec, bmu):
        """Update the weightage matrix.

        Parameters
        ----------
        input_vec : Numpy Array
            Currently considered input vector. Must be in shape (n,), with n
            denoting the size of the input dimension.
        bmu : tuple
            [x, y]-Coordinates of the best matching unit for the currently
            considered input vector.

        """
        for node in itertools.product(range(self.map_shape[0]),
                                      range(self.map_shape[1])):
            n = self.NEIGHBORHOOD_FUNCTIONS[self.neighborhood](self, node, bmu)
            new_y = np.add(self.weights[node[0]][node[1]],
                           (self.alpha * n * np.subtract(input_vec,
                                                         self.weights[node])))
            self.weights[node[0]][node[1]] = new_y

    def find_bmu(self, input_vec):
        """Compute the best matching unit for a given input vector.

        Parameters
        ----------
        input_vec : Numpy Array
            currently considered input vector in shape (n,), with n denoting
            the size of the input_dimension

        Returns
        -------
        tuple
            The [x, y] - coordinates of the best matching unit.
        """
        bmu = [[0], [0]]
        min_dist = np.finfo(np.float64).max
        for node in itertools.product(range(self.map_shape[0]),
                                      range(self.map_shape[1])):
            dist = np.linalg.norm(np.subtract(self.weights[node[0]][node[1]],
                                              input_vec))
            if dist < min_dist:
                bmu = node
                min_dist = dist

        return [bmu[0], bmu[1]]

    def fit(self, X, y):
        """Training routine of the SOM.

        Parameters
        ----------
        X : Numpy Array
            Trainings set in shape (-1, n) with n denoting the specified
            input_dimension.
        y : Numpy Array
            Training set labels.

        Returns
        -------
        pixel_classes : Numpy Array
            weightage matrix for visualization purposes.

        """
        start = time.time()
        for step in range(1, self.max_iterations):
            idx = np.random.randint(len(X))
            x = X[idx]
            bmu = self.find_bmu(x)
            if self.labels[bmu[0]][bmu[1]] is None:
                self.labels[bmu[0]][bmu[1]] = []
            self.labels[bmu[0]][bmu[1]].append(y[idx])
            self.adjust_weights(x, bmu)

            self.alpha = self.LEARNING_RATE_ALGOS[self.learning](self, step)
            self.sigma = self.decrease_sigma(step)

        pixel_classes = np.zeros(shape=(self.map_shape[0], self.map_shape[1]),
                                 dtype=object)
        for node in itertools.product(range(self.map_shape[0]),
                                      range(self.map_shape[1])):
            spot = self.labels[node[0]][node[1]]
            if spot is None or spot == []:
                label = 0
            else:
                label_count = Counter(spot)
                label = label_count.most_common(1)[0][0]
            pixel_classes[node[0]][node[1]] = label

        print("SOM trained in: {:.2f} seconds".format(time.time() - start))
        return pixel_classes

    def predict(self, X):
        """Predict a class for a given evaluation set.

        Parameters
        ----------
        X : Numpy Array
            Evaluation set in shape (-1, n) with n denoting the specified input
            dimension.

        Returns
        -------
        predictions : list
            list of class predictions for every input instance

        """
        predictions = []
        for x in X:
            bmu = self.find_bmu(x)
            bmu_labels = self.labels[bmu[0]][bmu[1]]
            if bmu_labels is None or bmu_labels == []:
                prediction = 0
            else:
                label_count = Counter(bmu_labels)
                prediction = label_count.most_common(1)[0][0]

            predictions.append(prediction)

        return predictions
