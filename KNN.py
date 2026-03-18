__authors__ = ['1711491']
__group__ = '49'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.train_data = np.array(train_data, dtype=float).reshape(len(train_data), -1)

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        # Aplanem les dades de test igual que les de train (N, D)
        test_data_reshaped = np.array(test_data, dtype=float).reshape(len(test_data), -1)

        # Calculem la distància entre cada punt de test i cada punt de train
        dists = cdist(test_data_reshaped, self.train_data, 'euclidean')

        # Obtenim els índexs dels k veïns amb la distància MÉS PETITA
        # np.argsort ens dóna els índexs ordenats de menor a major distància
        neighbors_indices = np.argsort(dists, axis=1)[:, :k]

        # Assignem les etiquetes (labels) corresponents a aquests índexs
        self.neighbors = self.labels[neighbors_indices]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        # Si self.neighbors és (N, K), volem un array de mida N
        predictions = []
        for row in self.neighbors:
            # np.unique ordena alfabèticament, cosa que trenca el desempat per proximitat.
            # Per mantenir l'ordre d'aparició (proximitat), usem aquest truc:
            vals, counts = np.unique(row, return_counts=True)

            # Si només hi ha un màxim clar:
            if np.sum(counts == np.max(counts)) == 1:
                most_voted = vals[np.argmax(counts)]
            else:
                # EN CAS D'EMPAT:
                # Busquem quina de les etiquetes que han empatat apareix PRIMER a 'row'
                # (el que significa que és el veí més proper).
                max_votes = np.max(counts)
                candidates = vals[counts == max_votes]
                # Mirem quin candidat té l'índex més baix a la fila original
                most_voted = min(candidates, key=lambda x: np.where(row == x)[0][0])

            predictions.append(most_voted)

        return np.array(predictions)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
