import numpy as np

import src.emnist_utils as utils


class DataContainer:
    """
    Container object wrap EMNIST train and test set
    and provide some utils to investigate them.
    """

    @staticmethod
    def restore_images(X):
        X_restored = np.zeros(X.shape)
        for i in range(X.shape[0]):
            img = X[i].reshape(28, 28)
            img = np.rot90(img, 3)
            img = np.flip(img, 1).reshape(28 * 28)
            X_restored[i] = img
        return X_restored

    """
    You can directly retrieve following attributes:
        X_train: train set images
        y_train: train set labels
        X_test: test set images
        y_test: test set labels
        mapping: mapping from label to character
        char_labels: mapping from label to character
    """
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.mapping = None
        self.char_labels = None

    def load_emnist(self):
        """
        Load EMNIST dataset using emnist_utils.py
        """
        X_train, self.y_train, X_test, self.y_test = utils.load_emnist()
        self.X_train = DataContainer.restore_images(X_train)
        self.X_test = DataContainer.restore_images(X_test)
        self.mapping = utils.load_label_mapping()
        self.char_labels = self.map_label_to_char(self.mapping.keys())

    def get_X_train(self):
        return self.X_train

    def get_y_train(self):
        return self.y_train

    def get_X_test(self):
        return self.X_test

    def get_y_test(self):
        return self.y_test

    def get_chars(self, idx, in_test_set=False):
        if in_test_set:
            return np.array([chr(self.mapping[self.y_test[i]]) for i in idx])
        else:
            return np.array([chr(self.mapping[self.y_train[i]]) for i in idx])

    def plot_all_chars(self, max_no_columns=None):
        """
        Plot all characters in the dataset
        """
        utils.plot_all_chars(self.X_train[0:1000],
                                  self.y_train[0:1000], self.mapping, max_no_columns)

    def plot_character(self, char, no_images=10, max_no_columns=None):
        """
        Plot some images of specific character
        """
        utils.plot_random_images_of_char(char, self.X_train[0:10000],
                                         self.y_train[0:10000], self.mapping,
                                         no_images, max_no_columns)

    def map_label_to_char(self, y):
        return np.array([chr(self.mapping[label]) for label in y])

