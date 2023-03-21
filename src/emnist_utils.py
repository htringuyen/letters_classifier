"""
utils functions for EMNIST dataset:
    - load_emnist: load EMNIST dataset
    - load_label_mapping: load mapping dict that map each label to character
    - plot_all_chars: plot all characters in EMNIST dataset
    - plot_random_images_of_char: plot some images of given character from images_data
    - plot_characters: plot characters from images_data
    note: read function docs for more information
"""
from math import sqrt

import numpy as np
import gzip
from pathlib import Path
import matplotlib.pyplot as plt

# Name of EMNIST data files
EMNIST_BALANCED_FILE_NAMES = ["emnist-balanced-train-images-idx3-ubyte.gz",
                              "emnist-balanced-train-labels-idx1-ubyte.gz",
                              "emnist-balanced-test-images-idx3-ubyte.gz",
                              "emnist-balanced-test-labels-idx1-ubyte.gz"]
ROW_SIZES = [784, 1, 784, 1]

DATA_PATH = Path().absolute().parent / "data"


def get_data_path():
    return DATA_PATH


def load_emnist(data_names=EMNIST_BALANCED_FILE_NAMES, row_sizes=ROW_SIZES, data_path=DATA_PATH / "emnist_gz"):
    """
    Load EMNIST dataset
    :param data_names: name of data files in gz format
    :param row_sizes: number of row of array data from each file
    :param data_path: the path to data directory
    :return:
    """
    data_list = []  # list that store data
    for data_name, row_size in zip(data_names, row_sizes):
        with gzip.open(data_path / data_name, "rb") as f:
            file_content = f.read()
            data = file_content[16 if row_size == 784 else 8:]
            arr = np.frombuffer(data, dtype=np.uint8)
            data_list.append(
                arr.reshape(int(len(arr) / row_size), row_size) if row_size == 784 else arr)
    return data_list


def load_label_mapping(data_path=DATA_PATH / "mapping", file_name="emnist-balanced-mapping.txt"):
    """
    Load mapping dict that map each label to character
    :param data_path:
    :param file_name:
    :return: mapping: dictionary that maps label to character
    """
    with open(data_path / file_name, "r") as f:
        lines = f.readlines()
        mapping = {}
        for line in lines:
            label, char_code = line.strip().split()
            mapping[int(label)] = int(char_code)
    return mapping


def __plot_character(image_data, plot_axis=False):
    """
    Plot a character, this is a private function, please use plot_characters instead
    :param image_data: an array that is a character image
    :param plot_axis: whether to plot axis
    :return:
    """
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis(plot_axis)


def plot_characters(image_data, max_no_columns=None, plot_axis=False):
    """
    Plot each character image corresponding to each row of array image_data
    :param image_data: an array that of which row is a character image
    :param max_no_columns: maximum number of characters in a row
    :param plot_axis: whether to plot axis
    :return:
    """
    max_no_columns = max_no_columns if max_no_columns else int(sqrt(len(image_data)))
    no_images = image_data.shape[0]
    if no_images == 0:
        return
    no_columns = min(no_images, max_no_columns)
    no_rows = int(np.ceil(no_images / no_columns))
    for idx, image in enumerate(image_data):
        plt.subplot(no_rows, no_columns, idx + 1)
        __plot_character(image, plot_axis)
    plt.subplots_adjust(wspace=0, hspace=0)


def random_images_idx_of_char(char, labels, mapping, no_images=10):
    """
    Randomly get indices of images that corresponding to a character
    Args:
        char: character which we want to get images of
        labels: labels of images
        no_images: the number of images
        mapping: mapping from label to character
    Returns:
        images_idx:
    """
    matched_idx = list(filter(lambda idx: mapping[labels[idx]] == to_char_code(char),
                              range(len(labels))))
    return np.random.choice(matched_idx, min(no_images, len(matched_idx)))


def to_char_code(char):
    """
    Convert character to char code
    return the char code if input is integer
    :param char: char that need to be converted
    :return: char code
    """
    return char if isinstance(char, int) else ord(char)


def to_char(char_code):
    """
    convert char code to character
    return the input itself if it is a character
    :param char_code: that char code that need to be converted
    :return:
    """
    return char_code if isinstance(char_code, str) else chr(char_code)


def plot_random_images_of_char(char, images_data, labels, mapping, no_images=10, max_no_columns=None):
    """
    Plot some images of specific character from images_data
    :param char: character that we want to plot
    :param images_data: contains array of images
    :param labels: labels of images
    :param mapping: mapping from label to character
    :param no_images: number of images that we want to plot
    :param max_no_columns: maximum number of images in a row
    :return:
    """
    images_idx = random_images_idx_of_char(char, labels, mapping, no_images)
    plot_characters(images_data[images_idx], max_no_columns)
    plt.xlabel("Random images of character {}".format(to_char(char)))
    plt.show()


def plot_all_chars(image_data, labels, mapping, max_no_columns=None):
    """
    Plot all characters in balanced EMNIST dataset
    :param image_data: array images data that contains set of images
    :param labels: corresponding labels of images
    :param mapping: mapping from label to character
    :param max_no_columns: maximum number of images in a row
    :return:
    """
    char_idx = []
    for char in mapping.values():
        char_idx.append(random_images_idx_of_char(char, labels, mapping, 1))
    plot_characters(image_data[char_idx], max_no_columns)