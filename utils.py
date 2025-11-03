"""STA314 Homework 3.

Copyright and Usage Information
===============================

This file is provided solely for the personal and private use of students
taking STA314 at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited.
"""


import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    """ Computes the element wise logistic sigmoid of x.
    """
    return 1.0 / (1.0 + np.exp(-x))


def load_train():
    """ Loads training data for digits_train.
    """
    data = np.load("digits.npz")
    train_inputs = np.hstack((data["train2"], data["train3"]))
    train_targets = np.hstack((np.zeros((1, data["train2"].shape[1])), np.ones((1, data["train3"].shape[1]))))
    return train_inputs.T, train_targets.T


def load_train_small():
    """ Loads training data for digits_train_small.
    """
    data = np.load("digits.npz")
    train_inputs = np.hstack((data["train2"][:, :2], data["train3"][:, :2]))
    train_targets = np.hstack((np.zeros((1, 2)), np.ones((1, 2))))
    return train_inputs.T, train_targets.T


def load_valid():
    """ Loads validation data.
    """
    data = np.load("digits.npz")
    valid_inputs = np.hstack((data["valid2"], data["valid3"]))
    valid_targets = np.hstack((np.zeros((1, data["valid2"].shape[1])), np.ones((1, data["valid3"].shape[1]))))
    return valid_inputs.T, valid_targets.T


def load_test():
    """ Loads validation data.
    """
    data = np.load("digits.npz")
    test_inputs = np.hstack((data["test2"], data["test3"]))
    test_targets = np.hstack((np.zeros((1, data["test2"].shape[1])), np.ones((1, data["test3"].shape[1]))))
    return test_inputs.T, test_targets.T


def plot_digits(digit_array):
    """ Visualizes each example in digit_array.
    :param digit_array: N x D array of pixel intensities.
    :return: None
    """
    CLASS_EXAMPLES_PER_PANE = 5

    # assume two evenly split classes
    examples_per_class = int(digit_array.shape[0] / 2)
    num_panes = int(np.ceil(float(examples_per_class) / CLASS_EXAMPLES_PER_PANE))

    for pane in range(num_panes):
        print("Displaying pane {}/{}".format(pane + 1, num_panes))

        top_start = pane * CLASS_EXAMPLES_PER_PANE
        top_end = min((pane + 1) * CLASS_EXAMPLES_PER_PANE, examples_per_class)
        top_pane_digits = extract_digits(digit_array, top_start, top_end)

        bottom_start = top_start + examples_per_class
        bottom_end = top_end + examples_per_class
        bottom_pane_digits = extract_digits(digit_array, bottom_start, bottom_end)

        show_pane(top_pane_digits, bottom_pane_digits)


def extract_digits(digit_array, start_index, end_index):
    """ Returns a list of 16 x 16 pixel intensity arrays starting
    at start_index and ending at end_index.
    """
    digits = []
    for index in range(start_index, end_index):
        digits.append(extract_digit_pixels(digit_array, index))
    return digits


def extract_digit_pixels(digit_array, index):
    """ Extracts the 16 x 16 pixel intensity array at the specified index.
    """
    return digit_array[index].reshape(16, 16).T


def show_pane(top_digits, bottom_digits):
    """ Displays two rows of digits on the screen.
    """
    all_digits = top_digits + bottom_digits
    fig, axes = plt.subplots(nrows=2, ncols=int(len(all_digits) / 2))
    for axis, digit in zip(axes.reshape(-1), all_digits):
        axis.imshow(digit, interpolation="nearest", cmap=plt.gray())
        axis.set_xticklabels([])
        axis.set_yticklabels([])
        axis.axis("off")
    # fig.subplots_adjust(wspace=0,
    #                     hspace=0)
    plt.tight_layout(h_pad=-7)
    plt.show()


def save_images(images, filename):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_digits(images)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)


if __name__ == "__main__":
    x_train, y_train = load_train()
    show_pane(extract_digits(x_train, 0, 5), extract_digits(x_train, -6, -1))
