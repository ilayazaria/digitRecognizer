import os.path
import pickle
import math
from typing import List

from PIL import Image
from keras.datasets import mnist
from matplotlib import pyplot as plt

from consts import FIRST_LAYER_SIZE, OUTPUT_LAYER_SIZE, LEARNING_RATE
from Neuron import Neuron
import datetime


def get_value_from_neuron_list(neuron_list: List[Neuron]):
    return [neuron.value for neuron in neuron_list]


def get_bias_from_neuron_list(neuron_list: List[Neuron]):
    return [neuron.bias for neuron in neuron_list]


def ReLU(Z):
    return max(0, Z)


def softmax(inputs):
    temp = [math.exp(v) for v in inputs]
    total = sum(temp)
    return [t / total for t in temp]


def forward_prop(inputs: List, first_layer: List[Neuron], output_layer: List[Neuron]):
    for neuron in first_layer:
        neuron.calculate_activation(inputs)
        neuron.activate(ReLU)
    first_layer_inputs = get_value_from_neuron_list(first_layer)
    for neuron in output_layer:
        neuron.calculate_activation(first_layer_inputs)
    output_layer_values = get_value_from_neuron_list(output_layer)
    return softmax(output_layer_values)


def calculate_loss_output_layer(output_layer, expected_values):
    return [o - e for o, e in zip(output_layer, expected_values)]


def derivative_ReLU(activation):
    return activation > 0


def back_prop(inputs, first_layer: List[Neuron], output_layer: List[Neuron], expected_values, softmax_values):
    output_layer_loss = calculate_loss_output_layer(softmax_values, expected_values)
    first_layer_loss = [0] * FIRST_LAYER_SIZE
    for i, output_neuron in enumerate(output_layer):
        for j, first_neuron in enumerate(first_layer):
            first_layer_loss[j] += output_layer_loss[i] * output_neuron.in_weigths[j]
    for i, loss in enumerate(first_layer_loss):
        first_layer_loss[i] = loss * derivative_ReLU(first_layer[i].value)

    for i, output_layer_neuron in enumerate(output_layer):
        for j, first_layer_neuron in enumerate(first_layer):
            output_layer_neuron.in_weigths[j] -= LEARNING_RATE * first_layer_neuron.value * output_layer_loss[i]
        output_layer_neuron.bias -= LEARNING_RATE * output_layer_loss[i]

    for i, first_layer_neuron in enumerate(first_layer):
        for j, input_value in enumerate(inputs):
            first_layer_neuron.in_weigths[j] -= LEARNING_RATE * input_value * first_layer_loss[i]
        first_layer_neuron.bias -= LEARNING_RATE * first_layer_loss[i]


def build_expected_values_array(number):
    exp_values = [0] * OUTPUT_LAYER_SIZE
    exp_values[number] = 1
    return exp_values


def get_selection(softmax_values):
    max_value = max(softmax_values)  # Find the maximum value in the list
    max_index = softmax_values.index(max_value)  # Find the index of the maximum value
    return max_index


def see_difference(first_list, second_list):
    differences = [(i, a, b) for i, (a, b) in enumerate(zip(first_list, second_list)) if
                   a != b and a - b > 0.01 or a - b < -0.01]
    print(differences)


def list_divide_by(num_lst, number):
    return [value / number for value in num_lst]


def main():
    if os.path.exists('./nn-trained.pkl'):
        with open('nn-trained.pkl', 'rb') as f:
            first_layer = pickle.load(f)
            output_layer = pickle.load(f)
    else:
        first_layer = [Neuron(784) for _ in range(FIRST_LAYER_SIZE)]
        output_layer = [Neuron(FIRST_LAYER_SIZE) for _ in range(OUTPUT_LAYER_SIZE)]
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print(datetime.datetime.now())
    success = 0
    failure = 0
    for i in range(10):
        for image, value in zip(train_X, train_y):
            pixels = list(image.flatten())
            pixels = list_divide_by(pixels, 100)
            softmax_values = forward_prop(pixels, first_layer, output_layer)
            predicted_value = get_selection(softmax_values)
            if value == predicted_value:
                success += 1
            else:
                failure += 1
            if (success + failure) % 500 == 0:
                print(success / (success + failure))
            expected_values = build_expected_values_array(value)
            back_prop(pixels, first_layer, output_layer, expected_values, softmax_values)

        with open('nn-trained.pkl', 'wb') as f:
            pickle.dump(first_layer, f)
            pickle.dump(output_layer, f)
    for img, value in zip(test_X, test_y):
        pixels = list(img.flatten())
        pixels = list_divide_by(pixels, 100)
        softmax_values = forward_prop(pixels, first_layer, output_layer)
        predicted_value = get_selection(softmax_values)
        if predicted_value != value:
            width, height = 28, 28
            image = Image.new('L', (width, height))
            image.putdata(pixels)

            # Display the image with the prediction as the title
            plt.imshow(image, cmap='gray')
            plt.title("predicted: " + str(predicted_value) + " expected: " + str(value))
            plt.axis('off')  # Hide axes
            plt.show()


main()
