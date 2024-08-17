import random


class Neuron:

    def __init__(self, number_of_previous_neurons):
        self.value = 0
        self.bias = random.uniform(-0.5, 0.5)
        self.in_weigths = [random.uniform(-0.1, 0.1) for _ in range(number_of_previous_neurons)]

    def calculate_activation(self, inputs):
        weighted_sum = self.bias
        for input_neuron, weight in zip(inputs, self.in_weigths):
            weighted_sum += input_neuron * weight
        self.value = weighted_sum

    def activate(self, activation_function):
        self.value = activation_function(self.value)
