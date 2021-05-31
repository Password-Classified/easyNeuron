'''
# easyNeuron
`easyNeuron` is a lightweight neural network framework written in Python for Python as one file.
It only uses Python Standard Library modules - not even numpy - to program it. I may also release
a full version that uses `numba` (non-standard library) to speed up code by running on the GPU.

----------

## Dependencies
This module uses only Python `Standard Library` modules for it - and here they are.

 - math
 - os
 - pickle
 - random
 - sys
 - decimal
 - pprint
 - timeit

### Docstrings

This module has extensive docstrings that are being constantly updated through time. They use
MarkDown formatting, so I am not sure if they show up properly on IDLE, but should on IDEs like
VS Code. Please raise any issues if there are terminological or grammatical issues on any docstrings.
'''

__version__ = 1.1

import math
import os
import pickle
import random
import sys
from decimal import Decimal
from pprint import pprint
from timeit import default_timer as timer

time_start = timer()

# Classmethods


class Matrix(classmethod):
    '''
    A classmethod for matrix operations,
    since numpy isn't used here, I had
    to write my own matrix operations.
    '''

    def dot(list_1, list_2):
        '''
        Return the dot product between 2
        matrices (which are both 2 or 1
        dimensional).
        '''

        return Decimal(sum(x*y for x, y in zip(list_1, list_2)))

    # def dot_prime(x, y):
    #     pass

    def transpose(matrix: list, disp=False):
        '''
        Returns a **transposed** matrix from the
        matrix you inputed to start with.

        If you set the `disp` parameter to `True`
        '''
        mat_len = len(matrix)
        mat_wid = len(matrix[0])
        new = []

        for i in range(mat_wid):
            new.append([])
            for x in range(mat_len):
                new[i].append(0)

        for x in range(mat_len):
            for y in range(mat_wid):
                new[y][x] = matrix[x][y]

        if disp:
            pprint(new)
        return new


class Timing(classmethod):
    def get_time(disp=False):
        current_time = timer()-time_start
        if disp:
            print(f'Time Elapsed: {current_time}')
        return current_time


class Data(classmethod):
    '''
    A classmethod for data manipulation,
    acquirement, loading and saving.
    '''

    def load_object(file_to_open):
        '''
        Load a list or any other object from a
        text file that will be created/opened.
        '''
        try:
            file_to_open_data = open(file_to_open, 'r')
            data = pickle.load(bytes(file_to_open_data))
        except FileNotFoundError:
            raise FileNotFoundError(
                f'An error has occured loading file_to_open {str(file_to_open_data)}.')
        finally:
            file_to_open_data.close()

        return data

    def save_object(data, file_to_open):
        try:
            file_to_open_data = open(file_to_open, 'w')
            data = pickle.dump(bytes(file_to_open))
            file_to_open_data.write(data)
        except FileExistsError:
            raise FileExistsError(
                f'An error has occured saving file_to_open {str(file_to_open_data)}')
        finally:
            file_to_open_data.close()

        return data


class Activation(classmethod):
    def sigmoid(inputs: list or tuple or float):
        '''
        Run the Sigmoid activation forwards.
        (for forwardpropagation)
        '''
        if type(inputs) == list or type(inputs) == tuple:
            output = []
            for i in inputs:
                output.append(Decimal(1/(1+math.e**float(i))))
            return output

        else:
            return Decimal(1/(1+math.e**float(inputs)))

    def sigmoid_prime(inputs: list or tuple or float):
        if type(inputs) == list or type(inputs) == tuple:
            output = []
            for i in inputs:
                output.append((1/(1+math.e**float(-i))) *
                              (1-(1/(1+math.e**float(-i)))))
            return output

        else:
            return (1/(1+math.e**float(-inputs))) * (1-(1/(1+math.e**float(-inputs))))

    def relu(inputs: list or tuple or float):
        '''
        Run the ReLU activation forwards.
        (for forwardpropagation)
        '''
        if type(inputs) == list or type(inputs) == tuple:
            output = []
            for i in inputs:
                output.append(Decimal(max(0, i)))
            return output
        else:
            Decimal(max(0, inputs))

    def relu_prime(inputs: list or tuple):
        if type(inputs) == list or type(inputs) == tuple:
            output = []
            for i in inputs:
                if i < 0:
                    output.append(0)
                else:
                    output.append(1)
            return output
        else:
            if inputs < 0:
                return 0
            else:
                return 1


# Parent Classes
class Layer(object):
    '''
    Parent class to all layers, containing
    the `__dunder__` methods needed.
    '''

    def __init__(self):
        self.biases = []
        self.weights = []
        self.output = []
        self._type = 'Undefined'

    def __repr__(self):
        return f'Layer_{self.type}(output={self.output})'

    def __str__(self):
        return f'Layer_{self.type}(output={self.output})'

    def __bool__(self):
        if self.output != []:
            return True

    def __len__(self):
        return len(self.output)

    def __eq__(self, o: object):
        try:
            if self.__class__ == o.__class__:
                return (self.output, self.type) == (o.output, o.type)
            else:
                return NotImplemented
        except:
            raise TypeError(
                f'Layer_{self.type} object is not comparable to given {type(o)} object.')

    def __hash__(self):
        return hash((self.output))

    def __bytes__(self):
        return bytes(tuple(self.output))

    def __enter__(self):
        return self.output

    def __exit__(self, type, value, traceback):
        pass

    @property
    def type(self):
        return self._type

class Cost(object):
    '''
    Parent class to all costs, containing
    the `__dunder__` methods needed.
    '''

    def __init__(self):
        self.output = []
        self._type = 'Undefined'

    def __repr__(self):
        return f'Cost_{self.type}(output={self.output})'

    def __str__(self):
        return f'Cost_{self.type}(output={self.output})'

    def __bool__(self):
        if self.output != []:
            return True

    def __len__(self):
        return len(self.output)

    def __eq__(self, o: object):
        try:
            if self.__class__ == o.__class__:
                return (self.output, self.type) == (o.output, o.type)
            else:
                return NotImplemented
        except:
            raise TypeError(
                f'Cost_{self.type} object is not comparable to given {type(o)} object.')

    def __hash__(self):
        return hash((self.output))

    def __bytes__(self):
        return bytes(self.output)

    def __enter__(self):
        return self.output

    def __exit__(self, type, value, traceback):
        pass

    @property
    def type(self):
        return self._type

class Optimizer(object):
    '''
    Parent class to all optimizers  , containing
    the `__dunder__` methods needed.
    '''

    def __init__(self):
        self.output = []
        self._type = 'Undefined'

    def __repr__(self):
        return f'Optimizer_{self.type}(output={self.output})'

    def __str__(self):
        return f'Optimizer_{self.type}(output={self.output})'

    def __bool__(self):
        if self.output != []:
            return True

    def __len__(self):
        return len(self.output)

    def __eq__(self, o: object):
        try:
            if self.__class__ == o.__class__:
                return (self.output, self.type) == (o.output, o.type)
            else:
                return NotImplemented
        except:
            raise TypeError(
                f'Optimizer_{self.type} object is not comparable to given {type(o)} object.')

    def __hash__(self):
        return hash((self.output))

    def __bytes__(self):
        return bytes(self.output)

    def __enter__(self):
        return self.output

    def __exit__(self, type, value, traceback):
        pass

    @property
    def type(self):
        return self._type


# Subclasses: Layers
class Layer_Dense(Layer):
    '''
    Create a layer with all neurons
    attached to next layer. Used a
    lot in classifiers. The best/
    default turn-to for developers.
    '''

    def __init__(self, n_inputs: int, n_neurons: int, bias_init: float = 0):
        if n_inputs <= 0:
            raise ValueError('"n_inputs" parameter should be > 0.')
        elif n_neurons <= 0:
            raise ValueError('"n_neurons" parameter should be > 0.')

        self.biases = []
        for x in range(n_neurons):
            self.biases.append(bias_init)

        self.weights = []
        for i in range(n_neurons):
            self.weights.append([])
            for n in range(n_inputs):
                self.weights[i].append(random.randrange(-4, 4))

        self._type = 'Dense'
        self.output = []

    def randomize(self, bias: int = 0):
        '''
        Randomize the weights and biases.
        Weights are randomized at the
        start, but biases are initialized
        to a default of 0.
        '''
        self.biases = []
        for x in range(len(self.biases)):
            self.biases.append(bias)

        self.weights = []
        for i in range(len(self.biases)):
            self.weights.append([])
            for n in range(len(self.weights)):
                self.weights[i].append(random.randint(-4, 4))

    def forward(self, inputs):
        '''
        Run the Dense Layer forwards.
        (forward propagate). It takes the
        dot product (matrices multiplied
        together) plus the bias of each
        neuron to give an output to be
        run through the activation.
        '''
        self.output = []
        for neuron in range(len(self.biases)):  # iterate for the num of neurons
            dotted = Matrix.dot(self.weights[neuron], inputs)
            self.output.append(Decimal(dotted + self.biases[neuron]))

        return self.output


if __name__ == '__main__':
    pass