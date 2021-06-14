'''
# easyNeuron
`easyNeuron` is a lightweight neural network framework written in Python for Python as one file.
It only uses Python Standard Library modules - not even numpy - to program it. I may also release
a full version that uses `numba` (non-standard library) to speed up code by running on the GPU.

----------

## Dependencies
This module uses only Python `Standard Library` modules for it - and here they are.

 - csv
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


#### Github Repository: https://github.com/Password-Classified/easyNeuron - this is not published yet.
'''

__version__ = 1.2

import csv
import math
import os
import pickle
import random
import sys
from decimal import Decimal
from pprint import pprint
from timeit import default_timer as timer


# Classmethod for matrices.
class Matrix(classmethod):
    '''
    A classmethod for matrix operations,
    since numpy isn't used here, I had
    to write my own matrix operations.
    '''

    def dot(list_1: list or tuple, list_2: list or tuple):
        '''
        Return the dot product between 2
        matrices (which are both 2 or 1
        dimensional).
        '''

        return Decimal(sum(x*y for x, y in zip(list_1, list_2)))

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

    def depth(inputs):
        count = 0
        for item in inputs:
            if isinstance(item, inputs):
                count += Matrix.depth(item)
        return count+1
  
# Classmethods so I don't need to create objects for each of these.
class Activation(classmethod):
    valid_activations = ['sigmoid', 'sigmoid_prime',
                         'relu', 'relu_prime']

    def sigmoid(inputs: list or tuple or float or int):
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

    def sigmoid_prime(inputs: list or tuple or float or int):
        if type(inputs) == list or type(inputs) == tuple:
            output = []
            for i in inputs:
                output.append((1/(1+math.e**float(-i))) *
                              (1-(1/(1+math.e**float(-i)))))
            return output

        else:
            return (1/(1+math.e**float(-inputs))) * (1-(1/(1+math.e**float(-inputs))))

    def relu(inputs: list or tuple or float or int):
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
            return Decimal(max(0, inputs))

    def relu_prime(inputs: list or tuple or int):
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
            
class Costs(classmethod):
    def MSE(inputs, targets):
        inp = [inputs, targets]
        for inpt in inp:
            tp = type(inpt)
            if tp != list:
                if tp == tuple:
                    inpt = list(inpt)
                elif tp == int or tp == float:
                    inpt = [inpt]
                else:
                    if type(inputs) == type(targets):
                        raise TypeError(
                            f'Both parameters should be list, tuple, int or float, not {tp}.')
                    else:
                        raise TypeError(
                            f'Both parameters should be list, tuple, int or float, not {type(inputs)} and {type(targets)}.')

        if type(inputs) == list or type(inputs) == tuple:
            
            length = len(inputs)
            if length != len(targets):
                raise IndexError(
                    f'Inputs ({length}) has not the same size as targets ({len(targets)}).')

            output = 0
            for i in range(length):
                output += ((inputs[i] - targets[i])**2)/2
            output /= length

            return output

        else:
            return Decimal(((targets-inputs)**2)/2)

    def MSE_prime(inputs, targets):
        inp = [inputs, targets]
        for inpt in inp:
            tp = type(inpt)
            if tp != list:
                if tp == tuple:
                    inpt = list(inpt)
                elif tp == int or tp == float:
                    inpt = [inpt]
                else:
                    if type(inputs) == type(targets):
                        raise TypeError(
                            f'Both parameters should be list, tuple, int or float, not {tp}.')
                    else:
                        raise TypeError(
                            f'Both parameters should be list, tuple, int or float, not {type(inputs)} and {type(targets)}.')

        length = len(inputs)
        if length != len(targets):
            raise IndexError(
                f'Inputs ({length}) has not the same size as targets ({len(targets)}).')

        output = 0
        for i in range(length):
            output += (inputs[i] - targets[i])
        output /= length

        return output


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
        self._act = 'Undefined'
        self._type = 'Undefined'

    def __repr__(self):
        return f'Layer_{self.type}(activation={self._act})'

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

    @property
    def activation(self):
        return self._act
        
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


# Object subclasses
class Layer_Dense(Layer):
    '''
    Create a layer with all neurons
    attached to next layer. Used a
    lot in classifiers. The best/
    default turn-to for developers.
    '''

    def __init__(self, n_inputs: int, n_neurons: int, activation: str, bias_init: float = 0):
        if n_inputs <= 0:
            raise ValueError('"n_inputs" parameter should be > 0.')
        elif n_neurons <= 0:
            raise ValueError('"n_neurons" parameter should be > 0.')
        if not activation in Activation.valid_activations:
            raise ValueError(
                f'"activations" parameter must be in the list valid_activations, not {activation}.\nThe valid activations are:\n    {Activation.valid_activations}')

        self.biases = []
        for x in range(n_neurons):
            self.biases.append(bias_init)

        self.weights = []
        for i in range(n_neurons):
            self.weights.append([])
            for n in range(n_inputs):
                self.weights[i].append(random.randrange(-4, 4))

        self._type = 'Dense'
        self._act = activation
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

        # Dot product
        for neuron in range(len(self.biases)):  # iterate for the num of neurons
            dotted = Matrix.dot(self.weights[neuron], inputs)
            self.output.append(Decimal(dotted + self.biases[neuron]))

        # Activation
        for i in range(len(self.output)):
            self.output[i] = getattr(Activation, f'{self.activation}')(i) # run activation on it

        return self.output

# Demo
if __name__ == '__main__':
    '''
    The simple dataset below has numbers
    between 1 and 35, and if the number is
    greater or equal to 20, the output is 1.
    If it is less, it is 0. This is a simple dataset
    I'll train the first network on.
    '''
    inputs = [[29], [35], [17], [2], [16], [1], [11], [4], [35], [31], [21], [17], [4], [7], [24], [12], [28], [16], [5], [11], [32], [18], [27], [18], [18], [23], [10], [16], [28], [12], [34], [35], [17], [22], [24], [7], [7], [28], [26], [21], [9], [18], [25], [3], [16], [25], [25], [16], [15], [6], [21], [9], [31], [28], [3], [35], [27], [31], [23], [9], [3], [25], [32], [33], [6], [35], [27], [27], [28], [24], [27], [15], [16], [34], [13], [16], [15], [14], [28], [15], [13], [18], [13], [22], [20], [18], [2], [31], [26], [11], [12], [25], [35], [20], [31], [23], [27], [21], [32], [18]]
    labels = [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]

    
    l1 = Layer_Dense(1, 3, activation='sigmoid') # Input layer
    l2 = Layer_Dense(3, 2, activation='sigmoid') # Hidden layer
    l3 = Layer_Dense(2, 1, activation='sigmoid') # Output layer
    
    # Forward propagate
    for i in inputs: # Iterate through inputs
        l1.forward(i)
        l2.forward(l1.output)
        l3.forward(l2.output)

        print(f'Cost: {Costs.MSE(l3.output, i)}')