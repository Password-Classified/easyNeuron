'''
# easyNeuron

`easyNeuron` is an easy-to-use lightweight neural network framework written in raw Python.
It only uses Python Standard Library modules - not even numpy - to program it. I may also release
a full version that uses `numba` (non-standard library) to speed up code by running on the GPU.

-------------------------------------------------------------------------------------------------

## Dependencies

This module uses only Python `Standard Library` modules for it - and here they are.

 - copy
 - csv
 - decimal
 - functools
 - math
 - pickle
 - random
 - secrets
 - statistics
 - timeit
 - typing

### Docstrings

This module has extensive docstrings that are being constantly updated through time. They use
MarkDown formatting, so I am not sure if they show up properly on IDLE, but should on IDEs like
VS Code. Please raise any issues if there are terminological or grammatical issues on any docstrings.


#### [Github Repository](https://github.com/Password-Classified/easyNeuron)
'''

import copy
import math
import pickle
import random
import secrets
import statistics
from decimal import Decimal
from functools import reduce
from timeit import default_timer as timer
from typing import Any, Union

time_start = timer()

# Types
_Data = Union[list, tuple, int, float]
_Number = Union[float, int]
_OptimizerType = Union[str, object]
_ListLike = Union[list, tuple]

_Data_Tuple = (list, tuple, int, float)
_Number_Tuple = (float, int)
_OptimizerType_Tuple = (str, object)
_ListLike_Tuple = (list, tuple)

# Developer Classmethods
class _Utils(classmethod):
    """
    _Util classmethods.

    Developer methods for the module, not necessary for users.
    
    """
    def _dispGrad(epoch: int, loss: float, disp_level: int = 0) -> None:
        """
        Display information on the current training state for GradDesc. Not needed by the user.

        ### Params

         - epoch = current epoch
         - loss = current cost function output
         - disp_level = amount to display

        """
        if disp_level >= 1: print(f"Epoch: {epoch+1}\tLOSS: {loss}")

    def _dispRand(epoch: int, loss: float, disp_level: int,  found: bool) -> None:
        """
        Display information of the epoch for RandDesc. Not needed used by user.

        ### Params

         - epoch = current epoch
         - loss = current cost function output
         - disp_level = amount to display
         - found = whether a new weight configuration has been found
        """
        if disp_level != 0:
            print(f'Epoch: {epoch + 1}\tLOSS: {round(loss, 5)} \tNew Weights: {str(found)}')

# General Classmethods
class Matrix(classmethod):
    '''

    A classmethod for matrix operations,
    since numpy isn't used here, I had
    to write my own matrix operations.
    '''
    def dot(list_1: _Data, list_2: _Data) -> list:
        '''
        Return the dot product between 2
        matrices (which are both 2 or 1
        dimensional) or return both the
        elements multiplie (for numbers).

        ### Params

         - list_1 = a list, tuple, float or integer
         - list_2 = a list, tuple. float or integer

        ### Returns


        '''
        if isinstance(list_1, _Number_Tuple) and isinstance(list_2, _Number_Tuple):
            return Decimal(list_1 * list_2)
        else:
            return Decimal(sum(float(x)*float(y) for x, y in zip(list_1, list_2)))

    def transpose(matrix: _ListLike) -> list:
        '''
        Returns a **transposed** matrix from the
        matrix you inputed to start with.
        '''
        new = list(zip(*matrix))

        for i, _ in enumerate(new):
            new[i] = list(new[i])

        return new

    def depth(inputs: _ListLike) -> int:
        if isinstance(inputs, list):
            return 1 + max(Matrix.depth(item) for item in inputs)
        else:
            return 0

class Timing(classmethod):

    def get_time(disp: bool=False) -> float:
        current_time = timer() - time_start
        if disp: print(f'Time Elapsed: {current_time}')
        return current_time

class Data(classmethod):
    '''
    A classmethod for data manipulation,
    acquirement, loading and saving.
    '''

    def load_object(file_to_open: str) -> list:
        '''
        Load a list or any other object from a
        text file that will be created/opened.
        '''
        with open(file_to_open, "rb") as file:
            data = pickle.load(file)

        return data

    def save_object(data: Any, file_to_open: str) -> str:
        with open(file_to_open, "wb") as file:
            pickle.dump(data, file)

        return data

    def scale(data: _ListLike, feature_range: tuple = (0, 1)) -> _Data:
        if len(feature_range) != 2:
            raise ValueError(
                f'"feature_range" tuple has to be length 2, not length {len(feature_range)}.')

        depth = Matrix.depth(data)
        largest = 0
        smallest = 0
        curr_data = data


        for item in curr_data:
            if depth >1:
                for deep in depth:
                    if item[deep] > largest:
                        largest = item[deep]
                    elif item[deep] < smallest:
                        smallest = item[deep]
            else:
                if item > largest:
                    largest = item
                elif item < smallest:
                    smallest = item
        raise NotImplementedError("This feature is coming soon and is presently not implemented fully.")

    def shuffle(data: _Data):
        return random.shuffle(data)

    def gen_cluster(size: int, difficulty: float) -> _Data:
        raw = []
        for _ in range(size):
            raw.append([Random.random_int(2500, 3500)/100 + Random.random_int(2500, 3500)/difficulty, Random.random_int(100, 800)/100 + Random.random_int(2000, 3500)/difficulty, 1])
            raw.append([Random.random_int(100, 800)/100 + Random.random_int(2500, 3500)/difficulty, Random.random_int(2500, 3500)/100 + Random.random_int(2000, 3500)/difficulty, 0])

        X = [[round(i[0], 5), round(i[1], 5)] for i in raw]
        y = [[round(i[2], 5)] for i in raw]

        return X, y

    def load_mnist() -> _Data:
        raise NotImplementedError("This is a new feature coming soon.")

    def load_dna() -> _Data:
        raise NotImplementedError("This feature is coming soon and is presently not implemented fully.")

    def load_words() -> _Data:
        raise NotImplementedError("This feature is coming soon and is presently not implemented fully.")

    def load_cities() -> _Data:
        with open('Data/Cities.txt') as file:
            output = file.readlines()

        return [i.strip('\n') for i in output]

class Random(classmethod):
    def seed(seed: int) -> None:
        random.seed(seed)

    def random_int(x: _Number, y: _Number):
        """
        A version of random.randrange() using
        the `secrets` module.

        ### Params

         - x = a number, start of range
         - y = a number, end of range
        """

        return secrets.randbelow(abs(y - x)) + x

# Network Classmethods
class Activation(classmethod):

    def sigmoid(inputs: _Data):
        '''
        Run the Sigmoid activation forwards.
        (for forwardpropagation)
        '''
        if isinstance(inputs, _ListLike_Tuple):
            output = []
            for i in inputs:
                output.append(Decimal(1/(1+math.e**-float(i))))
            return output

        else:
            return Decimal(1/(1+Decimal(math.e)**-Decimal(inputs)))

    def sigmoid_prime(inputs: _Data):
        if isinstance(inputs, _ListLike_Tuple):
            output = []
            for i in inputs:
                output.append( (1/(1+math.e**-float(i))) * (1-(1/(1+math.e**-float(i)))) )
            return output

        else:
            return (1/(1+math.e**-float(inputs))) * (1-(1/(1+math.e**-float(inputs))))

    def relu(inputs: _Data):
        '''
        Run the ReLU activation forwards.
        (for forwardpropagation)
        '''
        if isinstance(inputs, _ListLike_Tuple):
            output = []
            for i in inputs:
                output.append(Decimal(max(0, i)))
            return output
        else:
            return Decimal(max(0, inputs))

    def relu_prime(inputs: list or tuple or int):
        if isinstance(inputs, _ListLike_Tuple):
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

class Loss(classmethod):

    def MSE(inputs: _Data, targets: _Data) -> float:
        inp = [inputs, targets]
        for inpt in inp:
            tp = type(inpt)
            if tp != list:
                if tp == tuple:
                    inpt = list(inpt)
                elif tp == int or tp == float:
                    inpt = [inpt]
                else:
                    if isinstance(inputs, type(targets)):
                        raise TypeError(
                            f'Both parameters should be list, tuple, int or float, not {tp}.')
                    else:
                        raise TypeError(
                            f'Both parameters should be list, tuple, int or float, not {type(inputs)} and {type(targets)}.')

        if isinstance(inputs, _ListLike_Tuple):

            length = len(inputs)
            if length != len(targets):
                raise IndexError(
                    f'Inputs ({length}) has not the same size as targets ({len(targets)}).\nInputs = {inputs},\nTargets = {targets}')

            output = 0
            for i in range(length):
                output += ((inputs[i] - targets[i])**2)/2
            output /= length

            return output

        else:
            return Decimal(((targets-inputs)**2)/2)

    def MSE_prime(inputs: _Data, targets: _Data) -> float:
        inp = [inputs, targets]
        for inpt in inp:
            tp = type(inpt)
            if tp != list:
                if tp == tuple:
                    inpt = list(inpt)
                elif tp == int or tp == float:
                    inpt = [inpt]
                else:
                    if isinstance(input, type(targets)):
                        raise TypeError(
                            f'Both parameters should be list, tuple, int or float, not {tp}.')
                    else:
                        raise TypeError(
                            f'Both parameters should be list, tuple, int or float, not {type(inputs)} and {type(targets)}.')

        length = len(inputs)
        if length != len(targets):
            raise IndexError(
                f'Inputs ({length}) has not the same size as targets ({len(targets)}).\nInputs = {inputs},\nTargets = {targets}')

        output = 0
        for i in range(length):
            output += (inputs[i] - targets[i])
        output /= length

        return output

    def MAE(inputs: _Data, targets: _Data) -> float:
        inp = [inputs, targets]
        for inpt in inp:
            tp = type(inpt)
            if tp != list:
                if tp == tuple:
                    inpt = list(inpt)
                elif tp == int or tp == float:
                    inpt = [inpt]
                else:
                    if isinstance(inputs, type(targets)):
                        raise TypeError(
                            f'Both parameters should be list, tuple, int or float, not {tp}.')
                    else:
                        raise TypeError(
                            f'Both parameters should be list, tuple, int or float, not {type(inputs)} and {type(targets)}.')

        if isinstance(inputs, _ListLike_Tuple):

            length = len(inputs)
            if length != len(targets):
                raise IndexError(
                    f'Inputs ({length}) has not the same size as targets ({len(targets)}).\nInputs = {inputs},\nTargets = {targets}')

            output = 0
            for i in range(length):
                output += abs(inputs[i] - targets[i])
            output /= length

            return output

        else:
            return Decimal(abs(targets-inputs))

    def MAE_prime(inputs: _Data, targets: _Data) -> float:
        inp = [inputs, targets]
        for inpt in inp:
            tp = type(inpt)
            if tp != list:
                if tp == tuple:
                    inpt = list(inpt)
                elif tp == int or tp == float:
                    inpt = [inpt]
                else:
                    if isinstance(inputs, type(targets)):
                        raise TypeError(
                            f'Both parameters should be list, tuple, int or float, not {tp}.')
                    else:
                        raise TypeError(
                            f'Both parameters should be list, tuple, int or float, not {type(inputs)} and {type(targets)}.')

        if isinstance(inputs, _ListLike_Tuple): length = len(inputs)
        else: length = 1
        if length != len(targets):
            raise IndexError(
                f'Inputs ({length}) has not the same size as targets ({len(targets)}).\nInputs = {inputs},\nTargets = {targets}')

        output = []
        for i in range(length):
            if isinstance(inputs, _ListLike_Tuple):
                if inputs[i] < targets[i]: output.append(-1)
                else: output.append(1)
            else:
                if inputs < targets[i]: output.append(-1)
                else: output.append(1)

        return statistics.fmean(output)

# Parent Classes
class Layer(object):
    '''
    Parent class to all layers, containing
    the `__special__` methods needed.
    '''

    def __init__(self):
        """Sets default values for properties."""
        self.biases = []
        self.weights = []
        self.output = []
        self._act = 'Undefined'
        self._type = 'Undefined'

    def __repr__(self):
        """Display info as a string."""
        return f'{self.category}(activation={self._act})'

    def __str__(self):
        """Return as str."""
        return f'{self.category}(activation={self._act})'

    def __call__(self, inputs):
        return self.forward(inputs)

    def __len__(self):
        """Return number of neurons."""
        return len(self.biases)

    def __eq__(self, o: object):
        """Check if equivalent to object."""
        try:
            return self is o
        except Exception:
            raise TypeError(
                f'Layer_{self.category} object is not comparable to given {type(o)} object.')

    def __hash__(self):
        """Create a hash for a dictionary."""
        return hash(self)

    def __bytes__(self):
        """Turn to bytes."""
        return bytes(self)

    def __enter__(self):
        """Use within a with statement."""
        return self

    def __exit__(self, category, value, traceback):
        """Placeholder to avoid errors."""
        pass

    def forward(self, inputs):
        """Run layer forwards."""
        self.inputs = inputs
        return inputs

    @property
    def category(self) -> str:
        """The category, e.g. Dense."""
        return self._type

    @property
    def activation(self) -> str:
        """The activation for the layer. e.g. ReLU."""
        return self._act

class Optimizer(object):
    '''
    Parent class to all optimizers  , containing
    the `__special__` methods needed.
    '''

    def __init__(self):
        """Set property values."""
        self.output = []
        self._type = 'Undefined'

    def __repr__(self):
        """Return self as a string."""
        return f'{self.category}(output={self.output})'

    def __str__(self):
        """Return self as a string."""
        return f'{self.category}(output={self.output})'

    def __len__(self):
        """Return the length of the output."""
        return len(self.output)

    def __eq__(self, o: object):
        """Check if is equal to object."""
        try:
            return self is o
        except Exception:
            raise TypeError(
                f'Optimizer_{self.category} object is not comparable to given {type(o)} object.')

    def __hash__(self):
        """Hash the object."""
        return hash(self)

    def __bytes__(self):
        """Convert to bytes."""
        return bytes(self)

    def __enter__(self):
        """For use within a with statement."""
        return self

    def __exit__(self, category, value, traceback):
        """Placeholder to prevent exceptions."""
        pass

    @property
    def category(self) -> str:
        """The category, e.g. GradDesc."""
        return self._type

class Model(object):
    '''
    Parent class to all layers, containing
    the `__special__` methods needed.
    '''

    def __init__(self, network: list, optimizer: str = 'GradDesc',
                 loss: str = 'MSE'):
        """Create a model object of unspecified type."""
        self.biases = []
        self.weights = []
        self.output = []
        self.inputs = []
        self.optimizer = optimizer
        self.loss = loss
        self._net = network
        self._type = 'Undefined'

    def __repr__(self):
        """Return self as a string."""
        return f'{self.category}(activation={self._net})'

    def __str__(self):
        """Return self as a string."""
        return f'Layer_{self.category}(output={self.output})'

    def __len__(self):
        """Return length of layers."""
        return len(self.network)

    def __eq__(self, o: object):
        """Check if equal to other object."""
        try:
            return self is o
        except Exception:
            raise TypeError(
                f'Layer_{self.category} object is not comparable to given {type(o)} object.')

    def __hash__(self):
        """Hash object for dictionary."""
        return hash((self))

    def __bytes__(self):
        """Convert to bytes."""
        return bytes(self)

    def __enter__(self):
        """For use in a with statement."""
        return self

    def __exit__(self, category, value, traceback):
        """Placeholder to prevent exceptions."""
        pass

    def forward(self, inputs) -> list:
        """Run the whole network forward."""
        self.inputs = inputs
        return inputs

    def save(self) -> None:
        """Save object to binary file."""
        Data.save_object(self, f"{[ k for k,v in locals().items() if v == self][0]}.bin")

    @property
    def category(self) -> str:
        return self._type

    @property
    def network(self):
        return self._net

# Subclasses: Layers
class Dense(Layer):
    '''
    Create a layer with all neurons
    attached to next layer. Used a
    lot in classifiers. The best/
    default turn-to for developers.
    '''

    def __init__(self, n_inputs: int, n_neurons: int,
                 activation: str, weight_accuracy: float = 2,
                 weight_init: str='xavier', bias_init: float = 0) -> None:
        """
        Create a fully connected layer.

        ### Params

         - n_inputs = number of outputs from the previous layer or length of one line of data
         - n_neurons = number of neurons
         - activation = string name of activation
         - weight_accuracy = max decimal places in weights
         - weight_init = style of initialization
         - bias_init = value of initialization
        """
        if n_inputs <= 0:
            raise ValueError('"n_inputs" parameter should be > 0.')
        elif n_neurons <= 0:
            raise ValueError('"n_neurons" parameter should be > 0.')
        if not activation in valid_activations:
            raise ValueError(
                f'"activations" parameter must be in the list valid_activations, not {activation}.\nThe valid activations are:\n{valid_activations}')

        self.biases = []
        for _ in range(n_neurons):
            self.biases.append(bias_init)

        self.weights = []
        if weight_init == 'xavier':
            xav1_sqrt = math.sqrt(1/n_inputs)
            xav2_sqrt = math.sqrt(2/n_inputs)

        for i in range(n_neurons):
            self.weights.append([])
            for _ in range(n_inputs):
                if (weight_init == 'xavier' or weight_init == 'glorot') and activation != 'relu':
                    self.weights[i].append(round(random.normalvariate(0, 1)*xav1_sqrt, weight_accuracy))
                elif (weight_init == 'xavier' or weight_init == 'glorot') and activation == 'relu':
                    self.weights[i].append(round(random.normalvariate(0, 1)*xav2_sqrt, weight_accuracy))
                elif (weight_init == 'integer' or weight_init == 'range'):
                    self.weights[i].append(Random.random_int(-4, 4))

        self._type = 'Dense'
        self._act = activation
        self.output = []
        self.inputs = []

    def randomize(self, bias_init: int = 0) -> None:
        '''
        Randomize the weights and biases.
        Weights are randomized at the
        start, but biases are initialized
        to a default of 0.
        '''
        self.biases = []
        for _, _ in enumerate(self.biases):
            self.biases.append(bias_init)

        self.weights = []
        for i, _ in enumerate(self.biases):
            self.weights.append([])
            for _, _ in enumerate(self.weights):
                self.weights[i].append(random.normalvariate(0, 1))

    def forward(self, inputs: _ListLike) -> list:
        '''
        Run the Dense Layer forwards.
        (forward propagate). It takes the
        dot product (matrices multiplied
        together) plus the bias of each
        neuron to give an output to be
        run through the activation.
        '''
        self.output = []
        self.inputs = [inputs]

        # Dot product
        for neuron, _ in enumerate(self.biases):
            self.output.append(Decimal(float(Matrix.dot(self.weights[neuron], inputs)) + float(self.biases[neuron])))

        # Activation
        for i, _ in enumerate(self.output):
            self.output[i] = getattr(Activation, self.activation)(self.output[i]) # run activation on it

        return self.output

# Subclasses: Models
class FeedForward(Model):
    def __init__(self, network: list, optimizer: _OptimizerType = 'GradDesc',
                 loss: str = 'MSE') -> None:
        """
        Sequential model that can contain any
        layer type.

        ### Params

         - network = list of layer objects
         - optimizer = an optimizer object or string name, defaulted to 'GradDesc'
         - loss = string name for a loss

        ### Returns

         - Nothing"""
        if isinstance(optimizer, str):
            self.optimizer = globals()[optimizer]()
            if not optimizer.replace(' ', '_') in valid_optimizers:
                raise ValueError(f'{optimizer} is not a valid optimizer class.\nThe valid optimizer string names are {valid_optimizers}.')
        else:
            self.optimizer = optimizer

        for layer in network:
            for form in valid_layers:
                if (not (form in str(layer.__class__))) and (network.index(layer) == len(network) - 1):
                    raise ValueError(f'{layer.__class__} is not a valid layer class. The valid layer classes are {valid_layers}')

        self.output = []
        self.inputs = []
        self.loss = loss

        self._net = network
        self._type = 'FeedForward'

    def forward(self, inputs: _ListLike) -> list:
        self.inputs = inputs
        for layer in self.network:
            inputs = layer.forward(inputs)
        self.output = inputs
        return self.output

    def fit(self, X, y, epochs: int, disp_level: int = 1):
        return self.optimizer.train(self, X, y, epochs, disp_level)

# Subclasses: Optimizers
class GradDesc(Optimizer):
    def __init__(self, learning_rate: float = 0.0001):
        """
        Gradient descent optimizer for models, using
        regular backpropagation and simple gradient descent
        to train weights.

        ### Params

         - learning_rate = the learning rate, defaulted to 0.001: OPTIONAL

        ### Returns

         - Nothing
        """
        self.output = []
        self.learningRate = learning_rate
        self._weightGradientVector = []
        self._biasGradientVector = []
        self._type = 'GradDesc'

    def train(self, model: Model, X: _ListLike, y: _ListLike, epochs: int, disp_level:int = 1) -> list:
        '''
        Calculate the gradients and adjust the weights and
        biases for the specified model for the specified
        number of epochs.

        ### Params

         - model: a Model object with at least one layer
         - epochs: the number of epochs to train for
         - disp_level: how much to output in console
            +   -1 = display all data
            +    0 = display nothing
            +    1 = display epoch and loss

        ### Returns

         - self._history: the loss history of the model
                          so it can be plotted.
        '''
        self._disp_level = disp_level
        self._history = []

        if disp_level != 0: print()

        loss_prime = f'{model.loss}_prime'
        for epoch in range(epochs):
            for sampleIndex, sample in enumerate(X):
                model.forward(sample)

                for layerIndex, _ in enumerate(model.network):
                    act_prime = f'{model.network[-layerIndex].activation}'

                    weightLayVector = []
                    biasLayVector = []
                    gradMult = 0
                    for vector_layer_col in range(layerIndex):
                        gradMult *= reduce((lambda f, j: f * j), self._weightGradientVector[vector_layer_col])

                    for neuronIndex, _ in enumerate(model.network[-layerIndex].weights):

                        newWeightVector = []
                        for weight in range(len(model.network[-layerIndex].weights[neuronIndex])):

                            if layerIndex > 0:
                                weightGradient = getattr(
                                    Loss, loss_prime
                                    )(getattr(Activation, act_prime)(float(model.network[-layerIndex].inputs[neuronIndex][weight])), y[sampleIndex]) * gradMult * self.learningRate
                            else:
                                weightGradient = getattr(Loss, loss_prime)( float(getattr(Activation, act_prime)(float(model.network[-layerIndex].inputs[neuronIndex][weight])) ), y[sampleIndex]) * self.learningRate
                            newWeightVector.append(weightGradient)

                        if layerIndex > 0:
                                biasGradient = getattr(
                                    Loss, loss_prime
                                    )(getattr(Activation, act_prime)(1), y[sampleIndex]) * gradMult * self.learningRate
                        else:
                            biasGradient = getattr(Loss, loss_prime)( float(getattr(Activation, act_prime)( 1) ), y[sampleIndex]) * self.learningRate


                        weightLayVector.append(newWeightVector)
                        biasLayVector.append(biasGradient)

                    self._biasGradientVector.append(biasLayVector)
                    self._weightGradientVector.append(weightLayVector)

            for layerIndex, _ in enumerate(model.network):
                for neuronIndex, _ in enumerate(model.network[layerIndex].weights):
                    for weight, _ in enumerate(model.network[layerIndex].weights[neuronIndex]):
                        weight -= self._weightGradientVector[layerIndex][neuronIndex][weight]

                    model.network[layerIndex].biases[neuronIndex] -= self._biasGradientVector[layerIndex][neuronIndex]

            currLoss = statistics.fmean([getattr(Loss, model.loss)(model.forward(X[item]), y[item]) for item, _ in enumerate(X)])
            _Utils._dispGrad(epoch, currLoss, disp_level)
            self._history.append(currLoss)

        if disp_level != 0: print()

        return self._history

class RandDesc(Optimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        """
        Random optimizer that uses random changes to
        try to "brute force" it's way to an optimum.

        ### Params

        learning_rate = the learning rate, defaulted to 0.001: OPTIONAL

        ###Returns

         - Nothing


        ### Advantages

         - Easy to understand and implement for beginners

        ### Disadvangtages

         - Doesn't usually find global minimum loss
         - Slow
         - Inconsistent
        """
        self.output = []
        self.learningRate = learning_rate
        self._type = 'RandomDesc'

    def train(self, model: Model, X: list or tuple, y: list or tuple,  epochs: int, disp_level:int = 0):
        """
        Optimize the specified model object for the
        specified number of epochs.

        ### Params

        model = Model object which needs to be optimized.
        X = Training samples (X data)
        y = Training targets (labels)
        epochs = epoch iterations to train for.
        disp_level = level of information to be displayed (0 → 2)
        """
        if disp_level != 0: print()

        self.history = []
        oldLoss = float('inf')

        for epoch in range(epochs):
            losses = []
            for sample, _ in enumerate(X):
                oldWeights = [copy.copy(currLay.weights) for currLay in model.network]
                for layer, _ in enumerate(model.network):

                    if "Dense" in str(model.network[layer].__class__):
                        for neuron, _ in enumerate(model.network[layer].weights):
                            for weight, _ in enumerate(model.network[layer].weights[neuron]):
                                model.network[layer].weights[neuron][weight] += random.gauss(0, 1) * self.learningRate

                            model.network[layer].biases[neuron] += random.gauss(0, 1) * self.learningRate
                    else: raise NotImplementedError(f"Only dense layers are implemented for, not{str(model.network[layer].__class__)}")

                losses.append(getattr(Loss, model.loss)(model.forward(X[sample]), y[sample]))
            newLoss = statistics.fmean(losses)
            if newLoss <= oldLoss:
                _Utils._dispRand(epoch, newLoss, disp_level, True)
                oldLoss = newLoss
                self.history.append(float(newLoss))
            else:
                _Utils._dispRand(epoch, oldLoss, disp_level, False)
                self.history.append(float(oldLoss))
                count = 0
                for layer, weightSet in zip(model.network, oldWeights):
                    model.network[count].weights = weightSet
                    count += 1

        if disp_level != 0: print()

        return self.history


valid_activations = ['sigmoid', 'sigmoid_prime', 'relu', 'relu_prime']
valid_costs = ['MSE', 'MSE_prime', 'MAE', 'MAE_prime']
valid_layers = ['Dense']
valid_models = ['FeedForward']
valid_optimizers = ['GradDesc', 'RandDesc']

optimizer_strings = {
    'GradDesc': GradDesc(),
    'RandDesc': RandDesc()
}