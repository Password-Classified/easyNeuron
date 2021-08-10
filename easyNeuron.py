'''
# easyNeuron
`easyNeuron` is an easy-to-use lightweight neural network framework written in raw Python.
It only uses Python Standard Library modules - not even numpy - to program it. I may also release
a full version that uses `numba` (non-standard library) to speed up code by running on the GPU.

-------------------------------------------------------------------------------------------------

## Dependencies
This module uses only Python `Standard Library` modules for it - and here they are.

 - csv
 - math
 - pickle
 - random
 - decimal
 - pprint
 - timeit

### Docstrings

This module has extensive docstrings that are being constantly updated through time. They use
MarkDown formatting, so I am not sure if they show up properly on IDLE, but should on IDEs like
VS Code. Please raise any issues if there are terminological or grammatical issues on any docstrings.


#### [Github Repository](https://github.com/Password-Classified/easyNeuron)
'''

import copy
import csv
import math
import pickle
import random
import statistics
from decimal import Decimal
from functools import reduce
from timeit import default_timer as timer

time_start = timer()

# Control Variables



# General Classmethods
class Matrix(classmethod):
    '''
    A classmethod for matrix operations,
    since numpy isn't used here, I had
    to write my own matrix operations.
    '''

    def dot(list_1: list or tuple or int or float, list_2: list or tuple or int or float) -> list:
        '''
        Return the dot product between 2
        matrices (which are both 2 or 1
        dimensional).
        '''
        if (type(list_1) == int or type(list_2) == float) and (type(list_1) == int or type(list_2) == float):
            return Decimal(list_1 * list_2) 
        else:
            return Decimal(sum(float(x)*float(y) for x, y in zip(list_1, list_2)))

    def transpose(matrix: list) -> list:
        '''
        Returns a **transposed** matrix from the
        matrix you inputed to start with.
        '''
        new = list(zip(*matrix))
        
        for i in range(len(new)):
            new[i] = list(new[i])
        
        return new
            
    def depth(inputs) -> list:
        if isinstance(inputs, list):
            return 1 + max(Matrix.depth(item) for item in inputs)
        else:
            return 0

class Timing(classmethod):
    
    def get_time(disp=False) -> float:
        current_time = timer()-time_start
        if disp:
            print(f'Time Elapsed: {current_time}')
        return current_time

class Data(classmethod):
    '''
    A classmethod for data manipulation,
    acquirement, loading and saving.
    '''

    def load_object(file_to_open) -> list:
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

    def save_object(data, file_to_open) -> str:
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

    def scale(data: list, feature_range: tuple = (0, 1)):

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
            
    def shuffle(data):
        return random.shuffle(data)

    def gen_cluster(size, difficulty):
        raw = []
        for _ in range(size):
            raw.append([random.randrange(2500, 3500)/100 + random.randrange(2500, 3500)/difficulty, random.randrange(100, 800)/100 + random.randrange(2000, 3500)/difficulty, 1])
            raw.append([random.randrange(100, 800)/100 + random.randrange(2500, 3500)/difficulty, random.randrange(2500, 3500)/100 + random.randrange(2000, 3500)/difficulty, 0])

        X = [[round(i[0], 5), round(i[1], 5)] for i in raw]
        y = [[round(i[2], 5)] for i in raw]
        
        return X, y

    def load_tutorial_data():
        raw = []
        difficulty = 200  # Lower value produces harder, less clustered data
        for i in range(200):
            raw.append([random.randrange(2500, 3500)/100 + random.randrange(2500, 3500)/difficulty, random.randrange(100, 800)/100 + random.randrange(2000, 3500)/difficulty, 1])
            raw.append([random.randrange(100, 800)/100 + random.randrange(2500, 3500)/difficulty, random.randrange(2500, 3500)/100 + random.randrange(2000, 3500)/difficulty, 0])

        return [[i[0], i[1]] for i in raw], [i[2] for i in raw] # X and y datasets

    def load_mnist():
        train_samples = []
        train_labels = []
        scaled_train_samples = []
        try:
            with open('Data/MNIST.csv') as file:
                raw = csv.reader(file.readlines())

        except:
            raise FileNotFoundError(
                'You must have the folders of data installed to load MNIST data using easyNeuron.')

    def load_dna():
        pass
    
    def load_words():
        pass

    def load_cities():        
        with open('Data/Cities.txt') as file:
            output = file.readlines()
        
        for i in range(len(output)):
            output[i] = output[i].strip('\n')
        
        return output

class Random(classmethod):
    def seed(seed: int):
        random.seed(seed)

# Network Classmethods
class Activation(classmethod):

    def sigmoid(inputs: list or tuple or float or int):
        '''
        Run the Sigmoid activation forwards.
        (for forwardpropagation)
        '''
        if type(inputs) == list or type(inputs) == tuple:
            output = []
            for i in inputs:
                output.append(Decimal(1/(1+math.e**-float(i))))
            return output

        else:
            return Decimal(1/(1+Decimal(math.e)**-Decimal(inputs)))

    def sigmoid_prime(inputs: list or tuple or float or int):
        if type(inputs) == list or type(inputs) == tuple:
            output = []
            for i in inputs:
                output.append( (1/(1+math.e**-float(i))) * (1-(1/(1+math.e**-float(i)))) )
            return output

        else:
            return (1/(1+math.e**-float(inputs))) * (1-(1/(1+math.e**-float(inputs))))

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

class Loss(classmethod):
    # TODO: add printing of inputs and targets on exception.
    def MSE(inputs: list or tuple or float or int, targets: list or tuple or float or int):
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
                    f'Inputs ({length}) has not the same size as targets ({len(targets)}).\nInputs = {inputs},\nTargets = {targets}')

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
                f'Inputs ({length}) has not the same size as targets ({len(targets)}).\nInputs = {inputs},\nTargets = {targets}')

        output = 0
        for i in range(length):
            output += (inputs[i] - targets[i])
        output /= length

        return output

    def MAE(inputs: list or tuple or float or int, targets: list or tuple or float or int):
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
                    f'Inputs ({length}) has not the same size as targets ({len(targets)}).\nInputs = {inputs},\nTargets = {targets}')

            output = 0
            for i in range(length):
                output += abs(inputs[i] - targets[i])
            output /= length

            return output

        else:
            return Decimal(abs(targets-inputs))

    def MAE_prime(inputs, targets):
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
                f'Inputs ({length}) has not the same size as targets ({len(targets)}).\nInputs = {inputs},\nTargets = {targets}')

        output = 0
        for i in range(length):
            if inputs[i] < targets[i]: output.append(-1)
            else: output.append(1)

        return output

# Parent Classes
class Layer(object):
    '''
    Parent class to all layers, containing
    the `__special__` methods needed.
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

    def __call__(self, inputs):
        return self.forward(inputs)

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

    def forward(self, inputs):
        self.inputs = inputs
        return inputs

    @property
    def type(self):
        return self._type

    @property
    def activation(self):
        return self._act

class Optimizer(object):
    '''
    Parent class to all optimizers  , containing
    the `__special__` methods needed.
    '''

    def __init__(self):
        self.output = []
        self._type = 'Undefined'

    def __repr__(self):
        return f'Optimizer - {self.type}(output={self.output})'

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

class Model(object):
    '''
    Parent class to all layers, containing
    the `__special__` methods needed.
    '''

    def __init__(self, network: list, optimizer: str = 'GradDesc',
                 loss: str = 'MSE'):
        self.biases = []
        self.weights = []
        self.output = []
        self.inputs = []
        self.optimizer = optimizer
        self.loss = loss
        self._net = network
        self._type = 'Undefined'

    def __repr__(self):
        return f'Layer_{self.type}(activation={self._net})'

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

    def forward(self, inputs) -> list:
        self.inputs = inputs
        return inputs

    @property
    def type(self):
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
                 weight_init:str='xavier', bias_init: float = 0) -> None:
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
            for n in range(n_inputs):
                if (weight_init == 'xavier' or weight_init == 'glorot') and activation != 'relu':
                    self.weights[i].append(round(random.normalvariate(0, 1)*xav1_sqrt, weight_accuracy))
                elif (weight_init == 'xavier' or weight_init == 'glorot') and activation == 'relu':
                    self.weights[i].append(round(random.normalvariate(0, 1)*xav2_sqrt, weight_accuracy))
                elif (weight_init == 'integer' or weight_init == 'range'):
                    self.weights[i].append(random.randrange(-4, 4))

        self._type = 'Dense'
        self._act = activation
        self.output = []
        self.inputs = []

    def randomize(self, bias: int = 0) -> None:
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
                self.weights[i].append(random.normalvariate(0, 1))

    def forward(self, inputs: list or tuple) -> list:
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
        for neuron in range(len(self.biases)):
            self.output.append(Decimal(float(Matrix.dot(self.weights[neuron], inputs)) + float(self.biases[neuron])))

        # Activation
        for i in range(len(self.output)):
            self.output[i] = getattr(Activation, self.activation)(self.output[i]) # run activation on it

        return self.output

# Subclasses: Models
class FeedForward(Model):
    def __init__(self, network: list, optimizer: str or object = 'GradDesc',
                 loss: str = 'MSE') -> None:
        if type(optimizer) == str:
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

    def forward(self, inputs: list or tuple) -> list:
        self.inputs = inputs
        for layer in self.network:
            inputs = layer.forward(inputs)
        self.output = inputs
        return self.output

    def fit(self, X, y, epochs: int, disp_level: int = 1):
        return self.optimizer.train(self, X, y, epochs, disp_level)

# Subclasses: Optimizers
class GradDesc(Optimizer):
    def __init__(self, learning_rate: float = 0.001):
        self.output = []
        self.gradientVector = []
        self.learningRate = learning_rate
        self._type = 'GradDesc'
        
    def disp(self, layer: int, epoch: int, loss: float):
        if self._disp_level == -1:
            print(f'Epoch {epoch}: Layer {layer} LOSS: {loss}')
        elif self._disp_level == 1:
            if epoch != self.epoch:
                print(f'Epoch {epoch}: LOSS: {loss}')
        elif self.disp_level != 0:
            raise ValueError(f'GradDesc.disp_level attribute should be between the range of -1 and 1, not {self._disp_level}. Specify this in the GradDesc.computeGradients() or Model.fit() command.')
        
        self._epoch = epoch
        
    def train(self, model: Model, X: list or tuple, y: list or tuple, epochs: int, disp_level:int = 1):
        '''
        Calculate the gradients and adjust the weights and
        biases for the specified model.
        
        Params
        ======
        
         - model: a Model object with at least one layer
         - epochs: the number of epochs to train for
         - disp_level: how much to output in console
            +   -1 = display all data
            +    0 = display nothing
            +    1 = display epoch and loss
        '''
        self._disp_level = disp_level
        
        loss_prime = f'{model.loss}_prime'
        
        for epoch in range(epochs):
            for sample in range(len(X)):
                model.forward(X[sample])
                
                for layer in range(len(model.network)):
                    act_prime = f'{model.network[-layer].activation}'
                    
                    layVector = []
                    
                    for neuron in range(len(model.network[-layer].weights)):
                        
                        newVector = []
                        for weight in range(len(model.network[-layer].weights[neuron])):                            
                            # TODO: bias optimization
                            
                            if layer > 0:
                                gradMult = 0
                                for vector_layer_col in range(layer):
                                    gradMult *= reduce((lambda f, j: f * j), self.gradientVector[vector_layer_col])
                                gradient = getattr(Loss, loss_prime)(getattr(Activation, act_prime)(model.network[-layer].inputs[weight]), y[sample]) * gradMult * self.learningRate

                            else:
                                gradient = getattr(
                                    Loss, loss_prime
                                )(getattr(
                                    Activation, act_prime
                                )(model.network[-layer].inputs[weight]), y[sample]) * self.learningRate
                                
                            newVector.append(gradient)
                            
                        layVector.append(newVector)
                            
                    self.gradientVector.append(layVector)
                    
                    self.disp(model.network.index(layer), epoch, 0.5)

        return self.gradientVector

class RandDesc(Optimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        """
        Random optimizer that uses random changes to
        try to "brute force" it's way to an optimum.
        
        Parameters
        ==========
        learning_rate = the learning rate, defaulted to 0.001: OPTIONAL
        
        Returns
        =======
        Nothing
        
        
        Advantages
        ==========
         - Easy to understand and implement for beginners
        
        Disadvangtages
        ==============
         - Doesn't usually find global minimum loss
         - Slow
         - Inconsistent
        """
        self.output = []
        self.learningRate = learning_rate
        self._type = 'RandomDesc'
        
    def _disp(self, epoch: int, loss: float, disp_level: int,  found: bool) -> None:
        """
        Display information of the epoch. Not to be used by user.
        
        Parameters
        ==========
        epoch = current epoch
        loss = current cost function output
        disp_level = amount to display
        found = whether a new weight configuration has been found
        """
        if disp_level >= 1:
            print(f'Epoch: {epoch + 1}\tLoss: {round(loss, 5)} \tNew Weights: {str(found)}')

    def train(self, model: Model, X: list or tuple, y: list or tuple,  epochs: int, disp_level:int = 1):
        """
        Optimize the specified model object for the
        specified number of epochs.
        
        Parameters
        ==========
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
            for sample in range(len(X)):
                oldWeights = [copy.copy(currLay.weights) for currLay in model.network]
                for layer in range(len(model.network)):
                    if "Dense" in str(model.network[layer].__class__):
                        for neuron in range(len(model.network[layer].weights)):
                            for weight in range(len(model.network[layer].weights[neuron])):
                                model.network[layer].weights[neuron][weight] += random.gauss(0, 1) * self.learningRate

                            model.network[layer].biases[neuron] += random.gauss(0, 1) * self.learningRate
                    else: raise NotImplementedError(f"Only dense layers are implemented for, not{str(model.network[layer].__class__)}")
                    
                losses.append(getattr(Loss, model.loss)(model.forward(X[sample]), y[sample]))
            newLoss = statistics.fmean(losses)
            if newLoss <= oldLoss:
                self._disp(epoch, newLoss, disp_level, True)
                oldLoss = newLoss
                self.history.append(float(newLoss))
            else:
                self._disp(epoch, oldLoss, disp_level, False)
                self.history.append(float(oldLoss))
                count = 0
                for layer, weightSet in zip(model.network, oldWeights):
                    model.network[count].weights = weightSet
                    count += 1
        
        if disp_level != 0: print()
        
        return self.history


valid_activations = ['sigmoid', 'sigmoid_prime', 'relu', 'relu_prime']
valid_costs = ['MSE', 'MSE_prime']
valid_layers = ['Dense']
valid_models = ['FeedForward']
valid_optimizers = ['GradDesc', 'RandDesc']

optimizer_strings = {
    'Grad_Desc': GradDesc()
}