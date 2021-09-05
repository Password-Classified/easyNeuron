__version__ = 1.1


import math
import os
import pickle
import random
import sys
from decimal import Decimal
from pprint import pprint
from timeit import default_timer as timer

from text_to_speech import speak as say_this

time_start = timer()


# Classmethods
class Matrix(classmethod):
    def dot(list_1, list_2):
        return Decimal(sum(x*y for x, y in zip(list_1, list_2)))
        
    def dot_prime(x, y):
        pass

    def transpose(matrix: list, disp=False):
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

class Calculus(classmethod):
    def deriv(expression: str):
        with expression as exp:
            pass

class Timing(classmethod):
    def get_time(disp=False):
        current_time = timer()-time_start
        if disp:
            print(f'Time Elapsed: {current_time}')
        return current_time
    
    def say(msg):
        say_this(msg, save=False)

class Data(classmethod):
    def load(file_to_open):
        try:
            file_to_open_data = open(file_to_open, 'r')
            data = pickle.load(bytes(file_to_open_data))
        except FileNotFoundError:
            raise FileNotFoundError(
                f'An error has occured loading file_to_open {str(file_to_open_data)}.')
        finally:
            file_to_open_data.close()

        return data

    def save(data, file_to_open):
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





# Parent Classes
class Layer(object):
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

class Activation(object):
    def __init__(self):
        self.output = []
        self._type = 'Undefined'

    def __repr__(self):
        return f'Activation_{self.type}(output={self.output})'

    def __str__(self):
        return f'Activation_{self.type}(output={self.output})'

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
                f'Activation_{self.type} object is not comparable to given {type(o)} object.')

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

    def __init__(self, n_inputs:int, n_neurons:int, bias_init:float=0):
        if n_inputs <= 0: raise ValueError('"n_inputs" parameter should be > 0.')
        elif n_neurons <= 0: raise ValueError('"n_neurons" parameter should be > 0.')
        
        self.biases = []
        for x in range(n_neurons):
            self.biases.append(bias_init)

        self.weights = []
        for i in range(n_neurons):
            self.weights.append([])
            for n in range(n_inputs):
                self.weights[i].append(random.normalvariate(0, 1))

        self._type = 'Dense'
        self.output = []

    def randomize(self, bias:int = 0):
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

class Layer_MaxPool(Layer):
    pass


# Subclasses: Activations
class Activation_Sigmoid(Activation):
    '''
    This any-layer activation was the
    most popular activation until ReLU 
    and Softmax was brought in.

    Pros
    =============
    Average time - 1 millisecond
    Can be used on any layer
    Always between 0 and 1
    Varied, analogue output

    Cons
    =============
    Slow learner
    Has cutoff point of learning
    (Will not learn any more after a
    point).
    '''

    def __init__(self):
        self.output = []
        self._type = 'Sigmoid'

    def forward(self, inputs: list or tuple):
        '''
        Run the Sigmoid activation forwards.
        (forward propagation I/O)
        '''
        self.output = []
        for i in inputs:
            self.output.append(Decimal(1+math.e**float(i)))
        return self.output

    def prime(self, inputs: list or tuple):
        self.output = []
        for i in inputs:
            self.output.append(self.forward(i) * (1-self.forward(i)))
        return self.output

class Activation_ReLU(Activation):
    def __init__(self):
        self.output = []
        self._type = 'ReLU'

    def forward(self, inputs: list or tuple):
        '''
        Run the ReLU activation forwards.
        (forward propagation I/O)
        '''
        self.output = []
        for i in inputs:
            self.output.append(Decimal(max(0, i)))
        return self.output


# Subclasses: Costs
class Cost_CrossEntropy(Cost):

    def cost(self, inputs, targets):
        self.output = 0
        for i in range(len(inputs)):
            for targ in range(len(targets[i])):
                self.output += (-1 * (targets[i][targ])) * (math.log(inputs[i]))
        return self.output

    def prime(self, inputs, targets):
        pass


# Subclasses: Optimizers
class Optimizer_GradDesc(Optimizer):
    def __init__(self, network:list, cost:object, h=0.01):
        self.output = []
        self._type = 'GradDesc'
        self.network = network
        self.cost = cost
        self.h = h
        
    def forward_cost(self, inputs, targets):
        current_in = inputs
        
        for layer in self.network:
            current_in = layer[0].forward(current_in) # Run Layer
            current_in = layer[1].forward(current_in) # Run Layer Activation
        
        cost = self.cost.cost(current_in, targets)
        return cost
                  
    def optimize(self, inputs, targets, reps=100, epoch=10000, disp=True):
        if disp: print('Optimizing Values...')
        for rep_num in range(reps):
            for data_batch in inputs: # Repeat for each 
                        
                for layer in range(len(self.network)): 
                    for neuron in range(len(self.network[layer][0].weights)):
                        for weight_set in self.network[layer][0].weights:
                            weight_index = 0
                            for weight in weight_set:
                                
                                stalled = 3
                                direction = 1
                                cost = 999

                                for iteration in range(epoch):
                                    orig_weight = weight
                                    curr_weight = weight
                                    curr_weight += self.h * direction
                                    self.network[layer][0].weights[neuron][weight_index] = curr_weight
                                    
                                    new_cost = self.forward_cost(data_batch, targets)
                                    if new_cost >= cost:
                                        stalled += 1
                                        curr_weight = orig_weight
                                        direction *= -1
                                    elif new_cost < cost:
                                        stalled = 0
                                    
                                    if stalled >= 9 or weight >= sys.maxsize - 100:
                                        break
                                    # print(curr_weight)
                                    # print(self.network[layer][0].weights[0][0])
                                
                                # print(weight, weight_index)
                                # print(self.network[layer][0].weights[neuron][weight_index])
                                # self.network[layer][0].weights[neuron][weight_index] = curr_weight
                                # print(self.network[layer][0].weights[neuron][weight_index])
                                weight_index += 1            
                    
                    
                    
                    

# DEMO #
if __name__ == '__main__':  # Will not run if imported as a module
    print()  # New Line
    ### DATA ###
    # Data for blue or red flowers
    # Red is 0, Blue is 1
    demo_data = [[2, 9, 0],
                 [8, 1, 1],
                 [11, 3, 1],
                 [4, 14, 0],
                 [20, 1, 1],
                 [15.5, 2.6, 1],
                 [5, 21, 0]]
    
    training_data = [[2, 9],
                     [8, 1],
                     [11, 3],
                     [4, 14],
                     [20, 1],
                     [15.5, 2.6],
                     [5, 21]]
    training_targets = [[0], [1], [1], [0], [1], [1], [0]]

    mystery_flower_blue = [10, 3]
    mystery_flower_red = [4, 16]

    sig = Activation_Sigmoid()
    ce = Cost_CrossEntropy()
    l = Layer_Dense(2, 1)
    grad = Optimizer_GradDesc([[l, sig]], ce, h=0.1)

    grad.optimize(training_data, training_targets, reps=30, epoch=50000)

    pred = sig.forward(l.forward(mystery_flower_red))
    
    if pred[0] > 0.5: col = 'blue'
    elif pred[0] < 0.5: col = 'red'
    elif pred[0] == 0.5: col = 'I am not sure'
    
    print('It should be red.')
    print(f'My prediction is {pred[0]}, which is equal to... {col}!')
    
    pred = sig.forward(l.forward(mystery_flower_blue))
    if pred[0] > 0.5: col = 'blue'
    elif pred[0] < 0.5: col = 'red'
    elif pred[0] == 0.5: col = 'I am not sure'
    
    print('\nIt should be blue.')
    print(f'My prediction is {pred[0]}, which is equal to... {col}!')
    ## Training Loop ##
    # for i in range(len(training_data)):                
        # l.forward(training_data[i])
        # sig.forward(l.output)
        # ce.cost(training_data[i], training_targets)

    # say('Done')
    print()
    Timing.get_time(disp=True)
    Timing.say('Your neural network has finished training.')
