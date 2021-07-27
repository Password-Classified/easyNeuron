'''
easyNeuron is a lightweight neural network framework in raw Python.
Yes, that means no extra external libraries - not even numpy! The
primary mathematical module is the math module, part of the Python
Standard Library. Every function is equipped with comprehensive 
docstrings.

The modules used are:
 - math
 - random
 - timeit

See documentation at https://www.sites.google.com/rhsstudents.co.uk/simplepython/home.

A faster, full version will also be available and uses numba to
speed code more than 8 times the speed using a JIT compiler!
Other, extra methods may also come with the full version as it
will have the capabilities of matplotlib and other libraries for
things such as graphs of data.
'''

### imports ###
from timeit import default_timer as timer
import math
import random
import pickle

start = timer()


### Exceptions ###
class easyNeuron_IterationError(Exception):
    '''
    Raised when object is not iterable for a
    function. To avoid this, use iterable
    (list, tuple or slice) objects.
    '''
    pass

class easyNeuron_SizeImbalanceError(Exception):
    '''
    Raised when object is the wrong shape/size
    for a function. To avoid this, use objects
    of the right size. E.G. Have the same num
    of inputs to a layer as the output of the
    previous layer.
    '''
    pass

class Error_Check(classmethod):
    '''
    Classmethod for built in error checks.
    '''
    def iterate_Error(inputs):
        '''
        Check that the inputed object is iterable.
        '''
        if not (type(inputs) == list or type(inputs) == tuple):
            raise easyNeuron_IterationError('The object you entered is not iterable.\nUse a list or tuple for layer I/O.')

    def size_imbalance_error(inputs, target_x, target_y):
        '''
        Check the inputed object's size/shape is usable.
        '''
        Error_Check.iterate_Error(inputs)
        if type(inputs[0]) == list:
            if target_y == 'any' and len(inputs[0]) == target_x:
                pass            
        elif type(inputs[0]) == int and target_y == 'any':
            pass            
        elif len(inputs)-1 != target_x:
            raise easyNeuron_SizeImbalanceError('The iterable object is the wrong size/shape.\nEnsure it is the correct shape for a function.')
        elif len(inputs[0])-1 != target_y:
            raise easyNeuron_SizeImbalanceError('The iterable object is the wrong size/shape.\nEnsure it is the correct shape for a function.')



### Functions ###
def get_time(write=False):
    if write:
        print(f'Time Elapsed: {timer() - start}')
    return timer() - start


def load_data(file_to_open):
    try:
        file_to_open_data = open(file_to_open, 'r')
        data = pickle.load(bytes(file_to_open_data))
    except:
        raise FileNotFoundError(f'An error has occured loading file_to_open {str(file_to_open_data)}.')
    finally:
        file_to_open_data.close()
        
    return data


def save_data(data, file_to_open):
    try:
        file_to_open_data = open(file_to_open, 'w')
        data = pickle.dump(bytes(file_to_open))
        file_to_open_data.write(data)
    except:
        raise FileExistsError(f'An error has occured saving file_to_open {str(file_to_open_data)}')
    finally:
        file_to_open_data.close()
        
    return data




###  Layers  ###
class Layer_Dense(object): 
    '''
    Create a layer with all neurons
    attached to next layer. Used a
    lot in classifiers.
    '''    
    def __init__(self, n_inputs, n_neurons):
        self.biases = []
        for x in range(n_neurons):
            self.biases.append(0)
                        
        self.weights = []
        for i in range(n_neurons):
            self.weights.append([])
            for n in range(n_inputs):
                self.weights[i].append(random.randrange(-4, 4))
                    
    def __enter__(self):
        return self.weights

    def __exit__(self, type, value, traceback):
        pass
    

    def randomize(self):
        '''
        Randomize the weights and biases.
        Weights are randomized at the
        start, but biases are initialized
        to a default of 0.
        '''
        self.biases = []
        for x in range(len(self.biases)):
            self.biases.append(random.randrange(-4, 4))
                        
        self.weights = []
        for i in range(len(self.biases)):
            self.weights.append([])
            for n in range(len(self.weights)):
                self.weights[i].append(random.randint(-4, 4))
                
              
    def forward(self, inputs):
        '''
        Run the Dense Layer forwards.
        (For forwardpropagation aka. I/O).
        It takes the dot product (matrices
        multiplied together) plus the bias
        of each neuron to give an output to
        be run through the activation.
        '''
        Error_Check.iterate_Error(inputs)
        
        self.output = []
        for neuron in range(len(self.biases)): #iterate for the num of neurons
            dot_product = self.biases[neuron]
            dot_product = sum(x*y for x,y in zip(inputs, self.weights[neuron]))
            
            self.output.append(dot_product + self.biases[neuron])
                
class Layer_Max_Pool(object):
    def forward(self, inputs):
        pass
    



### Activations ###
class Linear(object):
    '''
    This is the most basic - it isn't
    even an activation, really. It
    returns the value without activation
    unless the gradient is changed from 1.
    
    Pros
    =============
    Average time - <1 millisecond
    Non-binary
    Fastest
 
    Cons
    =============
    Bad learner
    Only on hidden layers
    '''
    def forward(self, inputs, gradient=1):
        '''
        Run the Linear activation forwards.
        (For forwardpropagation aka. I/O)
        '''
        Error_Check.iterate_Error(inputs)
        
        self.output = []
        for i in inputs:
            self.output.append(i*gradient)
        return self.output
    
    def prime(self, inputs, gradient=1):
        Error_Check.iterate_Error(inputs)
        
        self.output = []
        for i in inputs:
            self.output.append(gradient)
        return self.output
    
class Activation_Sigmoid(object):
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
    def forward(self, inputs):
        '''
        Run the Sigmoid activation forwards.
        (For forwardpropagation aka. I/O)
        '''
        Error_Check.iterate_Error(inputs)
        
        
        self.output = []
        for i in inputs:
            self.output.append(1/(1+math.e**i))
        return self.output
    
    def prime(self, inputs):
        Error_Check.iterate_Error(inputs)
           
class Activation_ReLU(object):
    '''
    Rectified Linear Unit is the most
    popular hidden layer activation.
    
    Pros
    =============
    Average time - <1 millisecond
    Fast, learns quickly
    Avoids 'vanishing gradient'
    Can be used for complex classifiers
    Simulates real neuron activity
    
    Cons
    =============
    Rarely has a dying gradient issue
    Can only be used on hidden layers
    '''
    def forward(self, inputs):
        '''
        Run the ReLU activation forwards.
        (For forwardpropagation aka. I/O)
        '''
        Error_Check.iterate_Error(inputs)
        
        self.output = []
        for i in inputs:
            self.output.append(max(0, i))
        return self.output
    
    def prime(self, inputs):
        pass

class Activation_ELU(object):
    '''
    The Exponential Linear Unit is
    an uncommon variant of ReLU,
    which can produce negative outputs.
    '''
    def forward(self, inputs):
        '''
        Run the ELU activation forwards.
        (For forwardpropagation aka. I/O)
        '''
        Error_Check.iterate_Error(inputs)
        
        self.output = []
        for i in inputs:
            if i >= 0:
                self.output.append(i)
            else:
                self.output.append(math.e**i - 1)
        return self.output

    def prime(self, inputs):
        Error_Check.iterate_Error(inputs)
        
        self.output = []
        for i in inputs:
            if i >= 0:
                self.output.append(1)
            else:
                self.output.append(math.e**i)
        return self.output

class Activation_SiLU(object):
    def forward(self, inputs):
        '''
        Run the SiLU activation forwards.
        (For forwardpropagation aka. I/O)
        '''
        Error_Check.iterate_Error(inputs)
        
        self.output = []
        for i in inputs:
            self.output.append(i/(1+math.e**(-i)))
        return self.output

    def prime(self, inputs):
        pass

class Activation_Tanh(object):
    def forward(self, inputs):
        '''
        Run the Tanh activation forwards.
        (For forwardpropagation aka. I/O)
        '''
        Error_Check.iterate_Error(inputs)
        
        self.output = []
        for i in inputs:
            self.output.append(math.tanh(i))
        return self.output
    
    def prime(self, inputs):
        pass
    
class Activation_Softmax(object):
    def forward(self, inputs):
        '''
        Run the Softmax activation forwards.
        (For forwardpropagation aka. I/O)
        '''
        Error_Check.iterate_Error(inputs)
        
        self.output = []
        e_power_sum = 0
        
        for i in inputs:
            e_power_sum += math.e**i
        for i in inputs:
            self.output.append(math.e**i / e_power_sum)
                
        return self.output
    
    def prime(self, inputs):
        pass
    
class Activation_Sofplus(object):
    def forward(self, inputs):
        '''
        Run the Softplus activation forwards.
        (For forwardpropagation aka. I/O)
        '''
        Error_Check.iterate_Error(inputs)
        
        self.output = []

        for i in inputs:
            self.output.append(math.log(1+math.e**i))
        
        return self.output

    
    def prime(self, inputs):
        pass
       
class Activation_Step(object):
    '''
    The simplest activation.
    
    Pros
    =============
    Average time - <1 millisecond 
    Simple for beginners to understand
    
    Cons
    =============
    Too simple, cannot be used for complex networks.
    Seems certain, so slow to calculate loss and
    therefore a slow learner.                
    '''
    def forward(self, inputs):
        '''
        Run the Binary Step activation forwards.
        (For forwardpropagation aka. I/O)
        '''        
        Error_Check.iterate_Error(inputs)
        
        self.output = []
        for i in inputs:
            if i > 0:
                self.output.append(1)
            else:
                self.output.append(0)
        return self.output
    
    def prime(self, inputs):
        self.output = []
        for i in inputs:
            if i != 0:
                self.output.append(0)
            else:
                self.output.append(999)
        return self.output

class Activation_Gaussian(object):
    def forward(self, inputs):
        '''
        Run the Gaussian activation forwards.
        (For forwardpropagation aka. I/O)
        '''
        Error_Check.iterate_Error(inputs)
        
        self.output = []

        for i in inputs:
            self.output.append(math.e**(-(i**2)))
        
        return self.output

    
    def prime(self, inputs):
        pass

class Activation_Argmax(object):
    def forward(self, inputs):
        '''
        Run the Argmax activation forwards.
        (For forwardpropagation aka. I/O)
        '''
        Error_Check.iterate_Error(inputs)
        
        self.output = []

        for i in inputs:
            self.output.append(0)
        
        maxIndex = 0
        for x in range(len(inputs)):
            if inputs[x] > inputs[maxIndex]:
                maxIndex = x

        self.output[maxIndex] = 1
        
        return self.output
    
    def prime(self, inputs):
        pass



### Loss ###
class Loss_CrossEntropy(object):
    
    def loss(self, inputs, targets):
        Error_Check.iterate_Error(inputs)
        Error_Check.iterate_Error(targets)

        
        self.output = 0
        
        for i in range(len(inputs)):
            self.output += ( (-1 * (targets[i])))  *  (math.log(inputs[i]) )
               
        return self.output

class Loss_SSR(object):
    
    def loss(self, inputs, targets):
        Error_Check.iterate_Error(inputs)
        Error_Check.iterate_Error(targets)

        
        self.output = 0
        
        for i in range(len(inputs)):
            self.output += (inputs[i] - targets[i])**2
               
        return self.output




### Optimizers ###
class Optimizer_Adam(object):
    def __init__(self, B1=0.9, B2=0.99):
        self.B1 = B1
        self.B2 = B2
        
    def optimize(self):
        pass
        
        ## Step 1 ##
        ## Step 2 ##
        ## Step 3 ##
        ## Step 4 ##
        ## Step 5 ##

class Optimizer_AdaGrad(object):
    pass

class Optimizer_AdaMax(object):
    pass

class Optimizer_AdaDelta(object):
    pass

class Optimizer_SGD(object):
    pass

class Optimizer_RMSProp(object):
    pass

class Optimizer_GradDesc(object):
    def __init__(self, layers, cost, h=0.001):
        self.h = h
        self.layers = layers
        self.cost_tp = cost
        
    def __enter__(self):
        return self.layers
        
    def __exit__(self, type, value, traceback):
        pass
        
    def cost(self, inputs, targets):
        Error_Check.iterate_Error(inputs)
        Error_Check.iterate_Error(targets)

        self.cost_tp.loss(inputs, targets)
        self.output = self.cost_tp.output
        return self.cost_tp.output
    
    
    def optimize(self, inputs, targets):
        print('\nOptimizing Neural Network. Please wait...')
        
        print('Optimizing weights and biases...')
        for sect in self.layers:
            for neuron in range(len(sect[0].biases)):
                print('Optimizing next weight...')
                for weight in sect[0].weights[neuron]:
                    
                    direction = 1
                    stalled = 0
                
                    while True:
                        prev_costed = self.cost(inputs, targets)
                        
                        weight += direction * self.h
                        costed = self.cost(inputs, targets)
                        
                        if costed > prev_costed:
                            direction *= -1
                            stalled += 1
                            
                        elif costed < prev_costed:
                            stalled = 0                            
                        
                        
                        if stalled > 1 or costed == prev_costed:
                            break
                    
                    print('Optimizing next bias...')
                    # sect[0].biases[neuron] is the bias
                    direction = 1
                    stalled = 0
                
                    while True:
                        prev_costed = self.cost(inputs, targets)
                        
                        sect[0].biases[neuron] += direction * self.h
                        costed = self.cost(inputs, targets)
                        
                        if costed > prev_costed:
                            direction *= -1
                            stalled += 1
                            
                        elif costed < prev_costed:
                            stalled = 0                            
                        
                        
                        if stalled > 1:
                            break
                
            
            
                
                        
                    
            



### Demo ###
if __name__ == '__main__':

    ### DATA ###
    # Data for blue or red flowers
    # Red is 0, Blue is 1
    demo_data = [[2, 9, 0],
                 [8, 1, 1],
                 [11, 3, 1],
                 [4, 14, 0],
                 [20, 1, 1],
                 [15.5, 2.6, 1],
                 [5, 21, 0]
                ]
    
    training_data = [[2, 9],
                     [8, 1],
                     [11, 3],
                     [4, 14],
                     [20, 1],
                     [15.5, 2.6],
                     [5, 21]]
    
    training_targets = [0,1,1,0,1,1,0]
    
    mystery_flower = [20, 2]  # The hoped for output is a Blue flower
    
    print(f'\nWe have the data: {training_data} as a list.')
    
    
    
    # create objects
    dense1 = Layer_Dense(2, 3)
    act1 = Activation_Sigmoid()
    
    outputLayer = Layer_Dense(3, 1)
    act2 = Activation_Sigmoid()
    
    loss = Loss_CrossEntropy()
        
    grad = Optimizer_GradDesc([[dense1, act1],
                               [outputLayer, act2]],
                              loss,
                              h = 0.1)
    
    print('Weights and biases are initialized randomly.\n\n')



    #training loop
    for i in training_data:
        # layer 1
        dense1.forward(i)
        act1.forward(dense1.output)
    
        # output
        outputLayer.forward(act1.output)
        act2.forward(outputLayer.output)
        
        loss_crossed = loss.loss(i, training_targets) # Loss using cross entropy
        
        grad.optimize(i, training_targets)

        print(f'Layer 1 output: {act1.output}')
        print(f'Final Output: {act2.output}')
        print(f'Loss: {loss_crossed}.\n\n')
    
    dense1.forward(mystery_flower)
    act1.forward(dense1.output)
    
    outputLayer.forward(act1.output)
    act2.forward(outputLayer.output)

    if act2.output > 0.5:
        print('The flower is blue!')
    elif act2.output < 0.5:
        print('The flower is red.')
    else: # It is split either way.
        print('I am really not sure!')
        
        
    print(f'\nTime Elapsed: {get_time()} seconds.')