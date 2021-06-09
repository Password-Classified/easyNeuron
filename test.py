from easyNeuron import *

def check(x, y):
    if x != y:
        raise AssertionError(f'{x} is not equal to {y}')

check(Activation.sigmoid(0), 1/2)
check(Activation.sigmoid_prime(0), 1/4)

check(Activation.relu(1), 1)
check(Activation.relu_prime(1), 1)
check(Activation.relu(-1), 0)
check(Activation.relu_prime(-1), 0)