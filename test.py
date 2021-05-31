from easyNeuron import *

assert Activation.sigmoid(0) == 1/2
assert Activation.sigmoid_prime(0) == 1/4

assert Activation.relu(1) == 1
assert Activation.relu_prime(1) == 1
assert Activation.relu(-1) == 0
assert Activation.relu_prime(-1) == 0