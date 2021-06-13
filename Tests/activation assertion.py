# Import easyNeuron from parent folder for testing
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

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

print('''\n\n
==========================================
All assertion tests completed successfully.
===========================================\n''')