import inspect
import os
import sys
import unittest

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from easyNeuron import *


class ActivationTester(unittest.TestCase):

    def test_relu(self):
        for i in [
            [-2, 0],
            [-1, 0],
            [0, 0],
            [1, 1],
            [2, 2]
        ]:
            self.assertEquals(Activation.relu(i[0]), i[1])

    def test_sigmoid(self):
        for i in [
            [-2, 0.1192],
            [-1, 0.2689],
            [0, 0.5],
            [1, 0.7310],
            [2, 0.8807]
        ]:
            self.assertAlmostEqual(float(Activation.sigmoid(i[0])), i[1], 3)
            
    def test_sigmoidPrime(self):
        for i in [
            [-2, 0.1049],
            [-1, 0.1966],
            [0, 0.25],
            [1, 0.1966],
            [2, 0.1049]
        ]:
            self.assertAlmostEqual(float(Activation.sigmoid(i[0])), i[1], 3)

if __name__ == '__main__':
    unittest.main()