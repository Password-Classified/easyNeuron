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
            self.assertAlmostEqual(float(Activation.sigmoid_prime(i[0])), i[1], 3)
    
    def test_reluPrime(self):
        for i in [
            [-2, 0],
            [-1, 0],
            [0, 1],
            [1, 1],
            [2, 1]
        ]:
            self.assertEqual(float(Activation.relu_prime(i[0])), i[1], 3)


class DataTester(unittest.TestCase):

    def test_cities(self):
        self.assertTrue(
            Data.load_cities()[:5] == ['South Elmira', 'South Trey', 'West Hobarttown', 'Mohrstad', 'Funkmouth'])

class LossTester(unittest.TestCase):
    
    def test_MSE(self):
        for i in [
            [[1, 2, 3], [1.5, 2.5, 3.5], 0.5**2/2],   
        ]:
            self.assertEqual(float(Costs.MSE(i[0], i[1])), i[2])
            
class MethodTester(unittest.TestCase):
    
    def test_dot(self):
        for i in [
            [[2, 3], [3, 4]]
        ]: pass

if __name__ == '__main__':
    unittest.main()