import inspect
import os
import sys
import unittest

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from easyNeuron import *
print(f'\n\nOnefile import time: {Timing.get_time()}\n')

class ActivationTester(unittest.TestCase):

    def test_relu(self):
        for i in [
            [-2, 0],
            [-1, 0],
            [0, 0],
            [1, 1],
            [2, 2]
        ]: self.assertEqual(Activation.relu(i[0]), i[1])

    def test_sigmoid(self):
        for i in [
            [-2, 0.1192],
            [-1, 0.2689],
            [0, 0.5],
            [1, 0.7310],
            [2, 0.8807]
        ]: self.assertAlmostEqual(float(Activation.sigmoid(i[0])), i[1], 3)

    def test_sigmoidPrime(self):
        for i in [
            [-2, 0.1049],
            [-1, 0.1966],
            [0, 0.25],
            [1, 0.1966],
            [2, 0.1049]
        ]: self.assertAlmostEqual(float(Activation.sigmoid_prime(i[0])), i[1], 3)

    def test_reluPrime(self):
        for i in [
            [-2, 0],
            [-1, 0],
            [0, 1],
            [1, 1],
            [2, 1]
        ]: self.assertEqual(float(Activation.relu_prime(i[0])), i[1], 3)

class DataTester(unittest.TestCase):

    def test_cities(self):
        self.assertTrue(
            Data.load_cities()[:5] == ['South Elmira', 'South Trey', 'West Hobarttown', 'Mohrstad', 'Funkmouth'])

    def test_clusterGen(self):
        is_failed = False
        try:
            Data.gen_cluster(200, 200)
        except:
            is_failed = True
        self.assertFalse(is_failed, msg='Creation of clustered data failed.')

class TimingTester(unittest.TestCase):

    def test_getTime(self):
        errType, errValue, errTraceback = None, None, None
        is_failed = False
        try: Timing.get_time()
        except:
            errType, errValue, errTraceback = sys.exc_info()
            is_failed = True

        self.assertFalse(is_failed, msg=f'{errType}: {errValue} was raised with traceback:\n\n{errTraceback}\nduring the method "Timing.get_time()"')

class LossTester(unittest.TestCase):

    def test_MSE(self):
        for i in [
            [[1, 2, 3], [1.5, 2.5, 3.5], 0.5**2/2],
        ]:
            self.assertEqual(float(Loss.MSE(i[0], i[1])), i[2])

    def test_MSEPrime(self):
        for i in [
            [ [5, 3, 4], [5.1, 3.1, 4.1], -0.1 ],
            [ [3, 6, 9], [2.5, 5.5, 8.5], 0.5]
        ]: self.assertAlmostEqual(Loss.MSE_prime(i[0], i[1]), i[2])

    def test_MAE(self):
        for i in [
            [[1, 2, 3], [1.5, 2.5, 3.5], 0.5],
        ]:
            self.assertEqual(float(Loss.MAE(i[0], i[1])), i[2])

    def test_MAEPrime(self):
        for i in [
            [ [5, 3, 4], [5.1, 3.1, 4.1], -1],
            [ [3, 6, 9], [2.5, 5.5, 8.5], 1]
        ]: self.assertAlmostEqual(Loss.MAE_prime(i[0], i[1]), i[2], msg=Loss.MAE_prime(i[0], i[1]))

class MatrixTester(unittest.TestCase):

    def test_dot(self):
        for i in [
            [[2, 3], [3, 4], 18],
            [[2, 3, 9], [9, 55, 6], 237]
        ]: self.assertEqual(Matrix.dot(i[0], i[1]), i[2])

    def test_transpose(self):
        for i in [
            [[[2, 3], [4, 5], [78, 36]], [[2, 4, 78], [3, 5, 36]]],
            [[[2], [3], [4]], [[2, 3, 4]]]
        ]: self.assertEqual(Matrix.transpose(i[0]), i[1])

    def test_depth(self):
        for i in [
            [ [[[34]]], 3],
            [ [[78]], 2],
            [ [16], 1],
            [ [[[234, 234]]], 3],
            [ [[[[345, 345]]]], 4 ],
            [ [[[[[345], [345]]]]], 5]
        ]:  self.assertEqual(Matrix.depth(i[0]), i[1])

class LayerTester(unittest.TestCase):

    def test_dense_forward(self):
        for i in [
            ['relu', [45, 678, 38], [24, 16]],
            ['sigmoid', [1, 2, 3], [1.5, 2.5]]
        ]: self.assertNotEqual(
            float(Loss.MSE([float(x) for x in Dense(3, 2, i[0]).forward(i[1])], i[2])),
            float(Loss.MSE([float(x) for x in Dense(3, 2, i[0]).forward(i[1])], i[2])),
            msg='\nThe output of the first dense layer is equivalent to that of the second.\nRUN AGAIN to check it is not just by chance.'
        )

class ModelTester(unittest.TestCase):

    def test_FeedForward(self):
        is_failed = False
        try:
            FeedForward([
                Dense(2, 4, 'relu'),
                Dense(4, 4, 'relu'),
                Dense(4, 1, 'sigmoid')
            ])
        except:
            is_failed = True
        self.assertFalse(is_failed, msg='Creation of FeedForward model object failed.')

class OptimizerTester(unittest.TestCase):

    def test_gradDesc(self):
        is_failed = False
        try:
            GradDesc()
        except:
            is_failed = True
        self.assertFalse(is_failed, msg='Creation of GradDesc optimizer object failed.')

    def test_randDesc(self):
        is_failed = False
        try:
            RandDesc()
        except:
            is_failed = True
        self.assertFalse(is_failed, msg='Creation of RandDesc optimizer object failed.')

class FullTester(unittest.TestCase):

    def test_full_rand(self):
        network = FeedForward([
            Dense(2, 1, activation='sigmoid')
        ], optimizer='RandDesc', loss='MAE')
        X, y = Data.gen_cluster(200, 1000)
        history = network.fit(X, y, 10, disp_level=1)
        self.assertGreaterEqual(history[0], history[-1],  "The loss has risen since starting optimization.")

    def test_full_grad(self):
        network = FeedForward([
            Dense(2, 1, activation='sigmoid'),
        ], loss='MAE')
        X, y = Data.gen_cluster(200, 1000)
        history = network.fit(X, y, 30, disp_level=1)
        import matplotlib.pyplot as plt
        plt.plot(history)
        plt.show()
        self.assertGreaterEqual(history[0], history[-1], f"The loss has risen since starting optimization by {abs(history[0] - history[-1])}") # * The gradient does not update the weights yet, so that shall be next, but no history is recorded yet.

if __name__ == '__main__':
    unittest.main()