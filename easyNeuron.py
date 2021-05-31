'''
# easyNeuron
`easyNeuron` is a lightweight neural network framework written in Python for Python as one file.
It only uses Python Standard Library modules - not even numpy - to program it.

**Docstrings**
This module has extensive docstrings that are being constantly updated through time. They use
MarkDown formatting, so I am not sure if they show up properly on IDLE, but should on IDEs like
VS Code. Please raise any issues if there are terminological or grammatical issues on any docstrings.
'''

__version__ = 0.0

import math
import os
import pickle
import random
import sys
from decimal import Decimal
from pprint import pprint
from timeit import default_timer as timer

time_start = timer()

class Matrix(classmethod):
    '''
    A classmethod for matrix operations,
    since numpy isn't used here, I had
    to write my own matrix operations.
    '''
    
    def dot(list_1, list_2):
        '''
        Return the dot product between 2
        matrices (which are both 2 dimensional).
        '''
        
        return Decimal(sum(x*y for x, y in zip(list_1, list_2)))
        
    # def dot_prime(x, y):
    #     pass

    def transpose(matrix: list, disp=False):
        '''
        Returns a **transposed** matrix from the
        matrix you inputed to start with.
        
        If you set the `disp` parameter to `True`
        '''
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
    
class Timing(classmethod):
    def get_time(disp=False):
        current_time = timer()-time_start
        if disp:
            print(f'Time Elapsed: {current_time}')
        return current_time