# Import module from parent directory setup
import inspect
import os
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from easyNeuron import *


# Demo
if __name__ == '__main__':
    print('Import ', end=''); Timing.get_time(True) # Check how long it takes to import onefile

    ### Generated Clustered Data ###    
    raw = []
    difficulty = 200  # Lower value produces harder, less clustered data
    for i in range(200):
        raw.append([random.randrange(2500, 3500)/100 + random.randrange(2500, 3500)/difficulty, random.randrange(100, 800)/100 + random.randrange(2000, 3500)/difficulty, 1])
        raw.append([random.randrange(100, 800)/100 + random.randrange(2500, 3500)/difficulty, random.randrange(2500, 3500)/100 + random.randrange(2000, 3500)/difficulty, 0])

    X = [[i[0], i[1]] for i in raw]
    y = [i[2] for i in raw]
    
    '''
    import matplotlib.pyplot as plt
    plt.figure('Data Visualisation')
    plt.title('Example Data: Flower Petals')
    plt.xlabel('Length')
    plt.ylabel('Width')
    plt.scatter(X, y)
    plt.grid()
    plt.show()
    '''
    
    model = FeedForward([
        Dense(1, 2, activation='sigmoid', weight_init='integer')
    ])
    
    print(X[1])
    print(y[1])
    model.forward(X[0])
    print(model.network[0].forward([6]))
    print(model.output)