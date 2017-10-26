import math
import sys
import pandas as pd
import numpy as np

def main():
    if len(sys.argv) != 5:
        # (1) a training file, 
        # (2) a test file, 
        # (3) a learning rate, and 
        # (4) the number of iterations to run the algorithm.
        # python main.py train.dat test.dat 0.5 20
        print ("Please execute with 4 arguments <Training File> <Test File> <Learning Rate> <#Iterations>")
        exit()

    trainingFileName = sys.argv[1]
    testFileName = sys.argv[2]
    learningRate = float(sys.argv[3])
    iterations = int(sys.argv[4])
    
    # Reading in data from training file
    # with open(trainingFileName) as trainingFile:

    
    # Reading in data from test file
    # with open(testFileName) as testFile:
        




if __name__ == "__main__":
    main()


