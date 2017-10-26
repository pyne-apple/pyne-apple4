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
        # python Assignment04Part01.py train2.dat test2.dat 0.3 400
        print ("Please execute with 4 arguments <Training File> <Test File> <Learning Rate> <#Iterations>")
        exit()

    trainingFileName = sys.argv[1]
    testFileName = sys.argv[2]
    learningRate = float(sys.argv[3])
    iterations = int(sys.argv[4])
    
    # Reading in data from training file
    with open(trainingFileName) as trainingFile:
        firstLine = trainingFile.readline()
        headers = firstLine.split()
        # print(headers)

        trainingLists = []
        for line in trainingFile:
            trainingLists.append(list(int(x) for x in line.strip().replace('\t', '')))
        # print(trainingLists)

        # Convert training data into a pandas dataframe
        trainingDF = pd.DataFrame.from_records(trainingLists, columns=headers)
        # print(trainingDF)

        # Weights
        
        weights = [[0] * (len(headers)-1)] # Weight for each attribute
        weightHeaders = []
        for header in headers:
            if header == "class":
                continue
            weightHeaders.append("w(" + header + ")")
        # print(weights)
        # print(weightHeaders)
        weightsDF = pd.DataFrame.from_records(weights, columns=weightHeaders)
        # print(weightsDF)

        # For each iteration 
        # for i in range(0, iterations, 1):
            # For each weight


            
            # Print iteration, weights, output
            # After iteration 1: w(A1)=0.324, w(A2)=-0.021, w(A3)=3.414, output=2.353



    # All the network weights must be initialized to 0.


    # when printing the weight values, print exactly the four digits after the decimal place.
    
    # Reading in data from test file
    # with open(testFileName) as testFile:
        




if __name__ == "__main__":
    main()


