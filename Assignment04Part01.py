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

        classLists = []
        trainingLists = []
        for line in trainingFile:
            splitValues = list(int(x) for x in line.strip().replace('\t', ''))
            classLists.append(splitValues[-1])
            splitValues.pop(-1) # Remove class
            trainingLists.append(splitValues)
        # print(classLists)
        # print(trainingLists)

        x = np.array(trainingLists)
        y = np.array(classLists)
        # print(x)
        # print(y)

        # Weights
        weights = [[0.5] * (len(headers)-1)] # All the network weights must be initialized to 0.
        weightHeaders = []
        for header in headers:
            if header == "class":
                continue
            weightHeaders.append("w(" + header + ")")
        # print(weights)
        # print(weightHeaders)
        w = np.array(weights)
        # print(w)
        
        # For each iteration 
        for i in range(iterations):
            # print(x[i])
            # print(float(y[i]))
            # sig = sigmoid(np.dot(w, x[i]))
            output = np.dot(w, x[i])
            # print(output)
            error = float(y[i]) - sigmoid(output)
            # print(sigmoid(output))
            # print(error)
            
            # print(x[i].T)
            # adjustment = dot(inputs.T, error * self.__sigmoid_derivative(output))
            w += learningRate * error * x[i].T * sigmoidPrime(output)
            # print(sigmoidPrime(output))
            # w += learningRate * np.dot(x[i] * sigmoidPrime(output))
            print(w)

            
            

            # H = sigmoid(np.dot(X, Wh))                  # hidden layer results
            # Z = np.dot(H,Wz)                            # output layer, no activation
            # E = Y - Z                                   # how much we missed (error)
            # dZ = E * L                                  # delta Z
            # Wz +=  H.T.dot(dZ)                          # update output layer weights
            # dH = dZ.dot(Wz.T) * sigmoid_(H)             # delta H
            # Wh +=  X.T.dot(dH)  


        # For each iteration 
        # for i in range(0, iterations, 1):
            # output = sigmoid(pd.DataFrame.dot(trainingDF.iloc[i][0:-1], weightsDF.iloc[0]))
            # weightsDF.iloc[0] += pd.DataFrame.dot(trainingDF.iloc[i][0:-1], (trainingDF.iloc[i][-1] - output) * output * (1 - output))

            # For each weight
            
            # Print iteration, weights, output
            # After iteration 1: w(A1)=0.324, w(A2)=-0.021, w(A3)=3.414, output=2.353
            # weightsPrint = ""
            # for weight in weightsDF.iloc[0]:
            #     weightsPrint += weight;

            # print("After iteration " + str(i) + ": " + weightsPrint + "output=" + str(output))


    


    # when printing the weight values, print exactly the four digits after the decimal place.
    
    # Reading in data from test file
    # with open(testFileName) as testFile:


# activation function
# def sigmoid(x): 
#     return 1/(1 + np.exp(-x))      
    
# derivative of sigmoid
def sigmoidPrime(x): 
    return x * (1 - x)             

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

if __name__ == "__main__":
    main()


