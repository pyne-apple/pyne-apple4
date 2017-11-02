from numpy import *
import sys

def sigma(t):
    t = t * -1
    return float (1/(1+math.pow(2.71828, t)))

def derivative_sigma(t):
    sigman = sigma(t)
    return sigman * (1 - sigman)


if __name__ == "__main__":
    if len(sys.argv) != 5:
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

        classLists = []
        trainingLists = []
        for line in trainingFile:
            splitValues = list(int(x) for x in line.strip().replace('\t', ''))
            classLists.append([splitValues[-1]])
            splitValues.pop(-1) # Remove class
            trainingLists.append(splitValues)
        #print(classLists)
        #print(trainingLists)

        inputs = array(trainingLists)
        outputs = array(classLists)
        #print(inputs)
        #Now we have classlist, eachtraining list and headers (All we need)


        #Iterating for the number of trainingData
        #making a weights matrix initialized to 0
        w = len(trainingLists)
        h = len (headers)-1
        weights = [0 for y in range(h)]

        itr = 1
        for num in range(iterations//len(trainingLists)):
            for i in range(len(trainingLists)):
                #for each training instance
                print(weights)
                print(trainingLists[i])
                print("derivative_sigma: ",derivative_sigma(dot(weights, trainingLists[i])))
                error = classLists[i][0] - sigma(dot(weights, trainingLists[i]))
                print("class: ",classLists[i][0])
                print("error:", error)
                print("After iteration", itr,": ", end='')
                #for each attribute in training instance
                for header in headers:
                    if header == "class":
                        continue
                    weights[headers.index(header)] += (learningRate*error*trainingLists[i][headers.index(header)]*derivative_sigma(dot(weights, trainingLists[i])))
                    print ("w("+ str(header)+ ") =", round(weights[headers.index(header)], 4), end='')
                output = sigma(dot(weights, trainingLists[i]))
                print (" Output = ", round(output, 4))
                #print(weights)
                #print(output)
                itr = itr +1
                print("-------------------------")

        for i in range(iterations%len(trainingLists)):
            # for each training instance
            print(weights)
            print(trainingLists[i])
            print("derivative_sigma: ", derivative_sigma(dot(weights, trainingLists[i])))
            error = classLists[i][0] - sigma(dot(weights, trainingLists[i]))
            print("class: ", classLists[i][0])
            print("error:", error)
            print("After iteration", itr, end='')
            # for each attribute in training instance
            for header in headers:
                if header == "class":
                    continue
                weights[headers.index(header)] += (
                learningRate * error * trainingLists[i][headers.index(header)] * derivative_sigma(
                    dot(weights, trainingLists[i])))
                print("w(", str(header), ") =", round(weights[headers.index(header)], 4), end=' ')
            output = sigma(dot(weights, trainingLists[i]))
            print("Output = ", round(output, 4))
            # print(weights)
            # print(output)
            itr = itr + 1
            print("-------------------------HERE")







        #Initialize the network
        #neural_network = NN()

        # Train the network
        #neural_network.train(inputs, outputs, iterations, headers)

        # Test the neural network with a test example.
        # print(neural_network.learn(array([1, 0, 1])))