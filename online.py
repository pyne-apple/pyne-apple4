from numpy import *
import sys

class NeuralNet(object):
    def __init__(self):
        # Generate random numbers
        random.seed(1)

        # Assign random weights to a 3 x 1 matrix,
        self.synaptic_weights = 2 * random.random((13, 1)) - 1

    # The Sigmoid function
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Train the neural network and adjust the weights each time.
    def train(self, inputs, outputs, training_iterations):
        for iteration in range(training_iterations):
        
            # Pass the training set through the network.
            output = self.learn(inputs)

            # Calculate the error
            error = outputs - output

            # Adjust the weights
            adjustment = dot(inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def learn(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    if len(sys.argv) != 5:
        # (1) a training file, 
        # (2) a test file, 
        # (3) a learning rate, and 
        # (4) the number of iterations to run the algorithm.
        # python experiment.py train2.dat test2.dat 0.3 400
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
            classLists.append([splitValues[-1]])
            splitValues.pop(-1) # Remove class
            trainingLists.append(splitValues)
        # print(classLists)
        # print(trainingLists)

        inputs = array(trainingLists)
        outputs = array(classLists)
        # print(inputs)
        print(outputs)

        #Initialize the network
        neural_network = NeuralNet()
        
        # The training set.
        # inputs = array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
        # outputs = array([[1, 0, 1]]).T
        # Train the network
        neural_network.train(inputs, outputs, 100000)

        # Test the neural network with a test example.
        # print(neural_network.learn(array([1, 0, 1])))