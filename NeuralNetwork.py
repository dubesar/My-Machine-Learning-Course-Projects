import numpy as np
import pandas as pd

dataset=pd.read_csv("train (1).csv")
del dataset['Cabin']
del dataset['Ticket']
del dataset['PassengerId']
del dataset['Name']
del dataset['Embarked']
del dataset['Age']

# creating a dict file  
Sex = {'male': 1,'female': 0} 
  
# traversing through dataframe 
# Gender column and writing 
# values where key matches 
dataset.Sex = [Sex[item] for item in dataset.Sex]

dataset=np.array(dataset)
X_act=dataset[:,1:]
Y_act=dataset[:,0]

Y_act=Y_act.T

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNets:
    def __init__(self,x,y):
        self.input = x
        self.weights1=np.random.rand(self.input.shape[1],4)
        self.weights2=np.random.rand(4,1)
        self.y=y
        self.output=np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

if __name__ == "__main__":
    Y_act=dataset[:,1]
    X_act=dataset[:,2:]

    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])

    struct=NeuralNets(X,y)

    for i in range(1500):
        struct.feedforward()
        struct.backprop()

    print(struct.output)
        
