import numpy as np
from numpy import linalg as ln
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special

class neuralnetwork:
  #initializing the network
  def __init__(self, input_nodes, hidden_nodes, output_nodes, learn_rate):
    #setting the number of nodes in each of the layers
    self.inodes = input_nodes
    self.hnodes = hidden_nodes
    self.onodes = output_nodes

    #set the initial weights for the network
    self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
    self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

    #setting the learning rate
    self.lr = learn_rate

    #set and apply the sigmoid function
    self.activation_function = lambda x: scipy.special.expit(x)


  #train the model
  def train(self, inputs_list, targets_list):
    inputs = np.array(inputs_list, ndmin=2).T
    target = np.array(targets_list, ndmin=2).T

    # calculate signals into hidden layer
    hidden_inputs = np.dot(self.wih, inputs)
    # calculate the signals emerging from hidden layer
    hidden_outputs = self.activation_function(hidden_inputs)
    # calculate signals into final output layer
    final_inputs = np.dot(self.who, hidden_outputs)
    # calculate the signals emerging from final output layer
    final_outputs = self.activation_function(final_inputs)

    #calculate the error
    output_error = target - final_outputs
    hidden_error = np.dot(self.who.T, output_error)

    #update the weights
    self.who += self.lr * np.dot((output_error * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

    # update the weights for the links between the input and hidden layers
    self.wih += self.lr * np.dot((hidden_error * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
    pass

  #query the model
  def query(self, inputs_list):
    inputs = np.array(inputs_list, ndmin=2).T

    # calculate signals into hidden layer
    hidden_inputs = np.dot(self.wih, inputs)
    # calculate the signals emerging from hidden layer
    hidden_outputs = self.activation_function(hidden_inputs)
    # calculate signals into final output layer
    final_inputs = np.dot(self.who, hidden_outputs)
    # calculate the signals emerging from final output layer
    final_outputs = self.activation_function(final_inputs)

    return final_outputs

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learn_rate = 0.3

#instance of a neural network
n = neuralnetwork(input_nodes, hidden_nodes, output_nodes, learn_rate)




##########TRAIN THE MODEL########
data_file = open("mnist_train_100.csv", 'r')
data_list = data_file.readlines()
data_file.close()

for record in data_list:
  all_values = record.split(',')
  inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
  # create the target output values (all 0.01, except the desired label which is 0.99)
  targets = np.zeros(output_nodes) + 0.01
  # all_values[0] is the target label for this record
  targets[int(all_values[0])] = 0.99
  n.train(inputs, targets)




##########TEST THE MODEL###########
# load the mnist test data CSV file into a list
test_data_file = open("mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

all_values = test_data_list[0].split(',');
image_array = np.asfarray(all_values[1:]).reshape((28, 28))
plt.imshow(image_array, cmap="Greys", interpolation="none")
n.query((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
