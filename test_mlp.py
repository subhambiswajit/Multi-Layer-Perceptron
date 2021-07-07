import numpy as np
from nn import *

def test_mlp(data_file):
    # Load the test set
    test_X = np.loadtxt(open(data_file, "rb"), delimiter=",", dtype=np.float64) # test data
    test_X = addBiasTerm(test_X)


    # Load The network
    weight1 = np.load('weight1.npy')
    weight2 = np.load('weight2.npy')


    # Predict test set - one-hot encoded
    y_pred = predict(test_X, weight1, weight2)

    return y_pred


#sample test included
from acc_calc import accuracy

y_pred = test_mlp("data\\test_data.csv")
test_labels = np.loadtxt(open("data\\test_labels.csv", "rb"), delimiter=",")  # test labels
test_accuracy = accuracy(test_labels, y_pred)*100
print("Accuracy of model on test data", test_accuracy)