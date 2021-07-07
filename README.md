# Multi-Layer-Perceptron
This repository has an elementary implementation of a Multi Layer perceptron with 3 Layers and 5 Nodes in the hidden layer

![alt text](img/001.png)
<br>

#### Note: Implementation has input with 785 features. Sorry ! couldn't accomodate that many in the picture.


### The implementation is completely done using numpy.
<br>

## Directory and files description:
<br>

### 1. data: This has the datasets we used for training and testing the model.
### 2. acc_calc.py: utility for checking accuracy.
### 3. nn.py: This file has the training algorithms including backpropagation process
#### Utilities defined and their usages:
* calculateSigmoid ! calculates sigmoid of every element in the matrix.
* calculateMatrixDotProduct ! calculates dot product of two given matrices.
* calculateSigmoidDerivative ! Calculates (1 - sigMatrix)*sigMatrix, (1-O)*O
* predict ! predicts a data sample given input and relevant weights, returns in one hot
encoded format.
* errorCumulative!calculates total error of all data samples given target(t) and output(O).
* calculateYofLayer ! calculates sigmoid(WtX) of a any given layer.
* calculateAccuracy ! calculates accuracy given target(t) and output(O).
* addBiasTerm ! adds bias term 1 to all the data samples passed as a matrix
* trainModel !
1. Loads the training data and labels.
2. Initializes random weights for the start.
3. Runs epochs with feed forward and backpropagation logic till error value is below a
considered value.
4. Check accuracy on validate dataset.
5. Saves the trained weights.

### test_mlp.py
#### Utilities defined and their usages:
* Takes path of directory where test data exists.
* Loads the model.
* calls tarinModel from nn.py and gets the weights saved
* Predicts the data test samples.

### Sample console output
![alt text](img/002.png)

### How do i run it ? 
#### python test_mlp.py 
#### Watch the legendary neural network in action ! haha.. cheers 

