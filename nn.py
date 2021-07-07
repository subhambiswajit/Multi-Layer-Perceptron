# This file consists of a 3 layered MLP
# definition trainModel has logic for feedforward and backprogation process
import numpy as np

# Utility functions/definitions
def calculateSigmoid(inputMatrix): # calculates sigmoid of every element in the matrix
    multiplynegative = np.multiply(inputMatrix, -1)
    expnegative = np.exp(multiplynegative)
    addone = np.add(expnegative, 1)
    sigmoidvalue = 1. / addone
    return sigmoidvalue

def calculateMatrixDotProduct(matrix1, matrix2): # Calculates dot product of two given matrices
    return np.dot(matrix1, matrix2)

def calculateSigmoidDerivative(sigMatrix):  # Calculates (1 - sigMatrix)*sigMatrix, (1-O)*O
    return (1 - sigMatrix) * sigMatrix

def predict(X, W1, W2): # predicts a data sample given input and relevant weights, returns in one hot encoded format
    y1 = calculateYofLayer(W1, X)
    y1 = addBiasTerm(y1)
    y2 = calculateYofLayer(W2, y1)
    b = np.zeros_like(y2)
    b[np.arange(len(y2)), y2.argmax(1)] = 1
    return b

def errorCumulative(t, O): # calculates total error of all data samples given target(t) and output(O)
    # sum of (-0.5 * (t-0)^2) of all outputs
    error = np.sum(np.sum((0.5) * ((t - O) * (t - O)), dtype=np.float64, axis=1), dtype=np.float64, axis=0)
    return error

def calculateYofLayer(W, X): # calculates sigmoid(WtX) of a any given layer
    return calculateSigmoid(calculateMatrixDotProduct(W, X.transpose())).transpose()

def calculateAccuracy(t, O): # calculates accuracy given target(t) and output(O)
    correctPreds = np.sum((t == O).all(1))
    accuracy = (correctPreds / (t.shape[0]))
    return accuracy

def addBiasTerm(X): # adds bias term 1 to all the data samples passed as a matrix
    onesMatrix = np.ones((X.shape[0], X.shape[1] + 1))
    onesMatrix[:, 1:] = X
    return onesMatrix


# 1. loads the training data and labels
# 2. initializes random weights for the start
# 3. runs epochs till error value is below a considered value
# 4. check accuracy on validate dataset
def trainModel():
    # Loading features as X and target as t
    X = np.loadtxt(open("data\\train_data.csv", "rb"), delimiter=",", dtype=np.float64)  # train data
    t = np.loadtxt(open("data\\train_labels.csv", "rb"), delimiter=",")  # train labels
    val_X = X[:2500, :]  # validation data
    val_t = t[:2500, :]  # validation label

    eeta = (1 / (X.shape[0]))  # eeta factor deciding convergence rate, taking 1/(number of training samples)

    # adding bias to X, val_X, test_t
    X = addBiasTerm(X)
    val_X = addBiasTerm(val_X)
    # Setting initial training error to -infinity
    trainingError = np.inf
    # Considering 5 nodes in hidden layer
    # Initializing first layer of weights (input and hidden layer), Matrix shape [5 X 785], including weight for bias term
    W1 = np.random.rand(5, 785) * 0.01
    # Initializing second layer of weights (hidden layer and output layer), Matrix shape [4 X 6], including weight for bias term
    W2 = np.random.rand(4, 6)

    #epochs
    iteration = 0
    while trainingError > 300:
        # feedforward starts here..
        # Calculating sigmoid(W(t)X), Actuation values for hidden layer, Matrix shape[5 X 24754], [6 X M] with bias term,
        # M = number of training rows
        if iteration == 0:
            O1 = calculateYofLayer(W1, X) # O1 is hidden layer result sigmoid(W1tX)
            O1 = addBiasTerm(O1)
        else:
            O1[:, 1:] = calculateYofLayer(W1, X)

        # Calculating sigmoid(W(t)X), Actuation values for output layer, Matrix Shape [4 X M]
        O = calculateYofLayer(W2, O1) # O output layer result

        # Calculating cost error
        trainingError = errorCumulative(t, O)
        print('Training Iteration {iteration}, Error value {error}'.format(iteration=iteration, error=trainingError))
        # feedforward ends here..

        # backpropagation starts here
        t_O = t - O  # t - O
        delta_O = t_O * calculateSigmoidDerivative(O)  # (t - O) * (1 - O) * O
        delta_w2 = eeta * calculateMatrixDotProduct(delta_O.transpose(), O1)  # eeta * (delta_O * O1)

        delta_h = calculateMatrixDotProduct(delta_O, W2) * calculateSigmoidDerivative(O1)  # delta_O * W2 * (1-O1) * (O1)
        delta_w1 = eeta * calculateMatrixDotProduct(delta_h[:, 1:].transpose(), X)  # eeta * (delta_h * X)
        # backpropagation ends here...

        # update the weights
        W1 = W1 + delta_w1
        W2 = W2 + delta_w2
        iteration = iteration + 1

    print("Updated W1", W1)
    print("Updated W2", W2)

    # Running predictions over validation set
    output = predict(val_X, W1, W2)
    # for it, val in enumerate(val_t):
    #     print('Target value {target} and Output Value {output}'.format(target=val_t[it, :], output=output[it, :]))
    print("Accuracy of the model on validation set", calculateAccuracy(val_t, output)*100)

    np.save('weight1', W1)
    np.save('weight2', W2)
    return (W1, W2)

trainModel()