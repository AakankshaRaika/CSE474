import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

import datetime as dt

truth_matrix = np.zeros((50000, 10))


def initializeWeights(n_in, n_out):
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    return  1 / (1 + np.exp(-z))


def preprocess():
    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    # Pick a reasonable size for validation data
    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_label = test_label_preprocess[test_perm]


    # Feature selection
    # Your code here.
    #this is reducing it to colunms vs 50k*784 values. 
    bool_index =  np.all(train_preprocess == train_preprocess[0,:] , axis = 0) #this will give me bool values for indexes that are equal in value for that columbs
    
    index_zeros = np.where(~train_preprocess.any(axis=0))[0] # this will give me indexes of all the colums with 0.
    index_Ignored_Columns = np.where(np.all(train_preprocess == train_preprocess[0,:] , axis = 0))
    print (index_zeros)

    print (index_Ignored_Columns)


    train_data = train_data[:,bool_index == False]
    train_data = train_data / 255.0

    validation_data = validation_data[:,bool_index == False]
    validation_data = validation_data / 255.0

    test_data = test_data[:,bool_index == False]
    test_data = test_data / 255.0
    
    
    global truth_matrix

    for i in range(10000):
        truth_matrix[i,train_label[i]] = 1
    


    print('preprocess done')
    return train_data, train_label, validation_data, validation_label, test_data, test_label 

def ErrorFcn(w1, w2, o, n_class, n_input):
    global truth_matrix
    y = truth_matrix
    
    A = np.log(o)
    B = np.log(1-o)

    pre = (np.multiply(y,A))
    post = (np.multiply((1-y),B))
    
    error_sum =  -(1/n_input) * (np.sum( pre) + np.sum(post))
    return error_sum
    
def reg_ErrorFcn(w1, w2, o, n_class, n_input, lamda):
    no_reg = ErrorFcn(w1, w2, o, n_class, n_input)
    reg_part = lamda/(2*10000) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    return no_reg + reg_part
    
def gradFcn(w1, w2, z, o, n_input, n_hidden, n_class, lamda, data):
    
    global truth_matrix
    y = truth_matrix

    sigma = o-y

    grad_w2 = np.zeros(( n_class, n_hidden+1, 10000))
    for image in range(10000):
        for h_node in range(n_hidden+1):
            for o_node in range(n_class):
                if (not(h_node == n_hidden)):
                    grad_w2[ o_node, h_node, image] = sigma[image, o_node] * z[image, h_node]
                else: 
                    grad_w2[ o_node, h_node, image] = sigma[image, o_node] 
    
    reg_grad_w2 = grad_w2.sum(axis=2)

    reg_grad_w2 = reg_grad_w2 + lamda*w2
    reg_grad_w2 = reg_grad_w2/10000
    
    
    grad_w1 = np.zeros((n_input+1, n_hidden, 10000))
    for image in range(10000):
        for h_node in range(n_hidden):
            for pixel in range(n_input+1):
                temp_sum = 0
                for o_node in range(n_class):
                    temp_sum += sigma[image, o_node]*w2[ o_node, h_node]
                    
                if not(pixel == n_input):
                    grad_w1[pixel, h_node] = (1-z[image,h_node])*z[image,h_node]*data[image,pixel] * temp_sum
                else:
                    grad_w1[pixel, h_node] = (1-z[image,h_node])*z[image,h_node] * temp_sum
                        
    reg_grad_w1 = grad_w1.sum(axis=2)
    reg_grad_w1 = reg_grad_w1 + lamda * w1
    reg_grad_w2 = reg_grad_w2/10000
    
    return reg_grad_w1, reg_grad_w2

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.
    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    

    # Your code here
     
    z, o, _ = nnFeedForward(w1,w2,training_data)
    
    
    
    obj_val = reg_ErrorFcn(w1, w2, o, n_class, n_input, lambdaval)
   
    grad_w1, grad_w2 = gradFcn(w1, w2, z, o, n_input, n_hidden, n_class, lambdaval, training_data)


    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    global numIters
    numIters+=1
    print("The obj fn has been called {} times. (t:{})".format(numIters,dt.datetime.now()))
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.
    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    # Your code here
    _, sig_output , labels = nnFeedForward(w1,w2,data)    
    #labels = np.argmax(sig_output, axis=1)
    
    return labels
    
def nnFeedForward(w1,w2,data):
    w1 = np.transpose(w1)
    w2 = np.transpose(w2)
#    print ("This is the W1 matrix")
#    print (w1)
#    print (" ")
#    print ("This is the W2 matrix")
#    print (w2)
#    print (np.equal.reduce(w2))
#    print (" ")
#    print ("This is the data set :")
#    print (data)
#    print (np.equal.reduce(data))
#    print (" ")
    data_y, data_x = data.shape
#    
    in_data = np.c_[data, np.ones(data_y)]
#    print ("This is data after adding bias")
#    print (in_data)
#    print (np.equal.reduce(in_data , axis = 0))
#    print (" ")
    hidden_layer = np.dot(in_data, w1)
#    print ("This is the hidden_layer")
#    print (np.equal.reduce(hidden_layer))
#    print (" ")
    sig_hidden = sigmoid(hidden_layer)
#    print ("This is the sig_hidden")
#    print (sig_hidden)
#    print (" ")
    sig_hidden_y, sig_hidden_x = hidden_layer.shape
    sig_hidden = np.c_[sig_hidden, np.ones(sig_hidden_y)]
#    
    output = np.dot(sig_hidden,w2)
#    print ("this is the output")
#    print (output)
#    print (" ")
    sig_output = sigmoid(output)
#    print ("this is the sig output")
#    print (sig_output)
#    print (" ")
    labels = np.argmax(sig_output, axis=1)
#    print ("This is the label")
#    print (labels)
    return sig_hidden[:,:-1], sig_output[:,:] , labels



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 10

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.
global numIters
numIters = 0
print("Begun minimize: {}".format(dt.datetime.now()))
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
print("End minimize: {}".format(dt.datetime.now()))
# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)
sig_h , sig_o , labelsss = nnFeedForward(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
