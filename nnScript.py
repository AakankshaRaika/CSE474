import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

import datetime as dt

import pickle

truth_matrix = np.zeros((50000, 10))
global index_selected_columns
index_selected_columns = np.zeros((1))

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
    global index_selected_columns
    index_selected_columns = np.where(np.all(train_preprocess != train_preprocess[0,:] , axis = 0))
    print(index_selected_columns)
#==============================================================================
#     print (index_zeros)
# 
#     print (index_Ignored_Columns)
# 
#==============================================================================

    train_data = train_data[:,bool_index == False]
    train_data = train_data / 255.0

    validation_data = validation_data[:,bool_index == False]
    validation_data = validation_data / 255.0

    test_data = test_data[:,bool_index == False]
    test_data = test_data / 255.0
    
    
    global truth_matrix

    for i in range(50000):
        truth_matrix[i,int(train_label[i])] = 1
    


    #print('preprocess done')
    return train_data, train_label, validation_data, validation_label, test_data, test_label 

def ErrorFcn(w1, w2, o, n_class, n_input):
    global truth_matrix
    y = truth_matrix
    
    '''REMOVE THIS'''
#==============================================================================
#     global training_label
#     y = np.zeros((2,2))
#     for i in range(2):
#         y[i,training_label[i]] = 1
#     print(y)
#==============================================================================
    
    A = np.log(o)
    B = np.log(1-o)

    pre = (np.multiply(y,A))
    post = (np.multiply((1-y),B))
    
    error_sum =  -(1/50000) * np.sum((np.multiply(y, pre) + np.multiply(1-y,post)))

    return error_sum
    
def reg_ErrorFcn(w1, w2, o, n_class, n_input, lamda):
    no_reg = ErrorFcn(w1, w2, o, n_class, n_input)

    reg_part = lamda/(2*50000) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    return no_reg + reg_part
    
def gradFcn(w1, w2, z, o, n_input, n_hidden, n_class, lamda, data):
    
    global truth_matrix
    y = truth_matrix
    
    '''REMOVE THIS'''
#==============================================================================
#     global training_label
#     y = np.zeros((2,2))
#     for i in range(2):
#         y[i,training_label[i]] = 1

    sigma = o-y
    
    #print('sigma:\n{}'.format(sigma))
    num_images, num_features = data.shape
    grad_w2 = np.zeros(( n_class, n_hidden+1, num_images))


   
    z_y, z_x = z.shape
    z_mod = np.c_[z, np.ones(z_y)]
    
    '''uses einstein summation, super convineient'''
    grad_w2 = np.einsum('io,ih->ohi',sigma,z_mod)
    
#    for image in range(num_images):
#        for h_node in range(n_hidden+1):
#            for o_node in range(n_class):
#                if (not(h_node == n_hidden)):
#                    grad_w2[ o_node, h_node, image] = sigma[image, o_node] * z[image, h_node]
#                else: 
#                    grad_w2[ o_node, h_node, image] = sigma[image, o_node] 

    reg_grad_w2 = grad_w2.sum(axis=2)
    reg_grad_w2 = reg_grad_w2 + lamda*w2
    reg_grad_w2 = reg_grad_w2/num_images
    
    data_y, data_x = data.shape
    data = np.c_[data, np.ones(data_y)]
                    
                    
    grad_w1 = np.zeros((n_input+1, n_hidden, num_images))
#    w2t = np.transpose(w2)
#    
#    global z_v
#    global s_v
#    global w2t_v
#    global gr_w1_v
#    global t_data
#    z_v = z
#    s_v = sigma
#    w2t_v = w2t
#    t_data = data
#    gr_w1_v = grad_w1
    
    #This loop adds a prohibitive amount of time

    '''uses einstein summation, super convineient'''
    pre = np.multiply(np.multiply((1-z), z), np.dot(sigma, w2[:,:-1]))
    grad_w1 = np.einsum('ih,ip->phi',pre,data)
    

#    for image in range(num_images):
#        for h_node in range(n_hidden):
#            #pre = pre * np.sum(np.multiply(sigma[image], w2t[ h_node]) )
#            for pixel in range(n_input+1):
#                grad_w1[pixel, h_node, image] = pre[image,h_node] * data[image,pixel] 

                        
    reg_grad_w1 = np.transpose(grad_w1.sum(axis=2))
    reg_grad_w1 = reg_grad_w1 + lamda * w1
    reg_grad_w1 = reg_grad_w1/num_images
    
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
    if numIters %10==0:
        print("\t--The obj fn has been called {} times. (t:{}) \t\tObjective: {}".format(numIters,dt.datetime.now(),obj_val))
    return obj_val, obj_grad


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

    data_y, data_x = data.shape
#    
    in_data = np.c_[data, np.ones(data_y)]

    hidden_layer = np.dot(in_data, w1)
    sig_hidden = sigmoid(hidden_layer)
    sig_hidden_y, sig_hidden_x = hidden_layer.shape
    sig_hidden = np.c_[sig_hidden, np.ones(sig_hidden_y)]
#    
    output = np.dot(sig_hidden,w2)

    sig_output = sigmoid(output)

    labels = np.argmax(sig_output, axis=1)
    return sig_hidden[:,:-1], sig_output[:,:] , labels

''' ---- JUST TEST THE OBJECTIVE --- '''
#==============================================================================
# n_input = 5
# n_hidden = 3
# n_class = 2
# training_data = np.array([np.linspace(0,1,num=5),np.linspace(1,0,num=5)])
# global training_label
# training_label = np.array([0,1])
# lambdaval = 0
# params = np.linspace(-5,5, num=26)
# args = (n_input, n_hidden, n_class, training_data, training_label, lambdaval)
# objval,objgrad = nnObjFunction(params, *args)
# print(objval)
# print(objgrad)
# 
# w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
# w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
# z, o, l = nnFeedForward(w1,w2,training_data)
# 
#==============================================================================
'''SHOULD OUTPUT:
    7.87167506597
[  5.53482145e-07   2.18344343e-06   3.81340471e-06   5.44336599e-06
   7.07332727e-06   7.62680942e-06   2.89592552e-03   1.05820337e-02
   1.82681419e-02   2.59542501e-02   3.36403583e-02   3.65362838e-02
   1.23940768e-01   1.00541300e-01   7.71418319e-02   5.37423637e-02
   3.03428956e-02   1.54283664e-01   2.48543366e-07   1.10300888e-03
   4.52637482e-01   4.98089172e-01   1.86316859e-06   8.15058729e-03
   4.93227000e-01   4.99915361e-01]
   '''



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
hidden_vals = [10, 50, 100, 200]
for n_hidden in hidden_vals:
    for lamdaval in range(0,81,10):

        # set the number of nodes in output unit
        n_class = 10
        
        # initialize the weights into some random matrices
        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)
        
        # unroll 2 weight matrices into single column vector
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
    
        # set the regularization hyper-parameter
        
        
        with open("test_lamda{}___hidden{}.txt".format(lamdaval,n_hidden), "w") as outf:
                    
                   
                   
            args = (n_input, n_hidden, n_class, train_data, train_label, lamdaval)
            
            # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
            
            opts = {'maxiter': 50}  # Preferred value.
            global numIters
            numIters = 0
            print("\n\n\ntest: lamda{}\t\thidden{}\n".format(lamdaval,n_hidden))
            print("test: lamda{}\t\thidden{}\n".format(lamdaval,n_hidden),file=outf)
            print("Begun minimize: {}".format(dt.datetime.now()))
            print("Begun minimize: {}".format(dt.datetime.now()),file=outf)
            nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
            print("End minimize: {}".format(dt.datetime.now()))
            print("End minimize: {}".format(dt.datetime.now()),file=outf)
            # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
            # and nnObjGradient. Check documentation for this function before you proceed.
            # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)
            
            
            # Reshape nnParams from 1D vector into w1 and w2 matrices
            w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
            w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
            
            # Test the computed parameters
            
            predicted_label = nnPredict(w1, w2, train_data)
            
            
            obj = [index_selected_columns, n_hidden, w1, w2, lamdaval]

            #Dump the data
            pickle.dump(obj, open("test_lamda{}___hidden{}.pickle".format(lamdaval,n_hidden), 'wb'))
            
            
            # find the accuracy on Training Dataset
            
            print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
            print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%',file=outf)
            predicted_label = nnPredict(w1, w2, validation_data)
            
            # find the accuracy on Validation Dataset
            
            print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
            print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%',file=outf)
            predicted_label = nnPredict(w1, w2, test_data)
            
            # find the accuracy on Validation Dataset
            
            print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
            print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%',file=outf)
            print("----------\n\n\n".format(lamdaval,n_hidden))
x