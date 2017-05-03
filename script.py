import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import pickle


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.
     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.
    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector
    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args               #train_data size is D+1 * 1 and size of labali is N*1

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))
    
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    #=======================================================================#
    #Makes an new array of size train_data.shape * 1 and fills it with 1's  #
    #then adds that to the training data set as its first columb            #
    #=======================================================================#
    train_data = np.hstack((np.ones((train_data.shape[0],1)),train_data))
    #====================#
    #transpose the weight#
    #====================#
    initialWeights_t = initialWeights.T
    
    #=============================================#
    #Tetha_n = W^t*initialWeights_n with n = 0...N#
    #Calculate the sigmoid                        #
    #=============================================#
    initialWeights_t = np.reshape(initialWeights_t,(716,1))
    teta = sigmoid(np.dot(train_data,initialWeights_t))
    #========================================================================#
    #calculation for Error following the equation provided in the description#
    #========================================================================#
    ln_teta = np.log(teta)           
    ln_minusOne_teta = np.log(1.0-teta)
    minusOne_y = 1.0 - labeli
    
    one = np.multiply(labeli,ln_teta)
    two = np.multiply(minusOne_y,ln_minusOne_teta)
    three = one + two
    
    #calculating the sumition#
    _sum = np.sum(three)
    
    #calculating the error by dividing the summition by N#
    x , y = train_data.shape
    error = _sum/x          #x = N y = D
    error = -1*error
    #=============================================================================#
    #calculation for Error_grad following the equation provided in the description#
    #=============================================================================#
    _sum2 = np.multiply((teta - labeli),train_data)
    error_grad = np.sum(_sum2,axis=0)/x
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix
    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    
    x = np.hstack((np.ones((data.shape[0], 1)), data)) #stack arrays in sequence horizontally (column wise)
    
    p_calc = sigmoid(np.dot(x, W)) #using sigmoid to do dot product of the two arrays
    
    label = np.argmax(p_calc, 1) #return the maximum value row wise (Completed by Chandola and I during office hours, test this)
   
    ##################
    # HINT: Do not forget to add the bias term to your input data


    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.
    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector
    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix
    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

## Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
pl = predicted_label.reshape(predicted_label.size,1)
print('\n Training set Accuracy:' + str(100 * np.mean((pl == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
pl = predicted_label.reshape(predicted_label.size,1)
print('\n Validation set Accuracy:' + str(100 * np.mean((pl == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
pl = predicted_label.reshape(predicted_label.size,1)
print('\n Testing set Accuracy:' + str(100 * np.mean((pl == test_label).astype(float))) + '%')

f1 = open('params.pickle', 'wb') 
pickle.dump(W, f1) 
f1.close()



"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

from sklearn.svm import SVC
import time

X = train_data
y = train_label

#use a linear kernel
f = open('linear.txt','w')
print('\n\n------\n\nlinear:')
f.write('\n\n------\n\nlinear:')
lin = SVC(kernel = 'linear')
start_time = time.time()
lin.fit(X,y.ravel())
learn_time = time.time()-start_time
print("learntime:{}".format(learn_time))
f.write("learntime:{}".format(learn_time))
training_score = lin.score(train_data,train_label.ravel())
print('\n Training set Accuracy:' + str(100 * training_score) + '%')
f.write('\n Training set Accuracy:' + str(100 * training_score) + '%')
v_score = lin.score(validation_data,validation_label.ravel())
print('\n Validation set Accuracy:' + str(100 * v_score) + '%')
f.write('\n Validation set Accuracy:' + str(100 * v_score) + '%')
test_score =  lin.score(test_data,test_label.ravel())
print('\n Testing set Accuracy:' + str(100 * test_score) + '%')
f.write('\n Testing set Accuracy:' + str(100 * test_score) + '%')
f.close()







#use default gamma
f = open('gamma_default.txt','w')
print('\n\n------\n\defaultgamma:')
f.write('\n\n------\n\ndefaultgamma:')
d_gamma = SVC()
start_time = time.time()
d_gamma.fit(X,y.ravel())
learn_time = time.time()-start_time
print("learntime:{}".format(learn_time))
f.write("learntime:{}".format(learn_time))
training_score = d_gamma.score(train_data,train_label.ravel())
print('\n Training set Accuracy:' + str(100 * training_score) + '%')
f.write('\n Training set Accuracy:' + str(100 * training_score) + '%')
v_score = d_gamma.score(validation_data,validation_label.ravel())
print('\n Validation set Accuracy:' + str(100 * v_score) + '%')
f.write('\n Validation set Accuracy:' + str(100 * v_score) + '%')
test_score =  d_gamma.score(test_data,test_label.ravel())
print('\n Testing set Accuracy:' + str(100 * test_score) + '%')
f.write('\n Testing set Accuracy:' + str(100 * test_score) + '%')
f.close()

#use gamma of 1
f = open('gamma_1.txt','w')
print('\n\n------\n\ngamma = 1:')
f.write('\n\n------\n\nngamma = 1:')
gamma1 = SVC(gamma = 1)
start_time = time.time()
gamma1.fit(X,y.ravel())
learn_time = time.time()-start_time
print("learntime:{}".format(learn_time))
f.write("learntime:{}".format(learn_time))
training_score = gamma1.score(train_data,train_label.ravel())
print('\n Training set Accuracy:' + str(100 * training_score) + '%')
f.write('\n Training set Accuracy:' + str(100 * training_score) + '%')
v_score = gamma1.score(validation_data,validation_label.ravel())
print('\n Validation set Accuracy:' + str(100 * v_score) + '%')
f.write('\n Validation set Accuracy:' + str(100 * v_score) + '%')
test_score =  gamma1.score(test_data,test_label.ravel())
print('\n Testing set Accuracy:' + str(100 * test_score) + '%')
f.write('\n Testing set Accuracy:' + str(100 * test_score) + '%')
f.close()

## do each of 1,10,20...100 for C
for i in [1.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0]:
    f = open('C_{}.txt'.format(int(i)),'w')
    print('\n\n------\n\nC = {}:'.format(i))
    f.write('\n\n------\n\nC = {}:'.format(i))
    C_fit = SVC(C = i)
    start_time = time.time()
    C_fit.fit(X,y.ravel())
    learn_time = time.time()-start_time
    print("learntime:{}".format(learn_time))
    f.write("learntime:{}".format(learn_time))
    train_acc = C_fit.score(train_data,train_label.ravel())
    print('\n Training set Accuracy:' + str(100 * train_acc) + '%')
    f.write('\n Training set Accuracy:' + str(100 * train_acc) + '%')
    v_acc = C_fit.score(validation_data,validation_label.ravel())
    print('\n Validation set Accuracy:' + str(100 * v_acc) + '%')
    f.write('\n Validation set Accuracy:' + str(100 * v_acc) + '%')
    test_acc = C_fit.score(test_data,test_label.ravel())
    print('\n Testing set Accuracy:' + str(100 * test_acc) + '%')
    f.write('\n Testing set Accuracy:' + str(100 * test_acc) + '%')
    f.close()


"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')


f2 = open('params_bonus.pickle', 'wb')
pickle.dump(W_b, f2)
f2.close()
