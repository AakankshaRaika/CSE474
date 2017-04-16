import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

# Andrew
def splitArray(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    # Outputs
    # X_list - A list of 5 d x k matrices contaiing training values split by the associated y value
    # min_class - minimum class valule -- assumed integer
    # max_class - maximum class value -- assumed integer
    
    samples, features = np.shape(X)
    max_class = int(np.amax(y))
    min_class = int(np.amin(y))
    y = np.concatenate((y,y),axis=1)
    X_list = [[],[],[],[],[]]
    for classification in range(0, max_class-min_class+1):
        X_list[classification] = ( X[ np.where(y == classification + min_class)] ).reshape((-1, features))
    
    return X_list, min_class, max_class


# Andrew
def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    X_split, min_class, max_class = splitArray(X,y)
    
    means =  [[],[],[],[],[]]
    for classification in range(max_class-min_class+1):
        means[classification] = np.mean(X_split[classification],axis=0)
    covmat = np.cov(X.T,bias=1)
    return means,covmat

# Andrew
def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    X_split, min_class, max_class = splitArray(X,y)
    means =  [[],[],[],[],[]]
    covmats = [[],[],[],[],[]]
    for classification in range(max_class-min_class+1):
        means[classification] = np.mean(X_split[classification],axis=0)
        covmats[classification] =  np.cov(X_split[classification].T,bias=1)

    return means,covmats

# Andrew
def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    
    acc, ypred = qdaTest(means,[covmat,covmat,covmat,covmat,covmat],Xtest,ytest)

    return acc,ypred

# Andrew
def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    ypred = []
    scores = [[],[],[],[],[]]
    

    for category in range(5):
        for sample in range(len(ytest)):
            factor = 1/np.sqrt(np.linalg.norm(covmats[category]))
            
            mean_diffs = Xtest[sample] - means[category]
            mahalanobis = mean_diffs.T.dot(np.linalg.inv(covmats[category]).dot(mean_diffs))

            scores[category].append( factor * np.exp(-1*mahalanobis) )
        
    ypred = []
    for sample in range(len(ytest)):
        sample_scores = [scores[0][sample], scores[1][sample],
                         scores[2][sample], scores[3][sample],
                         scores[4][sample]]    
        predict = int(sample_scores.index(max(sample_scores))) + 1
        ypred.append(predict)

    ypred = np.asarray([ypred])
    num_correct = np.sum(ypred.T == ytest)
    
    acc = num_correct / len(ytest)
 
    return acc,ypred


# Aakanksha : Problem 2 
def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    # converstions from the 2nd j(w)formula given
    transposed_x = np.transpose(X)                       #flip flops the matrix
    #manupulating the dimensions for proper calculations
    dot_x = np.dot(transposed_x , X)			 #product of x and the flop(inv) of x
    dot_y = np.dot(transposed_x , y)                     #product of y and the flop(inv) of x
    inverse = np.linalg.inv(dot_x)                       #inverse of the dot_x
    w = np.dot(inverse, dot_y)                           #calculating the wight
    return w

#Aakanksha : Problem 2
def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse

    # IMPLEMENT THIS METHOD
    #literal converstion from the formula given
    xw = np.dot(Xtest,w)
    sub = np.subtract(ytest,xw)
    sqDif = np.square(sub)                                #subtracts the dot product of the w and the x from $
    rmse = np.sum(sqDif)                                  #performs a summition
    N = Xtest.shape[0]                                    #gets the sphape of xtest so that we can get N
    mse = np.divide(rmse,N)
    return mse

# Andrew
def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD    
    # https://onlinecourses.science.psu.edu/stat857/node/155
    
    N, d = np.shape(X)
    
    XT_dot_X = np.dot(X.T,X)
    lambd_eye = lambd*np.eye(d)
    sum_X_I_inv = np.linalg.inv( XT_dot_X + lambd_eye )
    XT_dot_y = np.dot(X.T, y)
    w = np.dot(sum_X_I_inv, XT_dot_y)
	
    return w

# Dom : Problem 4
def regressionObjVal(w, X, y, lambd):
    
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda
    
    # IMPLEMENT THIS METHOD
    #https://www.cs.utah.edu/~piyush/teaching/6-9-print.pdf
    N = X.shape[0]
    w = np.mat(w).T
    
    #error = (((y - X.dot(w)).T).dot((y - X.dot(w))) / (2*N)) + ((lambd * ((w.T).dot(w))) / 2)
    error = (((y - X.dot(w)).T).dot((y - X.dot(w)))) + ((lambd * ((w.T).dot(w))) / 2)
    #error_grad = (((((w.T).dot((X.T).dot(X))) - ((y.T).dot(X))) / N) + ((w.T) * lambd)).T
    error_grad = (((((w.T).dot((X.T).dot(X))) + ((y.T).dot(X))) / N) + ((w.T) * lambd)).T
    error_grad = np.ndarray.flatten(np.array(error_grad))
    return error, error_grad

# Aakanksha : Problem 5
def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1)) 
	
    # IMPLEMENT THIS METHOD
    shape = x.shape[0]               # gets the shape of x
    N = p+1                          # N = p + 1
    Xd = np.ones((shape,N))          # matrix Xd = (shape of x ) x N
    for i in range(1, N):            # looping to assign through out the
        Xd[:, i] = math.pow(x,i)     # assign x^i to the array xd we are making it linear
    return Xd


# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)
print("MSE without intercept mean of magnitude of weights: {}".format(np.mean(np.absolute(w))))

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
print("MSE with intercept mean of magnitude of weights: {}".format(np.mean(np.absolute(w_i))))

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
w_l_list = []
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    w_l_list.append(w_l)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()

w_l = w_l_list[np.argmin(mses3)]
print("Ridge Regresion, mean of magnitude of weights: {}".format(np.mean(np.absolute(w_i))))

min_lambda = lambdas[np.argmin(mses3)]
print("The minimum MSE for Test Data occurs at lambda: {} with an MSE of {}".format(min_lambda, float(mses3[np.argmin(mses3)])))

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(mses3)] 
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
