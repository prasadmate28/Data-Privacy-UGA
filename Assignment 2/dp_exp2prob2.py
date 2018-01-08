import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import timeit

#global variables

data = pd.read_csv('dataset/mushroomDB.txt')

#helper functions
def binarize_dataSet():
    # To read data and binarized

    # First binarize edibility column values
    new_binary_values_for_edibility = []
    for value in data['edibility']:
        if value == 'p':
            new_binary_values_for_edibility.append(int(1))
        else:
            new_binary_values_for_edibility.append(int(-1))
    data['edibility' + '0/1'] = pd.Series(new_binary_values_for_edibility).values

    for i in range(1, 23):
        if (i == 11):  # skip column number 11
            continue
        attr_splits = data.ix[:, i].unique()
        attr_name = data.ix[:, i].name

        for col_value in attr_splits:
            new_binary_attr = attr_name + '-' + col_value

            new_binary_values = []
            for value in data[data.ix[:, i].name]:
                if value == col_value:
                    new_binary_values.append(int(1))
                else:
                    new_binary_values.append(int(-1))
            data[new_binary_attr] = pd.Series(new_binary_values).values

    data.to_csv('dataset/binarizedDataSet.txt')

def logit(x):
    try:
        prob = 1/ (1 + np.exp(x))
    except Exception:
        print(Exception)
    return prob

#calculate optimum value of Beta using gradient descent
def gradient_descent(beta,alpha,num_of_iter):
    initial_likelhood_value = compute_log_likelihood(beta)
    likelihood_cost.append(initial_likelhood_value)
    for i in range(num_of_iter):
        gradient = compute_gradient(beta)
        beta = beta - alpha * gradient
        likelihood_value = compute_log_likelihood(beta)
        likelihood_cost.append(likelihood_value)

    return beta,likelihood_cost

def compute_gradient(beta):
    # using gradient descent formula
    grad = np.zeros(m)
    for t in range(n - 1):
        sig = logit(Y.T[t:t + 1] * np.dot(beta, X[t, :]))
        row_vect_response = Y.T[t:t + 1] * X[t, :]

        grad = grad + (1 - sig ) * row_vect_response
    return grad

def compute_log_likelihood(beta):
    log_cost = 0
    for t in range(n):
        row_response_vect = Y.T[t:t + 1] * np.dot(beta,X[t,:]) #beta into t th row in training data
        log_cost = log_cost + np.log(logit(row_response_vect))
    return log_cost

def model_accuracy(dataset,response,model):

    resp = np.dot(dataset,model)

    for i in range(len(resp)):
        if logit(resp[i]) > 0.5:
            resp[i] = 1
        else:
            resp[i] = -1
    print(resp)
    H = np.ones(response.shape[0]) + (np.array(resp) * np.array(response))

    acc = sum(np.array(H)) * 100 /(2*H.shape[0])
    return acc

def plot_likelihood_VS_iterations(likelihood_cost):
    print('======================  Plotting likelihood VS number of iterations ===================================')
    plt.figure(num = 1)
    plt.plot(list(range(0,num_of_iter+1)),likelihood_cost)
    plt.xlabel('No of iterations')
    plt.ylabel('Likelihood Cost')
    plt.title(' Likelihood graph')
    plt.grid(True)
    plt.show()

#initialize function calls

print('##########  Preparing Data  ###########')
cond = True
while (cond):
    print("Binarize dataset :: Yes - 0 \n No - 1")
    index = int(input("Enter the choice"))
    if index == 0:
        binarize_dataSet()
    elif index != 0 and index == 1:
        cond = False
    else: cond=True


# Data set for learning classification

binData = pd.read_csv('dataset/binarizedDataSet.txt')

data_size = binData.shape[0]
training_data_size = 7000
test_data_size = data_size - training_data_size

A = binData.loc[0:training_data_size - 1].as_matrix()  # read 7000 entries for training
B = binData.loc[training_data_size:data_size-1].as_matrix()  #read 1124 entries for testing
X = np.delete(A,np.s_[0:25],1)   #delete the redandant columns from training data
X_test = np.delete(B,np.s_[0:25],1)  #delete the redandant columns from test data
X = np.c_[np.ones(training_data_size),X]  #add ones columns in training data
X_test = np.c_[np.ones(test_data_size),X_test] #add ones column in test data
#--------------Save training and test data in CSV files-----------
np.savetxt("dataset/train.csv", X, delimiter=",")
np.savetxt("dataset/test.csv",X_test,delimiter=",")
#----------------------------------------------------------------
Y = binData['edibility0/1'].loc[0:training_data_size - 1].as_matrix().T #initialize response vector Y
Y_test = binData['edibility0/1'].loc[training_data_size:data_size].as_matrix().T
#defining parameters
n = X.shape[0] #no of rows of training data
m = X.shape[1] #no of features
beta = np.zeros(m) # initialize beta vector
alpha = 0.01 # initialize learning rate
num_of_iter = 100
likelihood_cost = [] #log likelihood function
print('$$$$$$$$$$$$$$  TRAINING LOGISTIC REGRESSION MODEL  -   MUSHROOM DATA  $$$$$$$$$$$$$$$$$$$$$$ \n')
start = timeit.default_timer()
#gradient descent
[beta,likelihood_cost] = gradient_descent(beta,alpha,num_of_iter)
stop = timeit.default_timer()
print ('Run time for training :: ',stop - start)
print('############   BeTa   AfTeR ', num_of_iter ,'iTERatIONS  ###############')
print(beta)
print('\n $$$$$$$$$$$$$$$$ Likelihood Cost  $$$$$$$$$$$$$$$$$$')
print(likelihood_cost)

#graphical interpretations

plot_likelihood_VS_iterations(likelihood_cost)
#Training Data Accuracy
print('Training accuracy achieved :: ',model_accuracy(X,Y,beta) , ' %')
#Test Data Accurancy
print('Testing accuracy achieved :: ',model_accuracy(X_test,Y_test,beta) , ' %')
