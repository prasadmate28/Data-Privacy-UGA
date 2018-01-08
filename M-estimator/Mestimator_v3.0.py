import numpy as np;
import pandas as pd;
# import matplotlib.pyplot as plt;
from sklearn import linear_model;
import itertools as it;
import csv;

data = pd.read_csv('cal_housing.txt')
n = len(data)
m = 9
X_ols = data.as_matrix();
Y = data['medianHouseValue']

bw1 = 5;
bw2 = 5;
bw3 = 5;
bw4 = 5;
bw5 = 5;
bw6 = 5;
bw7 = 5;
bw8 = 5;
bw9 = 5;

bw = [np.linspace(-113, -125, bw1), np.linspace(33, 41, bw2), np.linspace(0, 53, bw3), np.linspace(0, 40000, bw4),
      np.linspace(0, 6450, bw5), np.linspace(0, 36000, bw6), np.linspace(0, 6120, bw7), np.linspace(0, 16, bw8),
      np.linspace(14000, 506000, bw9)]

mid_points_bw = [np.linspace(-113 + (bw1 / 2), -125 - (bw1 / 2), bw1), np.linspace(33 + (bw2 / 2), 41 - (bw2 / 2), bw2),
                 np.linspace(0 + (bw3 / 2), 53 - (bw3 / 2), bw3), np.linspace(0 + (bw4 / 2), 40000 - (bw4 / 2), bw4),
                 np.linspace(0 + (bw5 / 2), 6450 - (bw5 / 2), bw5), np.linspace(0 + (bw6 / 2), 36000 - (bw6 / 2), bw6),
                 np.linspace(0 + (bw7 / 2), 6120 - (bw7 / 2), bw7), np.linspace(0 + (bw8 / 2), 16 - (bw8 / 2), bw8),
                 np.linspace(14000 + (bw9 / 2), 506000 - (bw9 / 2), bw9)]

cell_center = it.product(mid_points_bw[0], mid_points_bw[1], mid_points_bw[2], mid_points_bw[3],
                         mid_points_bw[4], mid_points_bw[5], mid_points_bw[6], mid_points_bw[7], mid_points_bw[8])


# with open('temp.csv', "w") as the_file:
#     csv.register_dialect("custom", delimiter=" ", skipinitialspace=True)
#     writer = csv.writer(the_file, dialect="custom")
#     writer.writerows(cell_center)

#   A = np.genfromtxt('temp.csv',delimiter=',')
# ----------------helper functions-----------------------

### Laplace noise
def _laplaceNoise_(sensitivity, epsilon, return_size=1):
    if (epsilon > 0):
        y = sensitivity / epsilon
    else:
        y = 10

    noise = np.random.laplace(loc=0, scale=y, size=return_size)
    return noise


### oridinary least square regression
def OLS_model():
    clf = linear_model.LinearRegression()
    clf.fit(X_ols, Y)
    print('MSE for ordinary Least squre regression :: ', np.mean((Y - clf.predict(X_ols)) ** 2))
    return clf.coef_


###Pertubation
def perform_perturbation(multi_distribution):
    for i in range(20):
        for k, v in multi_distribution.items():
            multi_distribution[k] = v + int(_laplaceNoise_(2, 1.0))
            if multi_distribution[k] < 0:
                multi_distribution[k] = 0
    multi_distribution[k] = multi_distribution[k] / 20
    return multi_distribution


### Mid point of bin
def get_midpoint_of_bin(multivariate_values):
    f_vect = np.zeros((n, m))
    row = 0
    # print('multivariate values :: ',len(multivariate_values))
    for k in multivariate_values:
        vect = []
        # print('K in multivariate ' ,k)
        length = len(str(int(k)))
        # print('lenght :: ',length)
        for i in range(length):
            x = k % 2
            mid = mid_points_bw[length - i - 1][x]
            # print('value of mid ::',mid)
            vect.append(mid)
            k = k / 10
        # print('Size of vect ',len(vect))
        vect = vect[::-1]
        f_vect[row] = f_vect[row] + np.array(vect)
        row = row + 1

    print('size of F-vect', np.array(f_vect).shape)
    return f_vect


### Contrast function
def min_contrast_func(data_point, perturbed_values):
    clf = linear_model.LinearRegression()

    X = np.array(data_point)
    x_index = 0

    # multipling count with value of mid point of cell Br

    for x in data.multivariate_counts:
        X[x_index] = X[x_index] * perturbed_values[x]
        x_index = x_index + 1

    print('shape X ', X.shape)
    print('response Y ', Y.shape)

    clf.fit(np.array(X), Y)
    parameters = clf.coef_
    print('MSE for M estimator :: ', np.mean((Y - clf.predict(X_ols)) ** 2))
    return parameters


### Minimization
def minimization(perturbed_values):
    tetha = np.zeros(m)
    print('keys size', len(data.multivariate_counts))

    z = get_midpoint_of_bin((data.multivariate_counts).as_matrix())

    tetha = tetha + np.array(min_contrast_func(z, perturbed_values))

    tetha = tetha / n
    return tetha


###prepare data
def prepare_data(data):
    new_dist_col = np.zeros(n)
    k = 1;
    for i in range(m):
        discretized_column = np.digitize(data.ix[:, i], bw[i])
        for j in range(len(discretized_column)):
            new_dist_col[j] = new_dist_col[j] + k * discretized_column[j]
        k *= 10
    return new_dist_col


## --------------

multivariate_dist = prepare_data(data)
se = pd.Series(multivariate_dist)
data['multivariate_counts'] = se.values

counts = data.multivariate_counts.value_counts().values
bins = data.multivariate_counts.value_counts().index._data
multi_distribution = dict(zip(list(bins), list(counts)))

perturbed_values = perform_perturbation(multi_distribution)
print('...........Done with perturbation.............')

print('values of parameter ::', minimization(perturbed_values))

print('...........Done with Mestimator.............')
print('Parameters of OLS regression model ::', OLS_model())

print('...........Done with OLS estimation.............')
