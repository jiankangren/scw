import theano
import csv
import pymc3 as pm
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from warnings import filterwarnings
filterwarnings('ignore')
sns.set_style('white')
from pandas import read_csv
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons
from mpl_toolkits.mplot3d import Axes3D


#open csv file
X = []
Y = []
#Y = np.asarray(Y)
#with open('exp2_train_post.csv', 'r') as csvfile:
#    reader = csv.reader(csvfile, delimiter=',')
#    for row in reader:
#        X.append([float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
#        Y.append(float(row[6][1:-1].split(',')[3]))
#X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)


dataframe = read_csv('../dataset/exp2_train_post.csv', usecols=[*range(0,435)], header=None, sep=',', engine='python')
X = dataframe.values
X = X.astype('float32')
dataframe = read_csv('../dataset/exp2_train_post.csv', usecols=[435], header=None, sep=',', engine='python')
Y = dataframe.values
Y = Y.transpose()
Y = Y.astype('float32')
Y = Y[0]
#X = np.asarray(X)
#Y = np.asarray(Y)

# plot all data
#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#ax.scatter(X[:,0],X[:,1], Y,c='r')
#plt.show()


#Y = Normalizer().fit_transform(Y.reshape(1,-1))
#Y = Y.reshape(-1,)
#Y = normalize(Y)
#Y = Y.astype(int).astype(float)
min = Y.min()
Y = Y - min
max = Y.max()
Y = Y / max
X = X.astype(float)
Y = Y.astype(float)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.67)

# plot training data
#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#ax.scatter(X_train[:,0],X_train[:,1], Y_train, c='r')
#plt.show()

def construct_nn(ann_input, ann_output):
    n_hidden = 15

    # Initialize random weights between each layer
    init_1 = np.random.randn(X.shape[1], n_hidden).astype(float)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(float)
    init_3 = np.random.randn(n_hidden, n_hidden).astype(float)
    init_out = np.random.randn(n_hidden).astype(float)

    with pm.Model() as neural_network:
        # Weights from input to hidden layer
        weights_in_1 = pm.Normal('w_in_1', 0, sd=1,
                                 shape=(X.shape[1], n_hidden),
                                 testval=init_1)

        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal('w_1_2', 0, sd=1,
                                shape=(n_hidden, n_hidden),
                                testval=init_2)
        # Weights from 1st to 2nd layer
        weights_2_3 = pm.Normal('w_2_3', 0, sd=1,
                                shape=(n_hidden, n_hidden),
                                testval=init_3)

        # Weights from hidden layer to output
        weights_3_out = pm.Normal('w_3_out', 0, sd=1,
                                  shape=(n_hidden,),
                                  testval=init_out)

        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input,
                                         weights_in_1))
        act_2 = pm.math.tanh(pm.math.dot(act_1,
                                         weights_1_2))
        act_3 = pm.math.tanh(pm.math.dot(act_2,
                                         weights_2_3))
        act_out = pm.math.sigmoid(pm.math.dot(act_3,
                                              weights_3_out))

        # Binary classification -> Bernoulli likelihood
        out = pm.Normal('out',
                           act_out,0.1,
                           observed=ann_output,
                           total_size=Y_train.shape[0] # IMPORTANT for minibatches
                          )
    return neural_network

# Trick: Turn inputs and outputs into shared variables.
# It's still the same thing, but we can later change the values of the shared variable
# (to switch in the test-data later) and pymc3 will just use the new data.
# Kind-of like a pointer we can redirect.
# For more info, see: http://deeplearning.net/software/theano/library/compile/shared.html
ann_input = theano.shared(X_train)
ann_output = theano.shared(Y_train)
neural_network = construct_nn(ann_input, ann_output)


from pymc3.theanof import set_tt_rng, MRG_RandomStreams
set_tt_rng(MRG_RandomStreams(42))


minibatch_x = pm.Minibatch(X_train, batch_size=32)
minibatch_y = pm.Minibatch(Y_train, batch_size=32)

neural_network_minibatch = construct_nn(minibatch_x, minibatch_y)
with neural_network_minibatch:
    inference = pm.ADVI()
    approx = pm.fit(150000, method=inference)

#with neural_network:
#    inference = pm.ADVI()
#    approx = pm.fit(n=10000, method=inference)

trace = approx.sample(draws=10000)

plt.figure()
plt.plot(-inference.hist)
plt.ylabel('ELBO')
plt.xlabel('iteration');
plt.show()


# Predict using train data
with neural_network:
    ppc = pm.sample_ppc(trace, samples=500, progressbar=False)
# Use probability of > 0.5 to assume prediction of class 1
pred = min + max * ppc['out'].mean(axis=0)
std = min + max * ppc['out'].std(axis=0)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X_train[:,0],X_train[:,1], Y_train-pred, c='r')
plt.show()
print('Mean train error = {}'.format((Y_train - pred).mean()))

plt.figure()
plt.plot((Y_train*max)+min)
plt.plot(pred)
plt.plot(pred + std)
plt.plot(pred - std)
plt.show()

# Replace arrays our NN references with the test data
ann_input.set_value(X_test)
with neural_network:
    ppc = pm.sample_ppc(trace, samples=500, progressbar=False)
# Use probability of > 0.5 to assume prediction of class 1
pred = min + max *ppc['out'].mean(axis=0)
std = min + max *ppc['out'].std(axis=0)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X_test[:,0],X_test[:,1], Y_test-pred, c='r')
plt.show()
print('Mean error = {}'.format((Y_test - pred).mean()))

plt.figure()
plt.plot((Y_test*max)+min)
plt.plot(pred)
plt.plot(pred + std)
plt.plot(pred - std)
plt.plot(())
plt.show()

#ax.scatter(X_test[:,0],X_test[:,1], , c='b')
#ax.scatter(X_test[pred==0, 0], X_test[pred==0, 1])
#ax.scatter(X_test[pred==1, 0], X_test[pred==1, 1], color='r')
#sns.despine()
#ax.set(title='Predicted labels in testing set', xlabel='X', ylabel='Y');


# grid = pm.floatX(np.mgrid[-3:3:100j,-3:3:100j])
# grid_2d = grid.reshape(2, -1).T
# dummy_out = np.ones(grid.shape[1], dtype=np.int8)
#
#
# ann_input.set_value(grid_2d)
# ann_output.set_value(dummy_out)
#
# with neural_network:
#     ppc = pm.sample_ppc(trace, samples=500, progressbar=False)
#
# cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
# fig, ax = plt.subplots(figsize=(14, 8))
# contour = ax.contourf(grid[0], grid[1], ppc['out'].mean(axis=0).reshape(100, 100), cmap=cmap)
# ax.scatter(X_test[pred==0, 0], X_test[pred==0, 1])
# ax.scatter(X_test[pred==1, 0], X_test[pred==1, 1], color='r')
# cbar = plt.colorbar(contour, ax=ax)
# _ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='X', ylabel='Y');
# cbar.ax.set_ylabel('Posterior predictive mean probability of class label = 0');
# plt.show()
#
# cmap = sns.cubehelix_palette(light=1, as_cmap=True)
# fig, ax = plt.subplots(figsize=(14, 8))
# contour = ax.contourf(grid[0], grid[1], ppc['out'].std(axis=0).reshape(100, 100), cmap=cmap)
# ax.scatter(X_test[pred==0, 0], X_test[pred==0, 1])
# ax.scatter(X_test[pred==1, 0], X_test[pred==1, 1], color='r')
# cbar = plt.colorbar(contour, ax=ax)
# _ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='X', ylabel='Y');
# cbar.ax.set_ylabel('Uncertainty (posterior predictive standard deviation)');
# plt.show()

# plt.plot(-inference.hist)
# plt.ylabel('ELBO')
# plt.xlabel('iteration');
#
# pm.traceplot(trace);
