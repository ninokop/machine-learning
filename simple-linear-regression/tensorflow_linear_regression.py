import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

alpha = 0.01
iters = 2000
step = 50

data = pd.read_csv('../datasets/population_location.txt', header=None, names=['Population', 'Profit'])
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
print(data.head())

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert from data frames to numpy matrices
train_X = np.array(X.values)
train_y = np.array(y.values)
n_samples = train_X.shape[0]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())
activation = W * X + b

cost = tf.reduce_sum(tf.square(activation-Y)) / (2 * n_samples)
optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(iters):
        for (x, y) in zip(train_X, train_y):
            sess.run(optimizer, feed_dict={X:x, Y:y})

        if epoch % step == 0:
            print('Epoch:%d' % (epoch+1),'cost:{:.9f}'.format(sess.run(cost,feed_dict={X:train_X,Y:train_y})),\
                  'W:',sess.run(W),'b:',sess.run(b))
    
    print("Optimization Finished!")
    print('Epoch:%d' % (epoch + 1), 'cost:{:.9f}'.format(sess.run(cost, feed_dict={X: train_X, Y: train_y})), \
          'W:', sess.run(W), 'b:', sess.run(b))

    plt.scatter(train_X, train_y,c='r',marker='o')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b))
    plt.savefig('tensorflow_linear_regression.png')
    plt.show()