# from __future__ import print_function, division
import tensorflow as tf
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

# print tensorflow version
print('Loaded TF version', tf.__version__)

a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.multiply(a, b, name="mul_c")
d = tf.add(a, b, name="add_d")
e = tf.add(c, d, name="add_e")

with tf.Session() as sess:
    print sess.run(e) # output => 23
    w = tf.summary.FileWriter("./hello_graph", sess.graph)
