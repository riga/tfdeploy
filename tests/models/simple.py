# -*- coding: utf-8 -*-


import tensorflow as tf


sess = tf.Session()

x = tf.placeholder("float", shape=[None, 10], name="input")

W = tf.Variable(tf.truncated_normal([10, 5], stddev=0.05))
b = tf.Variable(tf.zeros([5]))

y = tf.nn.softmax(tf.matmul(x, W) + b, name="output")

sess.run(tf.initialize_all_variables())
