# -*- coding: utf-8 -*-


import tensorflow as tf
import tfdeploy as td


sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 10], name="input")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

W = tf.Variable(tf.truncated_normal([10, 5], stddev=0.05))
b = tf.Variable(tf.zeros([5]))

W_drop = tf.nn.dropout(W, keep_prob)

y = tf.nn.softmax(tf.matmul(x, W_drop) + b, name="output")

if td._tf_version[:3] < (0, 12, 0):
    sess.run(tf.initialize_all_variables())
else:
    sess.run(tf.global_variables_initializer())
