# -*- coding: utf-8 -*-

import tensorflow as tf
import tfdeploy as td
import numpy as np


# setup tf graph
sess = tf.Session()
x = tf.placeholder("float", shape=[None, 784], name="input")
W = tf.Variable(tf.truncated_normal([784, 100], stddev=0.05))
b = tf.Variable(tf.zeros([100]))
y = tf.nn.softmax(tf.matmul(x, W) + b, name="output")
sess.run(tf.initialize_all_variables())

# setup td model
model = td.Model()
model.add(y, sess)
inp, outp = model.get("input", "output")

# testing code
batch = np.random.rand(10000, 784)

def test_tf():
    return y.eval(session=sess, feed_dict={x: batch})

def test_td():
    return outp.eval({inp: batch})
