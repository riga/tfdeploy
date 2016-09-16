# -*- coding: utf-8 -*-

"""
Runs a performance test for tfdeploy, tensorflow@CPU and tensorflow@GPU with different data and
network dimensions. The results are written to a json file "times.json" in the current directory.
"""

import os
import time
import json
import uuid
import itertools
import collections

import numpy as np
import tensorflow as tf
import tfdeploy as td


#
# test specs and constants
#

output_file = "times.json"

specs = collections.OrderedDict()
specs["features"]  = [10, 20, 50, 100, 200, 500, 1000]
specs["examples"]  = [100000]
specs["batchsize"] = [1000, 100, 10]
specs["layers"]    = [1, 2, 5, 10]
specs["units"]     = [10, 20, 50, 100, 200, 500]


#
# Data class that contains cached random numbers and handles batch iteration
#

# static instance
data = None

# class definition
class Data(object):

    def __init__(self, max_features, max_examples):
        super(Data, self).__init__()

        self._max_features = max_features
        self._max_examples = max_examples

        self._data = np.random.rand(max_examples, max_features).astype(np.float32)

        self._features  = None
        self._examples  = None
        self._batchsize = None

    def prepare(self, features, examples, batchsize):
        if features > self._max_features:
            raise ValueError("features must be lower than max_features (%d)" % self._max_features)

        if examples > self._max_examples:
            raise ValueError("examples must be lower than max_examples (%d)" % self._max_examples)
        
        if examples % batchsize != 0:
            raise ValueError("batchsize must be a divider of examples")

        self._features  = features
        self._examples  = examples
        self._batchsize = batchsize

    def __iter__(self):
        for i in range(self._examples / self._batchsize):
            yield self._data[(i*self._batchsize):((i+1)*self._batchsize), :self._features]


#
# model generation helpers
#

def create_tf_model(features, layers, units, device, input_name, output_name):
    with tf.device(device):
        x = tf.placeholder(tf.float32, shape=[None, features], name=input_name)
        y = x
        for i in range(layers):
            W = tf.Variable(tf.random_normal([features if y == x else units, units]))
            b = tf.Variable(tf.zeros([units]))
            y = tf.tanh(tf.matmul(y, W) + b, name=output_name if i == layers - 1 else None)
    return x, y


def create_models(features, layers, units):
    postfix = str(uuid.uuid4())[:8]
    input_name = "input_" + postfix
    output_name = "output_" + postfix

    tf_cpu_x, tf_cpu_y = create_tf_model(features, layers, units, "/cpu:0", input_name, output_name)
    tf_gpu_x, tf_gpu_y = create_tf_model(features, layers, units, "/gpu:0", input_name, output_name)

    tf_sess = tf.Session()
    tf_sess.run(tf.initialize_all_variables())

    td_model = td.Model()
    td_model.add(tf_cpu_y, tf_sess)
    td_x, td_y = td_model.get(input_name, output_name)

    tf_cpu_fn = lambda batch: tf_sess.run(tf_cpu_y, feed_dict={tf_cpu_x: batch})
    tf_gpu_fn = lambda batch: tf_sess.run(tf_gpu_y, feed_dict={tf_gpu_x: batch})
    td_fn     = lambda batch: td_y.eval({td_x: batch})

    return collections.OrderedDict([
        ("tf_cpu", tf_cpu_fn),
        ("tf_gpu", tf_gpu_fn),
        ("td", td_fn)
    ])


#
# actual test function
#

def test(features, examples, batchsize, layers, units):
    # prepare the data for the given input dimensions
    data.prepare(features, examples, batchsize)

    # create models / evaluation functions
    models = create_models(features, layers, units)

    # storage for measured runtimes
    times = collections.OrderedDict((name, []) for name in models.keys())

    # loop through batches and evaluation functions
    for batch in data:
        for name, fn in models.items():
            t1 = time.time()
            fn(batch)
            times[name].append(time.time() - t1)

    for name, l in times.items():
        a = np.array(l)
        times[name] = {
            "total"   : np.sum(a),
            "mean"    : np.mean(a),
            "variance": np.var(a)
        }

    return times


#
# main and entry hook
#

def main():
    combis = list(itertools.product(*specs.values()))

    # create data with maximum shape
    global data
    print("create data")
    data = Data(max(specs["features"]), max(specs["examples"]))
    print("done")

    # run actual tests
    results = []
    for i, combi in enumerate(combis):
        print("running test %d/%d" % (i + 1, len(combis)))
        d = collections.OrderedDict(zip(specs.keys(), combi))
        d["times"] = test(**d)
        results.append(d)
        if i == 3: break

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
