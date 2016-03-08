# tfdeploy

Deploy [tensorflow](https://www.tensorflow.org) graphs for *insanely-fast* model evaluation and export to *tensorflow-less* environments running [numpy](http://www.numpy.org).


## Why?

Working with tensorflow is awesome. Installing tensorflow on old OS's like SL6 isn't. This is quite a problem when you want to deploy your trained model to one of those machines.

tfdeploy solves this problem while only requiring numpy. It is a single file with less then 150 lines of core code, so you can easily copy it into your project. In addition, tfdeploy is [*way faster*](#performance) than using tensorflow's ``Tensor.eval``.

Install it via [pip](https://pypi.python.org/pypi/tfdeploy)

```bash
pip install tfdeploy
```

or by simply copying the file into your project.


## How?

The central class is ``tfdeploy.Model``. The following two examples demonstrate how a model can be created from a tensorflow graph, saved to and loaded from disk, and eventually evaluated.

##### Convert your graph

```python
import tensorflow as tf
import tfdeploy as td

# build your graph
sess = tf.Session()

# use names for input and output layers
x = tf.placeholder("float", shape=[None, 784], name="input")
W = tf.Variable(tf.truncated_normal([784, 100], stddev=0.05))
b = tf.Variable(tf.zeros([100]))
y = tf.nn.softmax(tf.matmul(x, W) + b, name="output")

sess.run(tf.initialize_all_variables())

# ... training ...

# create a tfdeploy model and save it to disk
model = td.Model()
model.add(y) # y and all its ops and related tensors are added recursively
model.save("model.pkl")
```

##### Load the model and evaluate (without tensorflow)

```python
import numpy as np
import tfdeploy as td

model = td.Model("model.pkl")

# shorthand to x and y
x = model.get("input")
y = model.get("output")

# evaluate
batch = np.random.rand(10000, 784)
result = y.eval({x: batch})
```

##### Write your own ``Operation``

tfdeploy supports most of the ``Operation``'s [implemented in tensorflow](https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html). However, if you miss one (in that case, submit an issue ;) ) or if you're using custom layers, you might want to extend tfdeploy:

```python
import tensorflow as tf
import tfdeploy as td

# ... write you model here ...

# let's assume your final tensor "y" relies on an op of type "InvertedSoftmax"
# before creating the td.Model, you should add that op to tfdeploy

class InvertedSoftmax(td.Operation):
    @classmethod
    def func(a):
        e = np.exp(-a)
        return np.divide(e, np.sum(e, axis=-1, keepdims=True))

# now, we're good to go
model = td.Model()
model.add(y)
model.save("model.pkl")
```


## Performance

tfdeploy is lightweight (1 file, < 150 lines of core code) and fast. Internal operations are nearly overhead-free. All mathematical operations use numpy vectorization. On average, evaluation is *70%* faster than plain tensorflow. (tba: test with large-scale network)

##### Test code (based on ["Convert your graph"](#convert-your-graph))

```python
batch = np.random.rand(10000, 784)

def test_tf():
    return y.eval(session=sess, feed_dict={x: batch})
    
x2 = model.get("input")
y2 = model.get("output")
def test_td():
    return y2.eval({x2: batch})
```

ipython shell:

```bash
In [1]: %timeit test_tf()
100 loops, best of 3: 8.78 ms per loop

In [2]: %timeit test_td()
100 loops, best of 3: 2.63 ms per loop

In [3]: 2.63/8.78
Out[3]: 0.2995444191343964
```


## Development

- Source hosted at [GitHub](https://github.com/riga/tfdeploy)
- Report issues, questions, feature requests on [GitHub Issues](https://github.com/riga/tfdeploy/issues)


## Authors

- Marcel R. ([riga](https://github.com/riga))
- Benjamin F. ([bfis](https://github.com/bfis))
