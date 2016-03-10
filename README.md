<img src="https://raw.githubusercontent.com/riga/tfdeploy/master/logo.png" alt="tfdeploy logo" width="250"/>
-

Deploy [tensorflow](https://www.tensorflow.org) graphs for *insanely-fast* evaluation and export to *tensorflow-less* environments running [numpy](http://www.numpy.org).


##### Evaluation usage

```python
import tfdeploy as td
import numpy as np

model = tfdeploy.Model("/path/to/model.pkl")
inp, outp = mode.get("input", "output")

batch = numpy.random.rand(10000, 784)
result = outp.eval({inp: batch})
```


##### Installation

Via [pip](https://pypi.python.org/pypi/tfdeploy)

```bash
pip install tfdeploy
```

or by simply copying the file into your project.


## Why?

Working with tensorflow is awesome. Model definition and training is simple yet powerful, and the range of built-in features is just striking.

However, when it comes down to model deployment and evaluation things get a bit more cumbersome than they should be. You either export your graph to a new file *and* [save your trained variables](https://www.tensorflow.org/versions/master/how_tos/variables/index.html#saving-variables) in a separate file, or you make use of tensorflow's [serving system](https://www.tensorflow.org/versions/master/tutorials/tfserve/index.html). Wouldn't it be great if you could just export your model to a simple numpy-based callable? Of course it would. And this is exactly what tfdeploy does for you.

To boil it down, tfdeploy

- is lightweight. A single file with < 150 lines of core code. Just copy it to your project.
- [way faster](#performance) then using tensorflow's ``Tensor.eval``.
- **does not need tensorflow** during evaluation.
- only depends on numpy.
- can load one or more models from a single file.


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
model.add(y, sess) # y and all its ops and related tensors are added recursively
model.save("model.pkl")
```

##### Load the model and evaluate (without tensorflow)

```python
import numpy as np
import tfdeploy as td

model = td.Model("model.pkl")

# shorthand to x and y
x, y = model.get("input", "output")

# evaluate
batch = np.random.rand(10000, 784)
result = y.eval({x: batch})
```

##### Write your own ``Operation``

tfdeploy supports most of the ``Operation``'s [implemented in tensorflow](https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html). However, if you miss one (in that case, submit a PR or an issue ;) ) or if you're using custom ops, you might want to extend tfdeploy:

```python
import tensorflow as tf
import tfdeploy as td

# ... write you model here ...

# let's assume your final tensor "y" relies on an op of type "InvertedSoftmax"
# before creating the td.Model, you should add that op to tfdeploy

class InvertedSoftmax(td.Operation):
    @staticmethod
    def func(a):
        e = np.exp(-a)
        return np.divide(e, np.sum(e, axis=-1, keepdims=True))

# this is equivalent to

@td.Operation.factory
def InvertedSoftmax(a):
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
    
x2, y2 = model.get("input", "output")
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


## License

The MIT License (MIT)

Copyright (c) 2016 Marcel R.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
