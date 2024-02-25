<img src="https://raw.githubusercontent.com/riga/tfdeploy/master/logo.png" alt="tfdeploy logo" width="250"/>

[![Build Status](https://travis-ci.org/riga/tfdeploy.svg?branch=master)](https://travis-ci.org/riga/tfdeploy) [![Documentation Status](https://readthedocs.org/projects/tfdeploy/badge/?version=latest)](http://tfdeploy.readthedocs.org/en/latest/?badge=latest) [![Package Status](https://badge.fury.io/py/tfdeploy.svg)](https://badge.fury.io/py/tfdeploy)

Deploy [tensorflow](https://www.tensorflow.org) graphs for *fast* evaluation and export to *tensorflow-less* environments running [NumPy](http://www.numpy.org).

> [!NOTE]  
> This project started as a personal playground to get an in-depth understanding of TensorFlow's operations and kernels.
> Up to a certain version, the NumPy based operations in tfdeploy provided full feature parity, but it is obvious that such a project cannot keep up with the vast development speed driven by TensorFlow devs and the open-source community.
>
> Therefore, tfdeploy is **no longer actively maintained**.
> However, the code base remains active as a easy-to-read reference implementation for most of the kernels that constitute the heart of todays ML landscape.


##### Evaluation usage

```python
import tfdeploy as td
import numpy as np

model = td.Model("/path/to/model.pkl")
inp, outp = model.get("input", "output")

batch = np.random.rand(10000, 784)
result = outp.eval({inp: batch})
```


##### Installation and dependencies

Via [pip](https://pypi.python.org/pypi/tfdeploy)

```bash
pip install tfdeploy
```

or by simply copying the file into your project.

NumPy ≥ 1.10 should be installed on your system. [SciPy](http://www.scipy.org/) is optional. See [optimization](#optimization) for more info on optional packages.

By design, TensorFlow is required when creating a model.


### Content

- [Why?](#why)
- [How?](#how)
    - [Convert your graph](#convert-your-graph)
    - [Load the model and evaluate](#load-the-model-and-evaluate)
    - [Write your own operation](#write-your-own-operation)
- [Ensembles](#ensembles)
- [Optimization](#optimization)
- [Performance](#performance)
- [Contributing](#contributing)
- [Development](#development)
- [Authors](#authors)
- [License](#license)

## Why?

Working with TensorFlow is awesome. Model definition and training is simple yet powerful, and the range of built-in features is just striking.

Model deployment in environments that are not able to run TensorFlow, however, things can be difficult (**note** that tfdeploy was developed before TensorFlow Lite was a thing).

To boil it down, tfdeploy

- is lightweight. A single file with < 150 lines of core code. Just copy it to your project.
- [faster](#performance) than using TensorFlow's `Tensor.eval`.
- **does not need TensorFlow** during evaluation.
- only depends on NumPy.
- can load one or more models from a single file.
- does not support GPUs (maybe [gnumpy](http://www.cs.toronto.edu/~tijmen/gnumpy.html) is worth a try here).


## How?

The central class is `tfdeploy.Model`. The following two examples demonstrate how a model can be created from a TensorFlow graph, saved to and loaded from disk, and eventually evaluated.

##### Convert your graph

```python
import tensorflow as tf
import tfdeploy as td

# setup tfdeploy (only when creating models)
td.setup(tf)

# build your graph
sess = tf.Session()

# use names for input and output layers
x = tf.placeholder("float", shape=[None, 784], name="input")
W = tf.Variable(tf.truncated_normal([784, 100], stddev=0.05))
b = tf.Variable(tf.zeros([100]))
y = tf.nn.softmax(tf.matmul(x, W) + b, name="output")

sess.run(tf.global_variables_initializer())

# ... training ...

# create a tfdeploy model and save it to disk
model = td.Model()
model.add(y, sess) # y and all its ops and related tensors are added recursively
model.save("model.pkl")
```

##### Load the model and evaluate

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

##### Write your own `Operation`

tfdeploy supports most of the `Operation`'s [implemented in tensorflow](https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html). However, if you miss one (in that case, submit a PR or an issue ;) ) or if you're using custom ops, you might want to extend tfdeploy by defining a new class op that inherits from `tfdeploy.Operation`:

```python
import tensorflow as tf
import tfdeploy as td
import numpy as np

# setup tfdeploy (only when creating models)
td.setup(tf)

# ... write you model here ...

# let's assume your final tensor "y" relies on an op of type "InvertedSoftmax"
# before creating the td.Model, you should add that op to tfdeploy

class InvertedSoftmax(td.Operation):
    @staticmethod
    def func(a):
        e = np.exp(-a)
        # ops should return a tuple
        return np.divide(e, np.sum(e, axis=-1, keepdims=True)),

# this is equivalent to
# @td.Operation.factory
# def InvertedSoftmax(a):
#     e = np.exp(-a)
#     return np.divide(e, np.sum(e, axis=-1, keepdims=True)),

# now we're good to go
model = td.Model()
model.add(y, sess)
model.save("model.pkl")
```

When writing new ops, three things are important:

- Try to avoid loops, prefer NumPy vectorization.
- Return a tuple.
- Don't change incoming tensors/arrays in-place, always work on and return copies.


## Ensembles

tfdeploy provides a helper class to evaluate an ensemble of models: `Ensemble`. It can load multiple models, evaluate them and combine their output values using different methods.

```python
# create the ensemble
ensemble = td.Ensemble(["model1.pkl", "model2.pkl", ...], method=td.METHOD_MEAN)

# get input and output tensors (which actually are TensorEnsemble instances)
input, output = ensemble.get("input", "output")

# evaluate the ensemble just like a normal model
batch = ...
value = output.eval({input: batch})
```

The return value of `get()` is a `TensorEnsemble` istance. It is basically a wrapper around multiple tensors and should be used as keys in the `feed_dict` of the `eval()` call.

You can choose between `METHOD_MEAN` (the default), `METHOD_MAX` and `METHOD_MIN`. If you want to use a custom ensembling method, use `METHOD_CUSTOM` and overwrite the static `func_custom()` method of the `TensorEnsemble` instance.


## Optimization

Most ops are written using pure numpy. However, multiple implementations of the same op are allowed that may use additional third-party Python packages providing even faster functionality for some situations.

For example, NumPy does not provide a vectorized *lgamma* function. Thus, the standard `tfdeploy.Lgamma` op uses `math.lgamma` that was previously vectorized using `numpy.vectorize`. For these situations, additional implementations of the same op are possible (the *lgamma* example is quite academic, but this definitely makes sense for more sophisticated ops like pooling). We can simply tell the op to use its SciPy implementation instead:

```python
td.Lgamma.use_impl(td.IMPL_SCIPY)
```

Currently, allowed implementation types are NumPy (`IMPL_NUMPY`, the default) and SciPy (`IMPL_SCIPY`).


##### Adding additional implementations

Additional implementations can be added by setting the `impl` attribute of the op factory or by using the `add_impl` decorator of existing operations. The first registered implementation will be the default one.

```python
# create the default lgamma op with numpy implementation
lgamma_vec = np.vectorize(math.lgamma)

@td.Operation.factory
# equivalent to
# @td.Operation.factory(impl=td.IMPL_NUMPY)
def Lgamma(a):
    return lgamma_vec(a),

# add a scipy-based implementation
@Lgamma.add_impl(td.IMPL_SCIPY)
def Lgamma(a):
    return sp.special.gammaln(a),
```


##### Auto-optimization

If SciPy is available on your system, it is reasonable to use all ops in their SciPy implementation (if it exists, of course). This should be configured before you create any model from TensorFlow objects using the second argument of the `setup` function:

```python
td.setup(tf, td.IMPL_SCIPY)
```

Ops that do not implement `IMPL_SCIPY` stick with the NumPy version (`IMPL_NUMPY`).


## Performance

tfdeploy is lightweight (1 file, < 150 lines of core code) and fast. Internal evaluation calls have only very few overhead and tensor operations use NumPy vectorization. The actual performance depends on the ops in your graph. While most of the TensorFlow ops have a numpy equivalent or can be constructed from NumPy functions, a few ops require additional Python-based loops (e.g. `BatchMatMul`). But in many cases (and for small to medium graphs) it's potentially faster than using TensorFlow's `Tensor.eval`.

This is a comparison for a basic graph where all ops are vectorized (basically `Add`, `MatMul` and `Softmax`):

```bash
> ipython -i tests/perf/simple.py

In [1]: %timeit -n 100 test_tf()
100 loops, best of 3: 109 ms per loop

In [2]: %timeit -n 100 test_td()
100 loops, best of 3: 60.5 ms per loop
```

## Contributing

If you want to contribute with new ops and features, I'm happy to receive pull requests. Just make sure to add a new test case to `tests/core.py` or `tests/ops.py` and run them via:

```bash
> python -m unittest tests
```


##### Test grid

In general, tests should be run for different environments:

|     Variation      |  Values |
| ------------------ | ------- |
| tensorflow version | `1.0.1` |
| python version     | 2, 3    |
| `TD_TEST_SCIPY`    | 0, 1    |
| `TD_TEST_GPU`      | 0, 1    |


##### Docker

For testing purposes, it is convenient to use docker. Fortunately, the official [tensorflow images](https://hub.docker.com/r/tensorflow/tensorflow/) contain all we need:

```bash
git clone https://github.com/riga/tfdeploy.git
cd tfdeploy

docker run --rm -v `pwd`:/root/tfdeploy -w /root/tfdeploy -e "TD_TEST_SCIPY=1" tensorflow/tensorflow:1.0.1 python -m unittest tests
```


## Development

- Source hosted at [GitHub](https://github.com/riga/tfdeploy)
- Report issues, questions, feature requests on [GitHub Issues](https://github.com/riga/tfdeploy/issues)


## Authors

- [Marcel R.](https://github.com/riga)
