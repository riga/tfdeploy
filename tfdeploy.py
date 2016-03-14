# -*- coding: utf-8 -*-

"""
Deploy tensorflow graphs for faster evaluation and export to tensorflow-less environments running
numpy.
"""


__author__     = "Marcel Rieger"
__copyright__  = "Copyright 2016, Marcel Rieger"
__credits__    = ["Marcel Rieger", "Benjamin Fischer"]
__contact__    = "https://github.com/riga/tfdeploy"
__license__    = "MIT"
__status__     = "Development"
__version__    = "0.1.8"

__all__ = ["Model", "Tensor", "Operation", "UnknownOperationException",
           "OperationMismatchException"]


import os
import re
from uuid import uuid4
from functools import reduce
from operator import mul
from itertools import product
from collections import defaultdict

try:
    # python 2
    import cPickle as pickle
except ImportError:
    # python 3
    import pickle

import numpy as np


_locals = locals()


# metaclass decorator from six package, credits to Benjamin Peterson
def add_metaclass(metaclass):
    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get("__slots__")
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop("__dict__", None)
        orig_vars.pop("__weakref__", None)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)
    return wrapper


class Model(object):
    """
    A trained model that contains one or more converted tensorflow graphs. When *path* is set, a
    previously saved model is loaded from that path. Usage:

    .. code-block:: python

       import tensorflow as tf
       import tfdeploy as td

       # build your graph, use names for input and output tensors
       sess = tf.Session()
       x = tf.placeholder("float", shape=[None, 784], name="input")
       W = tf.Variable(tf.truncated_normal([784, 100], stddev=0.05))
       b = tf.Variable(tf.zeros([100]))
       y = tf.nn.softmax(tf.matmul(x, W) + b, name="output")
       sess.run(tf.initialize_all_variables())

       # ... training ...

       # create a model and save it to disk
       model = Model()
       model.add(y, sess)
       model.save("model.pkl")

    .. py:attribute:: roots
       type: dict

       Contained root tensors mapped to a key.
    """

    value_index_cre = re.compile("\:\d+$")
    default_value_index = 0

    def __init__(self, path=None):
        super(Model, self).__init__()

        self.roots = {}

        # load when desired
        if path is not None:
            self.load(path)

    def get(self, *names, **kwargs):
        """ get(*names, key=None)
        Returns one or more tensors given by *names* using a deep lookup within the model. If *key*
        is not *None*, Only the root tensor with that *key* is traversed. *None* is returned when no
        tensor was found. In case a tensor is passed, it's name is used for the lookup.
        """
        tensors = tuple(self._get(name, **kwargs) for name in names)
        return tensors[0] if len(names) == 1 else tensors

    def _get(self, name, key=None):
        if isinstance(name, Tensor):
            name = name.name

        # append the default value_index if there's none
        if not self.value_index_cre.search(name):
            name += ":%d" % self.default_value_index

        # return the first occurance of a tensor with that name
        if key is not None:
            return self.roots[key].get(name)
        else:
            return reduce(lambda t1, t2: t1 or t2.get(name), self.roots.values(), None)

    def __getitem__(self, name):
        return self.get(name)

    def __contains__(self, name):
        return self.get(name) is not None

    def add(self, tensor, tf_sess=None, key=None, **kwargs):
        """
        Adds a new root *tensor* for a *key* which, if *None*, defaults to a consecutive number.
        When *tensor* is not an instance of :py:class:`Tensor` but an instance of
        ``tensorflow.Tensor``, it is converted first. In that case, *tf_sess* should be a valid
        tensorflow session and *kwargs* are forwarded to the :py:class:`Tensor` constructor.
        """
        if not isinstance(tensor, Tensor):
            tensor = Tensor(tensor, tf_sess, **kwargs)

        if key is None:
            key = len(self.roots)
            while key in self.roots:
                key += 1

        self.roots[key] = tensor

    def load(self, path):
        """
        Loads all tensors from a file defined by *path* and adds them to the root set.
        """
        path = os.path.expandvars(os.path.expanduser(path))
        with open(path, "r") as f:
            roots = pickle.load(f)

        for key, tensor in roots.items():
            self.add(tensor, key=key)

    def save(self, path):
        """
        Saves all tensors of the root set to a file defined by *path*.
        """
        path = os.path.expandvars(os.path.expanduser(path))
        with open(path, "w") as f:
            pickle.dump(self.roots, f)


class TensorRegister(type):
    """
    Meta class of :py:class:`Tensor` that performs instance caching indexed by tensorflow tensor
    instances.
    """

    instances = {}

    def __call__(cls, tf_tensor, *args, **kwargs):
        # simply caching
        if tf_tensor not in cls.instances:
            inst = super(TensorRegister, cls).__call__(tf_tensor, *args, **kwargs)
            cls.instances[tf_tensor] = inst
        return cls.instances[tf_tensor]


@add_metaclass(TensorRegister)
class Tensor(object):
    """
    Building block of a model. In *graph* terms, tensors represent connections between nodes (ops)
    of a graph. It contains information on the op it results from. The conversion uses the
    (tensorflow) instances *tf_tensor* and *tf_sess*, *tf_feed_dict* can be set to evaluate the
    tensor's current value.

    .. py:attribute:: name
       type: string

       The name of the tensor.

    .. py:attribute:: value_index
       type: int

       The value index of this tensor, i.e., the position in the op's output list.

    .. py:attribute:: op
       type: None, Operation

       The op instance that defines the value of this tensor. When created from a
       ``tensorflow.Placeholder`` or a ``tensorflow.Variable``, op will be *None*.

    .. py:attribute:: value
       type: None, numpy.ndarray

       The value of this tensor. When created from a ``tensorflow.Variable``, this will be the value
       of that variable, or *None* otherwise until it is evaluated the first time.
    """

    def __init__(self, tf_tensor, tf_sess, tf_feed_dict=None):
        super(Tensor, self).__init__()

        if not tf_sess:
            raise ValueError("bad tensorflow session: %s" % tf_sess)

        self.name = tf_tensor.name
        self.value_index = tf_tensor.value_index
        self.op = None
        self.value = None
        self.last_uuid = None

        # guess the value
        # explicitly evaluate variables and constants, use feed_dict for placeholders
        if tf_tensor.op.type in ("Variable", "Const"):
            self.value = tf_tensor.eval(session=tf_sess, feed_dict=tf_feed_dict)
        elif tf_tensor.op.type == "Placeholder":
            if tf_feed_dict is not None and tf_tensor in tf_feed_dict:
                self.value = tf_feed_dict[tf_tensor]

        # create the op
        # no op for variables, placeholders and constants
        if tf_tensor.op.type not in ("Variable", "Const", "Placeholder"):
            self.op = Operation.new(tf_tensor.op, tf_sess, tf_feed_dict=tf_feed_dict)

    def get(self, *names):
        """
        Returns one or more tensors given by *names* using a deep lookup within the inputs of the
        op. Note that *this* tensor is returned when the name matches. *None* is returned when no
        tensor was found.
        """
        tensors = tuple(self._get(name) for name in names)
        return tensors[0] if len(names) == 1 else tensors

    def _get(self, name):
        if self.name == name:
            return self
        elif self.op is None:
            return None
        else:
            return self.op.get(name)

    def eval(self, feed_dict=None, _uuid=None):
        """ eval(feed_dict=None)
        Returns the value of this tensor based on the evaluation of all dependent ops and tensors.
        You can overwrite values of dependent tensors using *feed_dict*, a mapping of tensors to
        numpy arrays, which is passed down the evaluation chain.
        """
        # set a cache uuid for this eval call
        if _uuid is None:
            _uuid = uuid4()

        # already cached? this is important for tensors that are used multiple time within the graph
        if _uuid == self.last_uuid:
            return self.value
        else:
            self.last_uuid = _uuid

        if feed_dict is None:
            feed_dict = {}

        # when _this_ tensor is in the feed_dict, return the fed value
        # otherwise, eval the op
        if self in feed_dict:
            self.value = feed_dict[self]
        elif self.op is not None:
            self.value = self.op.eval(feed_dict=feed_dict, _uuid=_uuid)[self.value_index]

        return self.value

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)


class OperationRegister(type):
    """
    Meta class of :py:class:`Operation` that performs instance caching indexed by tensorflow op
    instances. Additionaly, all derived classes are registered in a mapping using their type's for
    faster op class lookup.
    """

    classes = {}
    instances = {}

    def __new__(metacls, classname, bases, classdict):
        # when not set explicitly in that class, set type to the class name
        classdict.setdefault("types", (classname,))
        cls = super(OperationRegister, metacls).__new__(metacls, classname, bases, classdict)
        # register the class for each of its types
        for type in cls.types:
            metacls.classes[type] = cls
        return cls

    def __call__(cls, tf_op, *args, **kwargs):
        # simply caching
        if tf_op not in cls.instances:
            inst = super(OperationRegister, cls).__call__(tf_op, *args, **kwargs)
            cls.instances[tf_op] = inst
        return cls.instances[tf_op]


class UnknownOperationException(Exception):
    """
    An exception which is raised when trying to convert an unknown tensorflow.
    """
    pass


class OperationMismatchException(Exception):
    """
    An exception which is raised during instantiation of an op whose type does not match the
    underlying tensorflow op.
    """
    pass


@add_metaclass(OperationRegister)
class Operation(object):
    """
    Building block of a model. In *graph* terms, operations (ops) represent nodes that are connected
    via tensors. It contains information on its input tensors. The conversion uses the
    (tensorflow) instance *tf_op*, all *args* and *kwargs* are forwarded to the :py:class:`Tensor`
    constructor for this op's input tensors.

    .. py:attribute:: types
       type: tuple
       classmember

       The types of tensorflow ops that this op can represent.

    .. py:attribute:: unpack
       type: bool
       classmember

       If *True* (default), the values of evaluated input tensors are forwarded to *func* as single
       arguments, or, otherwise, as a list.

    .. py:attribute:: attrs
       type: tuple
       classmember

       Names of the configuration attributes of the original tensorflow op.

    .. py:attribute:: name
       type: string

       The name of the op.

    .. py:attribute:: inputs
       type: tuple

       Tensors that are input to this op. Their order is important as they are forwarded to *func*
       for evaluation.

    .. py:attribute:: kwargs
       type: list

       Keyword arguments containing configuration values that will be passed to *func*.
    """

    types = ()
    unpack = True
    attrs = ()
    output_dtypes = False

    def __init__(self, tf_op, *args, **kwargs):
        super(Operation, self).__init__()

        # compare types as a cross check
        if tf_op.type not in self.types:
            raise OperationMismatchException("operation types do not match: %s, %s" \
                                             % (self.types, tf_op.type))

        self.name = tf_op.name
        self.inputs = tuple(Tensor(tf_tensor, *args, **kwargs) for tf_tensor in tf_op.inputs)

        self.value = None
        self.last_uuid = None

        # store attributes as kwargs for calls to eval
        self.kwargs = [tf_op.get_attr(attr) for attr in (self.attrs or [])]

        # store output dtypes for calls to eval when x is True
        self.output_dtypes = [dtype_map[dtype] for dtype in tf_op._output_types]

    @classmethod
    def new(cls, tf_op, *args, **kwargs):
        """
        Factory function that takes a tensorflow op *tf_op* and returns an instance of the
        appropriate op class. *args* and *kwargs* are forwarded to the op constructor. Raises an
        exception of type :py:exc:`UnknownOperationException` in case the requested op type is not
        known.
        """
        if tf_op.type not in cls.classes:
            raise UnknownOperationException("unknown operation: %s" % tf_op.type)

        return cls.classes[tf_op.type](tf_op, *args, **kwargs)

    def set_attr(self, attr, value):
        """
        Overwrites the value of an attribute *attr* with a new *value*.
        """
        if attr not in self.attrs:
            raise AttributeError("no attribute '%s' in op '%s'" % (attr, self.name))

        self.kwargs[self.attrs.index(attr)] = value

    def get(self, *names):
        """
        Returns one or more tensors given by *names* using a deep lookup within this op. *None* is
        returned when no tensor was found.
        """
        tensors = tuple(self._get(name) for name in names)
        return tensors[0] if len(names) == 1 else tensors

    def _get(self, name):
        return reduce(lambda t1,t2: t1 or t2.get(name), self.inputs, None)

    def eval(self, feed_dict=None, _uuid=None):
        """ eval(feed_dict=None)
        Returns the value of the output tensor. See :py:meth:`Tensor.eval` for more info.
        """
        # set a cache uuid for this eval call
        if _uuid is None:
            _uuid = uuid4()

        # already cached?
        if _uuid == self.last_uuid:
            return self.value
        else:
            self.last_uuid = _uuid

        args = [t.eval(feed_dict=feed_dict, _uuid=_uuid) for t in self.inputs]
        if self.unpack:
            args.extend(self.kwargs)
        else:
            args = [args] + self.kwargs
        if self.__class__.output_dtypes:
            args.append(self.output_dtypes)

        self.value = self.func(*args)

        return self.value

    @staticmethod
    def func():
        """ func(*args)
        The actual op logic. Must be implemented in inheriting classes. All input tensors are
        forwarded to this method for evaluation. Should return a tuple.
        """
        raise NotImplementedError

    @staticmethod
    def factory(func=None, **kwargs):
        """
        Returns a new op class whose static function will be set to *func*. The name of *func* will
        also be the op class name.
        """
        def wrapper(func):
            name = func.__name__
            classdict = {"func": staticmethod(func)}
            classdict.update(kwargs)
            Op = Operation.__class__(name, (Operation,), classdict)
            Op.__doc__ = func.__doc__
            _locals[name] = Op
            return Op
        return wrapper if func is None else wrapper(func)


dtype_map = {
    1: np.float32,
    2: np.float64,
    3: np.int32,
    4: np.uint8,
    5: np.int16,
    6: np.int8,
    7: np.object,
    8: np.complex64,
    9: np.int64,
    10: np.bool,
    14: np.uint16,
    17: np.uint16,
    101: np.float32,
    102: np.float64,
    103: np.int32,
    104: np.uint8,
    105: np.int16,
    106: np.int8,
    107: np.object,
    108: np.complex64,
    109: np.int64,
    110: np.bool,
    114: np.uint16,
    117: np.uint16
}


lgamma_vec = np.vectorize(np.math.lgamma)
erf_vec = np.vectorize(np.math.erf)
erfc_vec = np.vectorize(np.math.erfc)


@Operation.factory
def Identity(a):
    """
    Identity op.
    """
    return np.copy(a),


@Operation.factory(types=("Cast", "StringToNumber"), output_dtypes=True)
def Cast(a, output_dtypes):
    """
    Cast op.
    """
    return np.copy(a).astype(output_dtypes[0]),


@Operation.factory
def Shape(a):
    """
    Shape op.
    """
    return np.array(a.shape, dtype=np.int32),


@Operation.factory
def Size(a):
    """
    Size op.
    """
    return np.array([a.size], dtype=np.int32),


@Operation.factory
def Rank(a):
    """
    Rank op.
    """
    return np.array([len(a.shape)], dtype=np.int32),


@Operation.factory
def Reshape(a, shape):
    """
    Reshape op.
    """
    return np.copy(a).reshape(shape),


@Operation.factory(attrs=("squeeze_dims",))
def Squeeze(a, squeeze_dims):
    """
    Squeeze op, i.e. removes singular axes.
    """
    if not squeeze_dims:
        squeeze_dims = list(range(len(a.shape)))
    slices = [(0 if (dim == 1 and i in squeeze_dims) else slice(None)) \
              for i, dim in enumerate(a.shape)]
    return np.copy(a)[slices],


@Operation.factory
def ExpandDims(a, dim):
    """
    Expand dim op, i.e. add singular axis at dim.
    """
    shape = list(a.shape)
    if dim >= 0:
        shape.insert(dim, 1)
    else:
        shape.insert(len(shape) + dim + 1, 1)
    return np.copy(a).reshape(*shape),


@Operation.factory
def Slice(a, begin, size):
    """
    Slicing op.
    """
    return np.copy(a)[[slice(*tpl) for tpl in zip(begin, begin+size)]],


@Operation.factory(attrs=("num_split",))
def Split(dim, a, n):
    """
    Split op.
    """
    return tuple(np.split(np.copy(a), n, axis=dim))


@Operation.factory
def Tile(a, n):
    """
    Tile op.
    """
    return np.tile(a, n),


@Operation.factory
def Pad(a, paddings):
    """
    Zero padping op.
    """
    return np.pad(a, paddings, mode="constant", constant_values=0),


@Operation.factory
def Concat(dim, *inputs):
    """
    Concat op.
    """
    return np.concatenate(inputs, axis=dim),


@Operation.factory
def Pack(*inputs):
    """
    Pack op.
    """
    return np.asarray(inputs),


@Operation.factory
def Unpack(a):
    """
    Unpack op.
    """
    return tuple(a)


@Operation.factory(attrs=("seq_dim", "batch_dim"))
def ReverseSequence(a, seq_lengths, seq_dim, batch_dim):
    """
    Sequential reverse op.
    """
    r = np.copy(a)
    invidxs = (len(r.shape) - 1) * [slice(None)]
    if seq_dim < batch_dim:
        invidxs[seq_dim] = slice(None, None, -1)
    else:
        invidxs[seq_dim - 1] = slice(None, None, -1)
    _invidxs = tuple(invidxs)
    selidxs = len(r.shape) * [slice(None)]
    for i, l in enumerate(seq_lengths):
        if not l:
            continue
        selidxs[batch_dim] = i
        selidxs[seq_dim] = slice(0, l)
        _selidxs = tuple(selidxs)
        r[_selidxs] = a[_selidxs][_invidxs]
    return r,


@Operation.factory
def Reverse(a, dims):
    """
    Reverse op.
    """
    idxs = tuple(slice(None, None, -1 if dim else None) for dim in dims)
    return np.copy(a[idxs]),


@Operation.factory
def Transpose(a, perm=None):
    """
    Transpose op.
    """
    return np.transpose(a, axes=perm),


@Operation.factory(types=("Add", "BiasAdd"))
def Add(a, b):
    """
    Addition op.
    """
    return np.add(a, b),


@Operation.factory
def Sub(a, b):
    """
    Subtraction op.
    """
    return np.subtract(a, b),


@Operation.factory
def Mul(a, b):
    """
    Multiplication op.
    """
    return np.multiply(a, b),


@Operation.factory
def Div(a, b):
    """
    Division op.
    """
    return np.divide(a, b),


@Operation.factory
def Mod(a, b):
    """
    Modulo op.
    """
    return np.mod(a, b),


@Operation.factory
def Cross(a, b):
    """
    Cross product op.
    """
    return np.cross(a, b),


@Operation.factory(unpack=False)
def AddN(inputs):
    """
    Multi add op.
    """
    return reduce(np.add, inputs),


@Operation.factory
def Abs(a):
    """
    Abs op.
    """
    return np.abs(a),


@Operation.factory
def Neg(a):
    """
    Neg op.
    """
    return np.negative(a),


@Operation.factory
def Sign(a):
    """
    Sign op.
    """
    return np.sign(a),


@Operation.factory
def Inv(a):
    """
    Reciprocal op.
    """
    return np.reciprocal(a),


@Operation.factory
def Square(a):
    """
    Square op.
    """
    return np.square(a),


@Operation.factory
def Round(a):
    """
    Round op.
    """
    return np.round(a),


@Operation.factory
def Sqrt(a):
    """
    Square root op.
    """
    return np.sqrt(a),


@Operation.factory
def Rsqrt(a):
    """
    Reciprocal square root op.
    """
    return np.reciprocal(np.sqrt(a)),


@Operation.factory
def Pow(a, b):
    """
    Power op.
    """
    return np.power(a, b),


@Operation.factory
def Exp(a):
    """
    Exponential op.
    """
    return np.exp(a),


@Operation.factory
def Log(a):
    """
    Logarithm op.
    """
    return np.log(a),


@Operation.factory
def Ceil(a):
    """
    Ceil round op.
    """
    return np.ceil(a),


@Operation.factory
def Floor(a):
    """
    Floor round op.
    """
    return np.floor(a),


@Operation.factory
def Maximum(a, b):
    """
    Maximum op.
    """
    return np.maximum(a, b),


@Operation.factory
def Minimum(a, b):
    """
    Minimum op.
    """
    return np.minimum(a, b),


@Operation.factory
def Cos(a):
    """
    Cos op.
    """
    return np.cos(a),


@Operation.factory
def Sin(a):
    """
    Sin op.
    """
    return np.sin(a),


@Operation.factory
def Lgamma(a):
    """
    lgamma op.
    """
    return lgamma_vec(a),


@Operation.factory
def Erf(a):
    """
    Gaussian error function op.
    """
    return erf_vec(a),


@Operation.factory
def Erfc(a):
    """
    Complementary gaussian error function op.
    """
    return erfc_vec(a),


@Operation.factory
def Diag(a):
    """
    Diag op.
    """
    r = np.zeros(2 * a.shape, dtype=a.dtype)
    for idx, v in np.ndenumerate(a):
        r[2 * idx] = v
    return r,


@Operation.factory(attrs=("transpose_a", "transpose_b"))
def MatMul(a, b, transpose_a, transpose_b):
    """
    Matrix multiplication op.
    """
    return np.dot(a if not transpose_a else np.transpose(a),
                  b if not transpose_b else np.transpose(b)),


@Operation.factory(attrs=("adj_x", "adj_y"))
def BatchMatMul(a, b, adj_a, adj_b):
    """
    Batched matrix multiplication op.
    """
    # apply adjoint op if required along last two axes
    axes = list(range(len(a.shape)))
    axes.append(axes.pop(-2))
    if adj_a:
        a = np.conj(np.transpose(a, axes=axes))
    if adj_b:
        b = np.conj(np.transpose(b, axes=axes))
    # create the target tensor
    r = np.empty(a.shape[:-2] + (a.shape[-2], b.shape[-1]))
    # no batched dot op in np, so loop over all indexes except last two dims
    for idx in product(*(range(dim) for dim in a.shape[:-2])):
        r[idx] = np.dot(a[idx], b[idx])
    return r,


@Operation.factory(types=("MatrixDeterminant", "BatchMatrixDeterminant"))
def MatrixDeterminant(a):
    """
    Matrix det op.
    """
    return np.linalg.det(a),


@Operation.factory(types=("MatrixInverse", "BatchMatrixInverse"))
def MatrixInverse(a):
    """
    Matrix inversion op.
    """
    return np.linalg.inv(a),


@Operation.factory(types=("Cholesky", "BatchCholesky"))
def Cholesky(a):
    """
    Cholesky decomposition op.
    """
    return np.linalg.cholesky(a),


@Operation.factory(types=("SelfAdjointEig", "BatchSelfAdjointEig"))
def SelfAdjointEig(a):
    """
    Eigen decomp op.
    """
    shape = list(a.shape)
    shape[-2] += 1
    return np.append(*np.linalg.eig(a)).reshape(*shape),


@Operation.factory(types=("MatrixSolve", "BatchMatrixSolve"))
def MatrixSolve(a, b):
    """
    Matrix solve op.
    """
    return np.linalg.solve(a, b),


@Operation.factory
def MatrixSolveLs(a, b, l2_regularizer):
    """
    Matrix least-squares solve op.
    """
    return np.linalg.lstsq(a, b)[0],


@Operation.factory
def Complex(a, b):
    """
    Complex number op.
    """
    return np.add(a, np.multiply(b, 1j)),


@Operation.factory
def ComplexAbs(a):
    """
    Complex number length op.
    """
    return np.abs(a),


@Operation.factory
def Conj(a):
    """
    Complex conjugate op.
    """
    return np.conj(a),


@Operation.factory
def Imag(a):
    """
    Complex imag op.
    """
    return np.imag(a),


@Operation.factory
def Real(a):
    """
    Complex real op.
    """
    return np.real(a),


@Operation.factory
def FFT2D(a):
    """
    Discrete 2D FT op.
    """
    return np.fft.fft2(a),


@Operation.factory
def IFFT2D(a):
    """
    Discrete inverse 2D FT op.
    """
    return np.fft.ifft2(a),


@Operation.factory(attrs=("keep_dims",))
def Sum(a, reduction_indices, keep_dims):
    """
    Sum reduction op.
    """
    return np.sum(a, axis=tuple(reduction_indices), keepdims=keep_dims),


@Operation.factory(attrs=("keep_dims",))
def Prod(a, reduction_indices, keep_dims):
    """
    Prod reduction op.
    """
    return np.prod(a, axis=tuple(reduction_indices), keepdims=keep_dims),


@Operation.factory(attrs=("keep_dims",))
def Min(a, reduction_indices, keep_dims):
    """
    Min reduction op.
    """
    return np.amin(a, axis=tuple(reduction_indices), keepdims=keep_dims),


@Operation.factory(attrs=("keep_dims",))
def Max(a, reduction_indices, keep_dims):
    """
    Max reduction op.
    """
    return np.amax(a, axis=tuple(reduction_indices), keepdims=keep_dims),


@Operation.factory(attrs=("keep_dims",))
def Mean(a, reduction_indices, keep_dims):
    """
    Mean reduction op.
    """
    return np.mean(a, axis=tuple(reduction_indices), keepdims=keep_dims),


@Operation.factory(attrs=("keep_dims",))
def All(a, reduction_indices, keep_dims):
    """
    All reduction op.
    """
    return np.all(a, axis=tuple(reduction_indices), keepdims=keep_dims),


@Operation.factory(attrs=("keep_dims",))
def Any(a, reduction_indices, keep_dims):
    """
    Any reduction op.
    """
    return np.any(a, axis=tuple(reduction_indices), keepdims=keep_dims),


def seg_map(func, a, ids):
    m = defaultdict(list)
    for i, e in enumerate(ids):
        m[e].append(i)
    r = np.empty((len(m),) + a.shape[1:], dtype=a.dtype)
    for i, idxs in m.items():
        r[i] = func(idxs)
    return r


@Operation.factory(types=("SegmentSum", "UnsortedSegmentSum"))
def SegmentSum(a, ids, *args):
    """
    Segmented sum op.
    """
    func = lambda idxs: reduce(np.add, a[idxs])
    return seg_map(func, a, ids),


@Operation.factory
def SegmentProd(a, ids):
    """
    Segmented prod op.
    """
    func = lambda idxs: reduce(np.multiply, a[idxs])
    return seg_map(func, a, ids),


@Operation.factory
def SegmentMin(a, ids):
    """
    Segmented min op.
    """
    func = lambda idxs: np.amin(a[idxs], axis=0)
    return seg_map(func, a, ids),


@Operation.factory
def SegmentMax(a, ids):
    """
    Segmented max op.
    """
    func = lambda idxs: np.amax(a[idxs], axis=0)
    return seg_map(func, a, ids),


@Operation.factory
def SegmentMean(a, ids):
    """
    Segmented mean op.
    """
    func = lambda idxs: np.mean(a[idxs], axis=0)
    return seg_map(func, a, ids),


@Operation.factory
def SparseSegmentSum(a, idxs, ids):
    """
    Sparse segmented sum op.
    """
    return SegmentSum.func(a[idxs], ids)


@Operation.factory
def SparseSegmentMean(a, idxs, ids):
    """
    Sparse segmented mean op.
    """
    return SegmentMean.func(a[idxs], ids)


@Operation.factory
def SparseSegmentSqrtN(a, idxs, ids):
    """
    Sparse segmented sum / sqrt(n=len(idxs)) op.
    """
    func = lambda _idxs: np.divide(reduce(np.add, a[idxs][_idxs]), np.math.sqrt(len(_idxs)))
    return seg_map(func, a, ids),


@Operation.factory
def ArgMin(a, dim):
    """
    Argmin op.
    """
    return np.argmin(a, axis=dim),


@Operation.factory
def ArgMax(a, dim):
    """
    Argmax op.
    """
    return np.argmax(a, axis=dim),


@Operation.factory
def ListDiff(a, b):
    """
    List diff op.
    """
    d = np.setdiff1d(a, b)
    return d, np.searchsorted(a, d).astype(np.int32)


@Operation.factory
def Where(a):
    """
    Boolean where op.
    """
    return np.argwhere(a),


@Operation.factory
def Unique(a):
    """
    Unique op.
    """
    _, idxs, inv = np.unique(a, return_index=True, return_inverse=True)
    return np.copy(a)[np.sort(idxs)], idxs[inv].astype(np.int32)


@Operation.factory
def InvertPermutation(a):
    """
    Invert perm op.
    """
    return np.argsort(a).astype(np.int32),


@Operation.factory
def LinSpace(start, stop, num):
    """
    Linspace op.
    """
    return np.linspace(start, stop, num=num, dtype=np.float32),


@Operation.factory
def Range(start, limit, delta):
    """
    Range op.
    """
    return np.arange(start, limit, delta, dtype=np.int32),


@Operation.factory(attrs=("dtype", "seed"))
def RandomStandardNormal(shape, dtype, seed):
    """
    Standard (mu=0, sigma=1) gaussian op.
    """
    if seed:
        np.random.seed(seed)
    return np.random.normal(size=reduce(mul, shape)).reshape(shape).astype(dtype_map[dtype]),


@Operation.factory(attrs=("dtype", "seed"))
def TruncatedNormal(shape, dtype, seed):
    """
    Standard (mu=0, sigma=1) gaussian op with truncation above 2 sigma.
    """
    if seed:
        np.random.seed(seed)
    n = reduce(mul, shape)
    r = np.empty(n, dtype=dtype_map[dtype])
    idxs = np.ones(n, dtype=np.bool)
    while n:
        r[idxs] = np.random.normal(size=n)
        idxs = np.abs(r) > 2
        n = np.sum(idxs)
    return r.reshape(shape),


@Operation.factory(attrs=("dtype", "seed"))
def RandomUniform(shape, dtype, seed):
    """
    Random uniform op.
    """
    if seed:
        np.random.seed(seed)
    return np.random.uniform(size=shape).astype(dtype_map[dtype]),


@Operation.factory(attrs=("seed",))
def RandomUniformInt(shape, minval, maxval, seed):
    """
    Random uniform int op.
    """
    if seed:
        np.random.seed(seed)
    return np.random.randint(minval, maxval, size=shape),


@Operation.factory(attrs=("seed",))
def RandomShuffle(a, seed):
    """
    Random uniform op.
    """
    if seed:
        np.random.seed(seed)
    r = a.copy()
    np.random.shuffle(r)
    return r,


@Operation.factory
def Relu(a):
    """
    Relu op.
    """
    return np.maximum(a, 0),


@Operation.factory
def Relu6(a):
    """
    Relu6 op.
    """
    return np.clip(a, 0, 6),


@Operation.factory
def Elu(a):
    """
    Elu op.
    """
    return np.where(a < 0, np.subtract(np.exp(a), 1), a),


@Operation.factory
def Softplus(a):
    """
    Softplus op.
    """
    return np.log(np.add(np.exp(a), 1)),


@Operation.factory
def Softsign(a):
    """
    Softsign op.
    """
    return np.divide(a, np.add(np.abs(a), 1)),


@Operation.factory
def Sigmoid(a):
    """
    Sogmoid (logistic) op.
    """
    return np.reciprocal(np.add(1, np.exp(-a))),


@Operation.factory
def Tanh(a):
    """
    Tanh op.
    """
    return np.tanh(a),


@Operation.factory
def Softmax(a):
    """
    Softmax op.
    """
    e = np.exp(a)
    return np.divide(e, np.sum(e, axis=-1, keepdims=True)),
