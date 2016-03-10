# -*- coding: utf-8 -*-

"""
Deploy tensorflow graphs for insanely-fast evaluation and export to tensorflow-less
environments running numpy.
"""


__author__     = "Marcel Rieger"
__copyright__  = "Copyright 2016, Marcel Rieger"
__credits__    = ["Marcel Rieger", "Benjamin Fischer"]
__contact__    = "https://github.com/riga/tfdeploy"
__license__    = "MIT"
__status__     = "Development"
__version__    = "0.1.6"

__all__ = ["Model", "Operation", "UnknownOperationException", "OperationMismatchException"]


import os
import re
import cPickle
from uuid import uuid4
from functools import reduce
from itertools import product
import numpy as np


_locals = locals()


class Model(object):
    """
    A trained model that contains one or more converted tensorflow graphs. Usage:

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

    .. py:attribute:: root
       type: set

       The set of all contained root tensors.
    """

    value_index_cre = re.compile("\:\d+$")
    default_value_index = 0

    def __init__(self, path=None):
        super(Model, self).__init__()

        self.root = set()

        # load when desired
        if path is not None:
            self.load(path)

    def get(self, *names):
        """
        Returns one or more tensors given by *names* using a deep lookup within the model. *None* is
        returned when no tensor was found. In case a tensor is passed, it's name is used for the
        lookup.
        """
        tensors = tuple(self._get(name) for name in names)
        return tensors[0] if len(names) == 1 else tensors

    def _get(self, name):
        if isinstance(name, Tensor):
            name = name.name

        # append the default value_index if there's none
        if not self.value_index_cre.search(name):
            name += ":%d" % self.default_value_index

        # return the first occurance of a tensor with that name
        return reduce(lambda t1,t2: t1 or t2.get(name), self.root, None)

    def __getitem__(self, name):
        return self.get(name)

    def __contains__(self, name):
        return self.get(name) is not None

    def add(self, tensor, sess=None):
        """
        Adds a new *tensor* to the root set. When *tensor* is not an instance of :py:class:`Tensor`
        but an instance of ``tensorflow.Tensor``, it is converted first. In that case, *sess* should
        be a valid tensorflow session.
        """
        if not isinstance(tensor, Tensor):
            tensor = Tensor(tensor, sess)

        self.root.add(tensor)

    def load(self, path):
        """
        Loads all tensors from a file defined by *path* and adds them to the root set.
        """
        path = os.path.expandvars(os.path.expanduser(path))
        with open(path, "r") as f:
            tensors = cPickle.load(f)
        for t in tensors:
            self.add(t)

    def save(self, path):
        """
        Saves all tensors of the root set to a file defined by *path*.
        """
        path = os.path.expandvars(os.path.expanduser(path))
        with open(path, "w") as f:
            cPickle.dump(self.root, f)


class TensorRegister(type):
    """
    Meta class of :py:class:`Tensor` that performs instance caching indexed by tensorflow tensor
    instances.
    """

    instances = {}

    def __call__(cls, tftensor, sess):
        # simply caching
        if tftensor not in cls.instances:
            cls.instances[tftensor] = super(TensorRegister, cls).__call__(tftensor, sess)
        return cls.instances[tftensor]


class Tensor(object):
    """
    Building block of a model. In *graph* terms, tensors represent connections between nodes (ops)
    of a graph. It contains information on the op it results from.

    .. py:attribute:: name
       type: string

       The name of the tensor.

    .. py:attribute:: op
       type: None, Operation

       The op instance that defines the value of this tensor. When created from a
       ``tensorflow.Placeholder`` or a ``tensorflow.Variable``, op will be *None*.

    .. py:attribute:: value
       type: None, numpy.ndarray

       The value of this tensor. When created from a ``tensorflow.Variable``, this will be the value
       of that variable, or *None* otherwise until it is evaluated the first time.
    """

    __metaclass__ = TensorRegister

    def __init__(self, tftensor, sess):
        super(Tensor, self).__init__()

        if not sess:
            raise ValueError("bad tensorflow session: %s" % sess)

        self.name = tftensor.name
        self.op = None
        self.value = None
        self.last_uuid = None

        # no op for variables, placeholders and constants
        # explicit value for variables and constants
        if tftensor.op.type in ("Variable", "Const"):
            self.value = tftensor.eval(session=sess)
        elif tftensor.op.type != "Placeholder":
            self.op = Operation.new(tftensor.op, sess)

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
            self.value = self.op.eval(feed_dict, _uuid)

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
        classdict.setdefault("type", classname)
        cls = super(OperationRegister, metacls).__new__(metacls, classname, bases, classdict)
        # register the class
        metacls.classes[cls.type] = cls
        return cls

    def __call__(cls, tfoperation, sess):
        # simply caching
        if tfoperation not in cls.instances:
            cls.instances[tfoperation] = super(OperationRegister, cls).__call__(tfoperation, sess)
        return cls.instances[tfoperation]


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


class Operation(object):
    """
    Building block of a model. In *graph* terms, operations (ops) represent nodes that are connected
    via tensors. It contains information on its input tensors.

    .. py:attribute:: inputs
       type: tuple

       Tensors that are input to this op. Their order is important as they are forwarded to *func*
       for evaluation.

    .. py:attribute:: type
       type: string

       The type if the op which should be the same as the original tensorflow op type.

    .. py:attribute:: unpack
       type: bool

       If *True* (default), the values of evaluated input tensors are forwarded to *func* as single
       arguments, or, otherwise, as a list.

    .. py:attribute:: attrs
       type: tuple

       Names of the configuration attributes of the original tensorflow op.

    .. py:attribute:: kwargs
       type: list

       Keyword arguments containing configuration values that will be passed to *func*.
    """

    __metaclass__ = OperationRegister

    type = None
    unpack = True
    attrs = None

    def __init__(self, tfoperation, sess):
        super(Operation, self).__init__()

        # compare types as a cross check
        if self.type != tfoperation.type:
            raise OperationMismatchException("operation types do not match: %s, %s" \
                % (self.type, tfoperation.type))

        self.inputs = tuple(Tensor(tftensor, sess) for tftensor in tfoperation.inputs)

        # store attributes as kwargs for calls to eval
        self.kwargs = [tfoperation.get_attr(attr) for attr in (self.attrs or [])]

    @classmethod
    def new(cls, tfoperation, sess):
        """
        Factory function that takes a tensorflow session *sess* and a tensorflow op *tfoperation*
        and returns an instance of the appropriate op class. Raises an exception of type
        :py:exc:`UnknownOperationException` in case the requested op type is not known.
        """
        if tfoperation.type not in cls.classes:
            raise UnknownOperationException("unknown operation: %s" % tfoperation.type)

        return cls.classes[tfoperation.type](tfoperation, sess)

    def get(self, *names):
        """
        Returns one or more tensors given by *names* using a deep lookup within this op. *None* is
        returned when no tensor was found.
        """
        tensors = tuple(self._get(name) for name in names)
        return tensors[0] if len(names) == 1 else tensors

    def _get(self, name):
        return reduce(lambda t1,t2: t1 or t2.get(name), self.inputs, None)

    def eval(self, feed_dict, _uuid):
        """ eval(feed_dict=None)
        Returns the value of the output tensor. See :py:meth:`Tensor.eval` for more info.
        """
        args = [t.eval(feed_dict=feed_dict, _uuid=_uuid) for t in self.inputs]
        if self.unpack:
            args.extend(self.kwargs)
        else:
            args = [args] + self.kwargs
        return self.func(*args)

    @staticmethod
    def func():
        """ func(*args)
        The actual op logic. Must be implemented in inheriting classes. All input tensors are
        forwarded to this method for evaluation.
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
            Op = Operation.__metaclass__(name, (Operation,), classdict)
            Op.__doc__ = func.__doc__
            _locals[name] = Op
            return Op
        return wrapper if func is None else wrapper(func)


lgamma_vec = np.vectorize(np.math.lgamma)
erf_vec = np.vectorize(np.math.erf)
erfc_vec = np.vectorize(np.math.erfc)


@Operation.factory
def Identity(a):
    """
    Identity op.
    """
    return a


@Operation.factory
def Add(a, b):
    """
    Addition op.
    """
    return np.add(a, b)


@Operation.factory
def Sub(a, b):
    """
    Subtraction op.
    """
    return np.subtract(a, b)


@Operation.factory
def Mul(a, b):
    """
    Multiplication op.
    """
    return np.multiply(a, b)


@Operation.factory
def Div(a, b):
    """
    Division op.
    """
    return np.divide(a, b)


@Operation.factory
def Mod(a, b):
    """
    Modulo op.
    """
    return np.mod(a, b)


@Operation.factory
def Cross(a, b):
    """
    Cross product op.
    """
    return np.cross(a, b)


@Operation.factory(unpack=False)
def AddN(inputs):
    """
    Multi add op.
    """
    return reduce(np.add, inputs)


@Operation.factory
def Abs(a):
    """
    Abs op.
    """
    return np.abs(a)


@Operation.factory
def Neg(a):
    """
    Neg op.
    """
    return np.negative(a)


@Operation.factory
def Sign(a):
    """
    Sign op.
    """
    return np.sign(a)


@Operation.factory
def Inv(a):
    """
    Reciprocal op.
    """
    return np.reciprocal(a)


@Operation.factory
def Square(a):
    """
    Square op.
    """
    return np.square(a)


@Operation.factory
def Round(a):
    """
    Round op.
    """
    return np.round(a)


@Operation.factory
def Sqrt(a):
    """
    Square root op.
    """
    return np.sqrt(a)


@Operation.factory
def Rsqrt(a):
    """
    Reciprocal square root op.
    """
    return np.reciprocal(np.sqrt(a))


@Operation.factory
def Pow(a, b):
    """
    Power op.
    """
    return np.power(a, b)


@Operation.factory
def Exp(a):
    """
    Exponential op.
    """
    return np.exp(a)


@Operation.factory
def Log(a):
    """
    Logarithm op.
    """
    return np.log(a)


@Operation.factory
def Ceil(a):
    """
    Ceil round op.
    """
    return np.ceil(a)


@Operation.factory
def Floor(a):
    """
    Floor round op.
    """
    return np.floor(a)


@Operation.factory
def Maximum(a, b):
    """
    Maximum op.
    """
    return np.maximum(a, b)


@Operation.factory
def Minimum(a, b):
    """
    Minimum op.
    """
    return np.minimum(a, b)


@Operation.factory
def Cos(a):
    """
    Cos op.
    """
    return np.cos(a)


@Operation.factory
def Sin(a):
    """
    Sin op.
    """
    return np.sin(a)


@Operation.factory
def Lgamma(a):
    """
    lgamma op.
    """
    return lgamma_vec(a)


@Operation.factory
def Erf(a):
    """
    Gaussian error function op.
    """
    return erf_vec(a)


@Operation.factory
def Erfc(a):
    """
    Complementary gaussian error function op.
    """
    return erfc_vec(a)


@Operation.factory
def Diag(a):
    """
    Diag op.
    """
    r = np.zeros(2 * a.shape)
    for idx, v in np.ndenumerate(a):
        r[2 * idx] = v
    return r


@Operation.factory
def Transpose(a, perm=None):
    """
    Transpose op.
    """
    return np.transpose(a, axes=perm)


@Operation.factory(attrs=("transpose_a", "transpose_b"))
def MatMul(a, b, transpose_a, transpose_b):
    """
    Matrix multiplication op.
    """
    return np.dot(a if not transpose_a else np.transpose(a),
                  b if not transpose_b else np.transpose(b))


@Operation.factory(attrs=("adj_x", "adj_y"))
def BatchMatMul(a, b, adj_a, adj_b):
    """
    Batched matrix multiplication op.
    """
    # apply adjoint op if required along last two axes
    axes = range(len(a.shape))
    axes.append(axes.pop(-2))
    if adj_a:
        a = np.conj(np.transpose(a, axes=axes))
    if adj_b:
        b = np.conj(np.transpose(b, axes=axes))
    # create the target tensor
    r = np.empty(a.shape[:-2] + (a.shape[-2], b.shape[-1]))
    # no batched dot op in np so loop over all indexes except last two dims
    for idx in product(*(xrange(dim) for dim in a.shape[:-2])):
        r[idx] = np.dot(a[idx], b[idx])
    return r


@Operation.factory
def MatrixDeterminant(a):
    """
    Matrix det op.
    """
    return np.linalg.det(a)


@Operation.factory
def BatchMatrixDeterminant(a):
    """
    Batched matrix det op.
    """
    return np.linalg.det(a)


@Operation.factory
def MatrixInverse(a):
    """
    Matrix inversion op.
    """
    return np.linalg.inv(a)


@Operation.factory
def BatchMatrixInverse(a):
    """
    Batched matrix inversion op.
    """
    return np.linalg.inv(a)


@Operation.factory
def Cholesky(a):
    """
    Cholesky decomposition op.
    """
    return np.linalg.cholesky(a)


@Operation.factory
def BatchCholesky(a):
    """
    Batched Cholesky decomposition op.
    """
    return np.linalg.cholesky(a)


@Operation.factory
def SelfAdjointEig(a):
    """
    Eigen decomp op.
    """
    shape = list(a.shape)
    shape[-2] += 1
    return np.append(*np.linalg.eig(a)).reshape(*shape)


@Operation.factory
def BatchSelfAdjointEig(a):
    """
    Batched eigen decomp op.
    """
    shape = list(a.shape)
    shape[-2] += 1
    return np.append(*np.linalg.eig(a)).reshape(*shape)


@Operation.factory
def MatrixSolve(a, b):
    """
    Matrix solve op.
    """
    return np.linalg.solve(a, b)


@Operation.factory
def BatchMatrixSolve(a, b):
    """
    Batched matrix solve op.
    """
    return np.linalg.solve(a, b)


@Operation.factory
def MatrixSolveLs(a, b, l2_regularizer):
    """
    Matrix least-squares solve op.
    """
    return np.linalg.lstsq(a, b)[0]


@Operation.factory
def Softmax(a):
    """
    Softmax op.
    """
    e = np.exp(a)
    return np.divide(e, np.sum(e, axis=-1, keepdims=True))


@Operation.factory
def Rank(a):
    """
    Rank op.
    """
    return len(a.shape)


@Operation.factory
def Range(start, limit, delta):
    """
    Range op.
    """
    return np.arange(start, limit, delta)
