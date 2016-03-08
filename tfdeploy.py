# -*- coding: utf-8 -*-

"""
Deploy tensorflow graphs for insanely-fast model evaluation and export to tensorflow-less
environments running numpy.
"""


__author__     = "Marcel Rieger"
__copyright__  = "Copyright 2016, Marcel Rieger"
__credits__    = ["Marcel Rieger", "Benjamin Fischer"]
__license__    = "MIT"
__status__     = "Development"
__version__    = "0.1.5"

__all__ = ["Model", "Operation", "UnknownOperationException", "OperationMismatchException"]


import os
import re
import cPickle
from uuid import uuid4
import numpy as np


_locals = locals()


class Model(object):
    """
    TODO.
    """

    value_index_cre = re.compile("\:\d+$")
    default_value_index = 0

    def __init__(self, path=None):
        super(Model, self).__init__()

        self.root = set()

        # load when desired
        if path is not None:
            self.load(path)

    def get(self, name):
        """
        Returns a tensor given by *name* using a deep lookup within the model. *None* is returned
        when no tensor was found. In case a tensor is passed, it's name is used for the lookup.
        """
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
            tensor = Tensor(sess, tensor)

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

    def __call__(cls, sess, tftensor):
        # simply caching
        if tftensor not in cls.instances:
            cls.instances[tftensor] = super(TensorRegister, cls).__call__(sess, tftensor)
        return cls.instances[tftensor]


class Tensor(object):
    """
    TODO.
    """

    __metaclass__ = TensorRegister

    def __init__(self, sess, tftensor):
        super(Tensor, self).__init__()

        if not sess:
            raise ValueError("bad tensorflow session: %s" % sess)

        self.name = tftensor.name
        self.op = None
        self.value = None
        self.last_uuid = None

        # no op for variables and placeholders
        # explicit value for variables
        if tftensor.op.type == "Variable":
            self.value = tftensor.eval(session=sess)
        elif tftensor.op.type != "Placeholder":
            self.op = Operation.new(sess, tftensor.op)

    def get(self, name):
        """
        Returns a tensor given by *name* using a deep lookup within the inputs of the op. Note that
        *this* tensor is returned when *name* matches. *None* is returned when no tensor was found.
        """
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

    def __call__(cls, sess, tfoperation):
        # simply caching
        if tfoperation not in cls.instances:
            cls.instances[tfoperation] = super(OperationRegister, cls).__call__(sess, tfoperation)
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
    TODO.
    """

    __metaclass__ = OperationRegister

    type = None

    def __init__(self, sess, tfoperation):
        super(Operation, self).__init__()

        # compare types as a cross check
        if self.type != tfoperation.type:
            raise OperationMismatchException("operation types do not match: %s, %s" \
                % (self.type, tfoperation.type))

        self.inputs = tuple(Tensor(sess, tftensor) for tftensor in tfoperation.inputs)

    @classmethod
    def new(cls, sess, tfoperation):
        """
        Factory function that takes a tensorflow session *sess* and a tensorflow op *tfoperation*
        and returns an instance of the appropriate op class. Raises an exception of type
        :py:exc:`UnknownOperationException` in case the requested op type is not known.
        """
        if tfoperation.type not in cls.classes:
            raise UnknownOperationException("unknown operation: %s" % tfoperation.type)

        return cls.classes[tfoperation.type](sess, tfoperation)

    def get(self, name):
        """
        Returns a tensor given by *name* using a deep lookup within this op. *None* is returned when
        no tensor was found.
        """
        return reduce(lambda t1,t2: t1 or t2.get(name), self.inputs, None)

    def eval(self, feed_dict, _uuid):
        """ eval(feed_dict=None)
        Returns the value of the output tensor. See :py:meth:`Tensor.eval` for more info.
        """
        return self.func(*(t.eval(feed_dict=feed_dict, _uuid=_uuid) for t in self.inputs))

    @staticmethod
    def func():
        raise NotImplementedError

    @staticmethod
    def factory(func):
        """
        Returns a new op class whose static function will be set to *func*. The name of *func* will
        also be the op class name.
        """
        name = func.__name__
        classdict = {"func": staticmethod(func)}
        Op = Operation.__metaclass__(name, (Operation,), classdict)
        _locals[name] = Op
        return Op


@Operation.factory
def Identity(a):
    return a


@Operation.factory
def Add(a, b):
    return np.add(a, b)


@Operation.factory
def Sub(a, b):
    return np.subtract(a, b)


@Operation.factory
def Mul(a, b):
    return np.multiply(a, b)


@Operation.factory
def Div(a, b):
    return np.divide(a, b)


@Operation.factory
def MatMul(a, b):
    return np.dot(a, b)


@Operation.factory
def Round(a):
    return np.round(a)


@Operation.factory
def Floor(a):
    return np.floor(a)


@Operation.factory
def Ceil(a):
    return np.ceil(a)


@Operation.factory
def Softmax(a):
    e = np.exp(a)
    return np.divide(e, np.sum(e, axis=-1, keepdims=True))
