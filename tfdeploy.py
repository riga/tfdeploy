# -*- coding: utf-8 -*-

"""
Deploy tensorflow graphs for insanely-fast model evaluation and export to tensorflow-less
environments via numpy.
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

    value_index_cre = re.compile("\:\d+$")
    default_value_index = 0

    def __init__(self, path=None):
        super(Model, self).__init__()

        self.root = set()

        if path is not None:
            self.load(path)

    def get(self, name):
        if isinstance(name, Tensor):
            name = name.name
        if not self.value_index_cre.search(name):
            name += ":%d" % self.default_value_index

        return reduce(lambda t1,t2: t1 or t2.get(name), self.root, None)

    def __getitem__(self, name):
        return self.get(name)

    def __contains__(self, name):
        return self.get(name) is not None

    def add(self, tensor, sess=None):
        if not isinstance(tensor, Tensor):
            tensor = Tensor(sess, tensor)

        self.root.add(tensor)

    def load(self, path):
        path = os.path.expandvars(os.path.expanduser(path))
        with open(path, "r") as f:
            tensors = cPickle.load(f)
        for t in tensors:
            self.add(t)

    def save(self, path):
        path = os.path.expandvars(os.path.expanduser(path))
        with open(path, "w") as f:
            cPickle.dump(self.root, f)


class TensorRegister(type):

    instances = {}

    def __call__(cls, sess, tftensor):
        if tftensor not in cls.instances:
            cls.instances[tftensor] = super(TensorRegister, cls).__call__(sess, tftensor)
        return cls.instances[tftensor]


class Tensor(object):

    __metaclass__ = TensorRegister

    def __init__(self, sess, tftensor):
        super(Tensor, self).__init__()

        if not sess:
            raise ValueError("bad tensorflow session: %s" % sess)

        self.name = None
        self.op = None
        self.value = None
        self.last_uuid = None

        self.name = tftensor.name

        if tftensor.op.type == "Variable":
            self.value = tftensor.eval(session=sess)
        elif tftensor.op.type != "Placeholder":
            self.op = Operation.new(sess, tftensor.op)

    def get(self, name):
        if self.name == name:
            return self
        elif self.op is None:
            return None
        else:
            return self.op.get(name)

    def eval(self, feed_dict=None, _uuid=None):
        if _uuid is None:
            _uuid = uuid4()

        if _uuid == self.last_uuid:
            return self.value
        else:
            self.last_uuid = _uuid

        if feed_dict is None:
            feed_dict = {}

        if self in feed_dict:
            self.value = feed_dict[self]
        elif self.op is not None:
            self.value = self.op.eval(feed_dict, _uuid)

        return self.value

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)


class OperationRegister(type):

    classes = {}
    instances = {}

    def __new__(metacls, classname, bases, classdict):
        classdict.setdefault("type", classname)
        cls = super(OperationRegister, metacls).__new__(metacls, classname, bases, classdict)
        metacls.classes[cls.type] = cls
        return cls

    def __call__(cls, sess, tfoperation):
        if tfoperation not in cls.instances:
            cls.instances[tfoperation] = super(OperationRegister, cls).__call__(sess, tfoperation)
        return cls.instances[tfoperation]


class UnknownOperationException(Exception):
    pass


class OperationMismatchException(Exception):
    pass


class Operation(object):

    __metaclass__ = OperationRegister

    type = None

    def __init__(self, sess, tfoperation):
        super(Operation, self).__init__()

        # check tfoperation type and our type
        if self.type != tfoperation.type:
            raise OperationMismatchException("operation types do not match: %s, %s" \
                % (self.type, tfoperation.type))

        self.inputs = tuple(Tensor(sess, tftensor) for tftensor in tfoperation.inputs)

    @classmethod
    def new(cls, sess, tfoperation):
        if tfoperation.type not in cls.classes:
            raise UnknownOperationException("unknown operation: %s" % tfoperation.type)

        return cls.classes[tfoperation.type](sess, tfoperation)

    def get(self, name):
        return reduce(lambda t1,t2: t1 or t2.get(name), self.inputs, None)

    def eval(self, feed_dict, _uuid):
        return self.func(*(t.eval(feed_dict=feed_dict, _uuid=_uuid) for t in self.inputs))

    @staticmethod
    def func():
        raise NotImplementedError

    @staticmethod
    def factory(func):
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
