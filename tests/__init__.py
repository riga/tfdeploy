# -*- coding: utf-8 -*-

# some logs
import os
import sys
import numpy as np
import tensorflow as tf
print(80 * "-")
print("python    : " + sys.version.split(" ")[0])
print("numpy     : " + np.version.version)
print("tensorflow: " + tf.__version__)
try:
    import scipy as sp
    spv = sp.version.version
except:
    spv = "NONE"
print("scipy     : " + spv)
envkeys = [key for key in os.environ.keys() if key.startswith("TD_")]
if envkeys:
    print("-")
    maxlen = max(len(key) for key in envkeys)
    for key in envkeys:
        print(key + (maxlen - len(key)) * " " + ": " + os.environ[key])
print(80 * "-")


# import all tests
from .core import *
from .ops import *
