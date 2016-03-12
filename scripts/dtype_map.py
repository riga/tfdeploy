# -*- coding: utf-8 -*-

"""
Script to print a mapping between tensorflow and numpy dtypes
"""


import tensorflow as tf


# create the mapping
dtype_map = {}

# fill it
types_pb2 = tf.core.framework.types_pb2
for attr in dir(types_pb2):
    if attr.startswith("DT_"):
        tf_type_enum = getattr(types_pb2, attr)
        try:
            dtype_map[tf_type_enum] = "np." + tf.DType(tf_type_enum).as_numpy_dtype.__name__
        except:
            pass

# print dict-like code
dtype_map = "\n".join("    %s: %s," % tpl for tpl in dtype_map.items())
print("{\n" + dtype_map[:-1] + "\n}")
