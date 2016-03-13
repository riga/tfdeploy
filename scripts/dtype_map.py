# -*- coding: utf-8 -*-

"""
Script that prints a mapping of tensorflow dtype nums to and numpy dtypes, e.g.:

> python dtype_map.py
dtype_map = {
    1: np.float32,
    2: np.float64,
    3: np.int32,
    4: np.uint8,
    ...
}
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
