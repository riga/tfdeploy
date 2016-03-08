# -*- coding: utf-8 -*-


import os
import sys
import unittest


# adjust the path to import tfdeploy
base = os.path.normpath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.append(base)
import tfdeploy


class TestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestCase, self).__init__(*args, **kwargs)

        self._cache = {}

    def get(self, model, *attrs):
        result = tuple()
        for attr in attrs:
            key = (model, attr)
            if key not in self._cache:
                tmp = __import__("tests.models." + model, globals(), locals(), [attr])
                self._cache[key] = getattr(tmp, attr)
            result += (self._cache[key],)
        return result if len(result) > 1 else result[0]
