#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import unittest


# adjust the path to import tensorfunk
base = os.path.normpath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.append(base)
import tensorfunk


class AllTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(AllTestCase, self).__init__(*args, **kwargs)


if __name__ == "__main__":
    unittest.main()
