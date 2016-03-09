# -*- coding: utf-8 -*-


import os
from distutils.core import setup
import tfdeploy as td


readme = "README.rst"
if os.path.isfile(readme):
    with open(readme) as f:
        long_description = f.read()
else:
    long_description = ""

keywords = [
    "tensorflow", "export", "dump", "numpy", "model", "predict", "evaluate"
]

classifiers = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Development Status :: 4 - Beta",
    "Operating System :: OS Independent",
    "License :: OSI Approved :: MIT License",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Information Technology",
    "Topic :: Scientific/Engineering :: Artificial Intelligence"
]


setup(
    name             = td.__name__,
    version          = td.__version__,
    author           = ", ".join(td.__credits__),
    description      = td.__doc__.strip(),
    license          = td.__license__,
    url              = td.__contact__,
    py_modules       = [td.__name__],
    keywords         = keywords,
    classifiers      = classifiers,
    long_description = long_description
)
