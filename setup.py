# -*- coding: utf-8 -*-


import os
from subprocess import Popen, PIPE
from distutils.core import setup
import tfdeploy as td


readme = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
if os.path.isfile(readme):
    cmd = "pandoc --from=markdown --to=rst " + readme
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, executable="/bin/bash")
    out, err = p.communicate()
    if p.returncode != 0:
        raise Exception("pandoc conversion failed: " + err)
    long_description = out
else:
    long_description = ""

keywords = [
    "tensorflow", "deploy", "export", "dump", "numpy", "model", "predict", "evaluate", "function",
    "method"
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
    author           = td.__author__,
    description      = td.__doc__.strip(),
    license          = td.__license__,
    url              = td.__contact__,
    py_modules       = [td.__name__],
    keywords         = keywords,
    classifiers      = classifiers,
    long_description = long_description,
    data_files       = ["LICENSE", "requirements.txt"]
)
