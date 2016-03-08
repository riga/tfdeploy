# -*- coding: utf-8 -*-


from distutils.core import setup
import tfdeploy as td


with open("README.md") as f:
    long_description = f.read()

keywords = [
    "tensorflow", "export", "dump", "numpy", "model", "predict", "evaluate"
]

classifiers = [
    "Programming Language :: Python",
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3"
]


setup(
    name             = td.__name__,
    version          = td.__version__,
    author           = ", ".join(td.__credits__),
    description      = td.__doc__.strip(),
    license          = td.__license__,
    url              = "https://github.com/riga/" + td.__name__,
    py_modules       = [td.__name__],
    keywords         = keywords,
    classifiers      = classifiers,
    long_description = long_description
)
