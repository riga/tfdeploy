# -*- coding: utf-8 -*-


from distutils.core import setup


setup(
    name         = "tensorfunk",
    version      = "0.0.0",
    packages     = ["tensorfunk"],
    description  = "tensorflow model converter to create tensorflow-independent prediction "
                   "functions.",
    author       = "Marcel Rieger",
    author_email = "marcelrieger@icloud.com",
    url          = "https://github.com/riga/pymitter",
    keywords     = [
        "tensorflow", "export", "dump", "numpy", "model", "predict", "evaluate"
    ],
    classifiers  = [
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3"
    ],
    long_description = """\
tensorfunk
==========

tensorflow model converter to create tensorflow-independent prediction functions.

"""
)
