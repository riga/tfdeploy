# -*- coding: utf-8 -*-


import sys
import os
import shlex


sys.path.insert(0, os.path.abspath(".."))
import tfdeploy as td


project = "tfdeploy"
author = td.__author__
copyright = td.__copyright__
version = td.__version__
release = td.__version__


templates_path = ["_templates"]
html_static_path = ["_static"]
master_doc = "index"
source_suffix = ".rst"


exclude_patterns = []
pygments_style = "sphinx"
html_logo = "../logo.png"
html_theme = "alabaster"


extensions = [
    "sphinx.ext.autodoc"
]
