# -*- coding: utf-8 -*-


import sys
import os
import shlex


sys.path.insert(0, os.path.abspath("../.."))

project = "tfdeploy"
author = "Marcel Rieger"
copyright = "2016, Marcel Rieger"
version = "0.1.7"
release = version

templates_path = ["_templates"]
html_static_path = ["_static"]
master_doc = "index"
source_suffix = ".rst"

exclude_patterns = []
pygments_style = "sphinx"
html_logo = "../../logo.png"
html_theme = "alabaster"

extensions = [
    "sphinx.ext.autodoc"
]
