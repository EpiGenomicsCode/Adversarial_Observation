# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../Adversarial_Observation'))
sys.path.insert(2, os.path.abspath('../'))
from Adversarial_Observation.Attacks import *
from Adversarial_Observation.utils import *
from Adversarial_Observation.visualize import *

# -- Project information -----------------------------------------------------

project = 'Adversarial_Observation'
copyright = '2023, Jamil Gafur'
author = 'Jamil Gafur'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
#  we want to document the __init__ method of our classes, we need to add the autodoc extension to the extensions list.

extensions = []
extensions.append('sphinx.ext.autodoc')
extensions.append('sphinx.ext.napoleon')
extensions.append('sphinx.ext.viewcode')
extensions.append('sphinx.ext.autosummary')
extensions.append('sphinx.ext.mathjax')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']