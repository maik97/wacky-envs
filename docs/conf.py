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
sys.path.insert(0, os.path.abspath('../'))
import wacky_envs

#import sphinx_rtd_theme


# -- Project information -----------------------------------------------------

project = 'wacky-env'
copyright = '2022, Maik Schürmann'
author = 'Maik Schürmann'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    #'sphinxcontrib.plantuml',
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
]


#plantuml = 'java -jar plantuml.jar'

# -- GraphViz configuration ----------------------------------
graphviz_output_format = 'svg'
inheritance_graph_attrs = dict(rankdir="TD")#, size='"6.0, 8.0"',
                               #fontsize=14, ratio='compress')
#graphviz_dot = 'graphviz/bin/dot.exe'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'python'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_mock_imports = ['torch', 'numpy', 'scipy', 'pandas', 'matplotlib', 'pygame', 'gym']
autosummary_mock_imports = ['torch', 'numpy', 'scipy', 'pandas', 'matplotlib', 'pygame', 'gym']
autosummary_generate = True

autodoc_member_order = 'bysource'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']