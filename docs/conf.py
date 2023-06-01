# Distributed under the MIT License.
# See LICENSE.txt for details.

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

# -- Project information -----------------------------------------------------

project = "SpECTRE Python interface"
copyright = "2017-2023, SXS Collaboration"
author = "SXS Collaboration"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    # Parse Numpy-style docstrings
    "sphinx.ext.napoleon",
    # Support markdown files
    "myst_parser",
    # "breathe",
]

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
}
autodoc_preserve_defaults = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- C++ documentation with Breathe + Exhale ---------------------------------

# We have also experimented with generating our C++ documentation with Sphinx.
# See issue: https://github.com/sxs-collaboration/spectre/issues/2138

# Tell sphinx what the primary language being documented is.
# primary_domain = "cpp"

# Tell sphinx what the pygments highlight language should be.
# highlight_language = "cpp"

# Configure Breathe for parsing and including C++ documentation.
# breathe_projects = {"SpECTRE": "@CMAKE_BINARY_DIR@/docs/xml"}
# breathe_default_project = "SpECTRE"

# Setup the Exhale extension for automatically building namespace,
# file, and group documentation. We have had issues with Exhale
# failing when dealing with function overloads. To enable you must
# add `'exhale'` to the extensions.
#
# exhale_args = {
#     # These arguments are required
#     "containmentFolder": "./api",
#     "rootFileName": "library_root.rst",
#     "rootFileTitle": "Library API",
#     "doxygenStripFromPath": "..",
#     # Suggested optional arguments
#     "createTreeView": True,
#     # TIP: if using the sphinx-bootstrap-theme, you need
#     # "treeViewIsBootstrap": True,
#     "exhaleExecutesDoxygen": False
#     # "exhaleDoxygenStdin": "INPUT = ../include"
# }

# # a simple label style which uses the bibtex keys for labels
# class NumberedLabelStyle(BaseLabelStyle):
#     def format_labels(self, sorted_entries):
#         for entry in sorted_entries:
#             # Add one since refs usually start at 1 not 0
#             yield str(sorted_entries.index(entry) + 1)

# class SpectreBibStyle(UnsrtStyle):
#     default_label_style = NumberedLabelStyle

# register_plugin('pybtex.style.formatting', 'SpectreStyle', SpectreBibStyle)
