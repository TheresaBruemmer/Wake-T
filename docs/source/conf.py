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
sys.path.insert(0, os.path.abspath('../..'))


# -- Import Wake-T version ---------------------------------------------------
from wake_t import __version__  # noqa: E402


# -- Project information -----------------------------------------------------
project = 'Wake-T'
project_copyright = '2021, Ángel Ferran Pousa'
author = 'Ángel Ferran Pousa'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon',
    'sphinx_panels', 'sphinx_gallery.gen_gallery'
    ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'  # "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Logo
html_logo = "_static/logo.png"
html_favicon = "_static/favicon_128x128.png"

# Theme options
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/AngelFP/Wake-T",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Slack",
            "url": "https://wake-t.slack.com/",
            "icon": "fab fa-slack",
        },
    ],
}

# Prevent panels extension from modifying page style.
panels_add_bootstrap_css = False

# Document __init__ class methods
autoclass_content = 'both'

# Configuration for generating tutorials.
from sphinx_gallery.sorting import FileNameSortKey  # noqa: E402

sphinx_gallery_conf = {
     'examples_dirs': '../../tutorials',
     'gallery_dirs': 'tutorials',
     'filename_pattern': '.',
     'within_subsection_order': FileNameSortKey,
}
