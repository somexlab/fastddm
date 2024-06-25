# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FastDDM"
year = datetime.date.today().year
copyright = f"2023-{year}, Enrico Lattuada, Fabian Krautgasser, Roberto Cerbino"
author = "Enrico Lattuada, Fabian Krautgasser, Roberto Cerbino"
version = "0.3.12"
release = version

pygments_style = "friendly"
pygments_dark_style = "native"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx_tabs.tabs",
    "IPython.sphinxext.ipython_console_highlighting",
    "matplotlib.sphinxext.plot_directive",
]

napoleon_include_special_with_doc = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "lmfit": ("https://lmfit.github.io/lmfit-py/", None),
}
autodoc_docstring_signature = True
autodoc_typehints_format = "short"
autodoc_member_order = "groupwise"

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "fastddm-logo-vertical.svg"
html_theme_options = {
    "sidebar_hide_name": True,
    "top_of_page_buttons": [],
    "dark_css_variables": {
        "color-brand-primary": "#5187b2",
        "color-brand-content": "#5187b2",
    },
    "light_css_variables": {
        "color-brand-primary": "#406a8c",
        "color-brand-content": "#406a8c",
    },
}
html_favicon = "fastddm-logo-favicon.svg"
