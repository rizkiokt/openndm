"""Sphinx configuration for the OpenNDM documentation.

Built with Sphinx using a theme provided by Read the Docs.

The prose pages are MyST Markdown, because they were written as Markdown and
are readable on GitHub as they stand; the API reference is reStructuredText
driven by autodoc against the installed package. Building the API reference
therefore needs OpenNDM importable, which means the compiled extension must be
present — see docs/README.md.
"""

from __future__ import annotations

import importlib.metadata

# -- Project information -----------------------------------------------------

project = "OpenNDM"
author = "Rizki Oktavian, PhD"
copyright = "2026, Rizki Oktavian"

try:
    release = importlib.metadata.version("openndm")
except importlib.metadata.PackageNotFoundError:
    # A docs-only checkout without the package installed still builds; the
    # version simply shows as the development one.
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_parser",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# The prose pages cross-reference each other with relative .md links, which is
# how they read correctly on GitHub. MyST rewrites those to the built pages.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "linkify",
    "substitution",
]
myst_heading_anchors = 3

# -- Autodoc -----------------------------------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
# openndm.gc imports OpenMC, which is not installed on the docs builder and is
# far too heavy to install there. Autodoc renders its signatures and docstrings
# against a stub instead (FR-OMC-14 keeps this path optional at runtime too).
autodoc_mock_imports = ["openmc", "uncertainties"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False
# The dataclasses in openndm.gc document their fields in a numpydoc
# `Attributes` section *and* carry them as real annotated attributes.
# Rendering the section as `:ivar:` fields keeps the prose but stops it
# registering a second, competing index entry for every field.
napoleon_use_ivar = True

# -- nbsphinx ----------------------------------------------------------------

# The example notebooks are committed with their outputs already in place, so
# the docs build renders them rather than executing them. Notebook 02 needs
# OpenMC and a nuclear data library and could never run on the builder.
nbsphinx_execute = "never"
nbsphinx_allow_errors = False

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"OpenNDM {version}"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "sticky_navigation": True,
    "titles_only": False,
    "style_external_links": True,
    "prev_next_buttons_location": "both",
}
html_context = {
    "display_github": True,
    "github_user": "rizkiokt",
    "github_repo": "openndm",
    "github_version": "main",
    "conf_py_path": "/docs/",
}


# -- Bring the example notebooks into the source tree ------------------------

# The notebooks live in examples/ at the repository root, where they are meant
# to be run from, and Sphinx will not read sources from outside its source
# directory. They are copied in at build time rather than duplicated in the
# tree, so examples/ stays the single copy. docs/examples/ is gitignored.
def _copy_examples(app):
    import pathlib
    import shutil

    source = pathlib.Path(app.srcdir).parent / "examples"
    target = pathlib.Path(app.srcdir) / "examples"
    if not source.is_dir():
        return
    target.mkdir(exist_ok=True)
    for notebook in source.glob("*.ipynb"):
        shutil.copy2(notebook, target / notebook.name)


def setup(app):
    app.connect("builder-inited", _copy_examples)
