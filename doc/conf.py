# This source code is part of the Gecos package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"

try:
    import numpy as np
    import pyximport
    pyximport.install(
        build_in_temp=False,
        setup_args={"include_dirs":np.get_include()},
        language_level=3
    )
except ImportError:
    pass

from os.path import realpath, dirname, join, isdir, isfile, basename
from os import listdir, makedirs
import sys

doc_path = dirname(realpath(__file__))
# Include gecos/src in PYTHONPATH
# in order to import the 'gecos' package
package_path = join(dirname(doc_path), "src")
sys.path.insert(0, package_path)
import hydride
# Include gecos/doc in PYTHONPATH
# in order to import modules for plot genaration etc.
sys.path.insert(0, doc_path)


#### General ####

extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.autosummary",
              "sphinx.ext.doctest",
              "sphinx.ext.mathjax",
              "sphinx.ext.viewcode",
              "numpydoc"]

templates_path = ["templates"]
source_suffix = [".rst"]
master_doc = "index"

project = "hydride"
copyright = "The Hydride contributors, 2021"
version = hydride.__version__

exclude_patterns = ["build"]

pygments_style = "sphinx"

todo_include_todos = False

# Prevents numpydoc from creating an autosummary which does not work
# due to Hydride's import system
numpydoc_show_class_members = False


#### HTML ####

html_theme = "alabaster"
html_static_path = ["static"]
html_css_files = [
    "hydride.css",
    "https://fonts.googleapis.com/css?" \
        "family=Crete+Round|Fira+Sans|&display=swap",
]
html_favicon = "static/assets/hydride_icon_32p.png"
htmlhelp_basename = "HydrideDoc"
html_sidebars = {"**": ["about.html",
                        "navigation.html",
                        "relations.html",
                        "searchbox.html",
                        "donate.html"]}
html_theme_options = {
    "description"   : "Adding hydrogen atoms to molecular models",
    "logo"          : "assets/hydride_logo.png",
    "logo_name"     : "true",
    "github_user"   : "biotite-dev",
    "github_repo"   : "hydride",
    "github_banner" : "true",
    "page_width"    : "85%",
    "fixed_sidebar" : "true"
}