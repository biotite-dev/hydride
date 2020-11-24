# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import re
import glob
from os.path import join, abspath, dirname
import os
from setuptools import setup, find_packages

original_wd = os.getcwd()
# Change directory to setup directory to ensure correct file identification
os.chdir(dirname(abspath(__file__)))

# Simply import long description from README file
with open("README.rst") as readme:
    long_description = readme.read()

# Parse the top level package for the version
# Do not use an import to prevent side effects
# e.g. required runtime dependencies
with open(join("src", "hydride", "__init__.py")) as init_file:
    for line in init_file.read().splitlines():
        if line.lstrip().startswith("__version__"):
            version_match = re.search('".*"', line)
            if version_match:
                # Remove quotes
                version = version_match.group(0)[1 : -1]
            else:
                raise ValueError("No version is specified in '__init__.py'")


setup(
    name="hydride",
    version = version,
    description = "Adding hydrogen atoms to molecular models",
    long_description = long_description,
    author = "The 'Hydride' contributors",
    license = "BSD 3-Clause",
    classifiers = [
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    url = "https://hydride.biotite-python.org",
    project_urls = {
        "Documentation": "https://hydride.biotite-python.org",
        "Repository": "https://github.com/biotite-dev/hydride",
    },
    
    zip_safe = False,
    packages = find_packages("src"),
    package_dir = {"" : "src"},
    
    install_requires = [#"biotite >= x.xx",
                        "numpy >= 1.13"],
    python_requires = ">=3.6",
    
    tests_require = ["pytest"],
)


# Return to original directory
os.chdir(original_wd)

