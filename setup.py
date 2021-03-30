# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import warnings
import re
import glob
from os.path import join, abspath, dirname
import os
import pickle
from setuptools import setup, find_packages
import biotite.structure.info as info
from biotite.structure.info.misc import _res_names
from src.hydride import FragmentLibrary, AtomNameLibrary


# Molecules that appear is most structures
# Hence, there is a high importance to get the hydrogen conformations
# for these molecules right
# Fragments from these molecules will overwrite any existing fragments
# from the standard library
PROMINENT_MOLECULES = [
    # Amino acids
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # Nucleotides
    "A", "C", "G", "T", "U", "DA", "DC", "DG", "DT", "DU", 
    # Solvent
    "HOH",
]


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


# Compile fragment library
mol_names = list(_res_names.keys()) + PROMINENT_MOLECULES
std_fragment_library = FragmentLibrary()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i, mol_name in enumerate(mol_names):
        print(
            f"Compiling fragment library... ({i+1}/{len(mol_names)})",
            end="\r"
        )
        try:
            mol = info.residue(mol_name)
        except KeyError:
            continue
        std_fragment_library.add_molecule(mol)
print("Compiling fragment library... Done" + " " * 20)
with open(join("src", "hydride", "fragments.pickle"), "wb") as fragments_file:
    pickle.dump(std_fragment_library._frag_dict, fragments_file)

# Compile atom name library
mol_names = list(_res_names.keys()) + PROMINENT_MOLECULES
std_name_library = AtomNameLibrary()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # To reduce the size of the atom name library, only atom names
    # from prominent molecules are stored
    for i, mol_name in enumerate(PROMINENT_MOLECULES):
        print(
            f"Compiling atom name library... ({i+1}/{len(mol_names)})",
            end="\r"
        )
        try:
            mol = info.residue(mol_name)
        except KeyError:
            continue
        std_name_library.add_molecule(mol)
print("Compiling atom name library... Done" + " " * 20)
with open(join("src", "hydride", "names.pickle"), "wb") as names_file:
    pickle.dump(std_name_library._name_dict, names_file)


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

    # Include fragment and atom name libraries
    package_data = {"hydride" : ["*.pickle"]},
    
    install_requires = ["biotite >= 0.27",
                        "numpy >= 1.13"],
    python_requires = ">=3.6",
    
    tests_require = ["pytest"],
)


# Return to original directory
os.chdir(original_wd)

