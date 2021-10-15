# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import warnings
import re
from os.path import join, abspath, dirname, normpath
import fnmatch
import os
import pickle
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import msgpack
import numpy as np
import biotite.structure as struc
import biotite.structure.info as info
from biotite.structure.info.misc import _res_names


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
# Import is relative to working directory
from src.hydride import FragmentLibrary, AtomNameLibrary

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



# Compile Cython into C
try:
    cythonize(
        "src/**/*.pyx",
        include_path=[np.get_include()],
        language_level=3
    )
except ValueError:
    # This is a source distribution and the directory already contains
    # only C files
    pass


def get_extensions():
    ext_sources = []
    for dirpath, dirnames, filenames in os.walk(normpath("src/hydride")):
        for filename in fnmatch.filter(filenames, '*.c'):
            ext_sources.append(os.path.join(dirpath, filename))
    ext_names = [source
                 .replace("src"+normpath("/"), "")
                 .replace(".c", "")
                 .replace(normpath("/"), ".")
                 for source in ext_sources]
    ext_modules = [Extension(ext_names[i], [ext_sources[i]],
                             include_dirs=[np.get_include()])
                   for i in range(len(ext_sources))]
    return ext_modules



def get_protonation_variants():
    with open("prot_variants.msgpack", "rb") as file:
        molecule_data = msgpack.unpack(file, use_list=False, raw=False)
    molecules = []
    for molecule_dict in molecule_data.values():
        molecule = struc.AtomArray(len(molecule_dict["res_name"]))

        molecule.add_annotation("charge", int)

        molecule.res_name = molecule_dict["res_name"]
        molecule.atom_name = molecule_dict["atom_name"]
        molecule.element = molecule_dict["element"]
        molecule.charge = molecule_dict["charge"]
        molecule.hetero = molecule_dict["hetero"]

        molecule.coord[:,0] = molecule_dict["coord_x"]
        molecule.coord[:,1] = molecule_dict["coord_y"]
        molecule.coord[:,2] = molecule_dict["coord_z"]

        molecule.bonds = struc.BondList(
            molecule.array_length(),
            bonds = np.stack([
                molecule_dict["bond_i"],
                molecule_dict["bond_j"],
                molecule_dict["bond_type"]
            ]).T
        )

        molecules.append(molecule)
    
    return molecules

# Compile fragment library
std_fragment_library = FragmentLibrary()
# Add protonation variants at first because the bond lengths for neutral
# fragments might variate slightly
for mol in get_protonation_variants():
    std_fragment_library.add_molecule(mol)
mol_names = list(_res_names.keys()) + PROMINENT_MOLECULES
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i, mol_name in enumerate(mol_names):
        if not i % 100:
            print(
                f"Compiling fragment library... ({i}/{len(mol_names)})",
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
std_name_library = AtomNameLibrary()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # To reduce the size of the atom name library, only atom names
    # from prominent molecules are stored
    for i, mol_name in enumerate(PROMINENT_MOLECULES):
        print(
            f"Compiling atom name library... ({i+1}/{len(PROMINENT_MOLECULES)})",
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

    ext_modules = get_extensions(),

    # Include fragment and atom name libraries
    package_data = {"hydride" : ["*.pickle"]},

    entry_points = {"console_scripts": "hydride = hydride.cli:main"},
    
    install_requires = ["biotite >= 0.28",
                        "numpy >= 1.13"],
    python_requires = ">=3.7",
    
    tests_require = ["pytest"],
)


# Return to original directory
os.chdir(original_wd)

