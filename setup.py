# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import fnmatch
import os
import pickle
import warnings
from os.path import abspath, dirname, isfile, join, normpath
import biotite.structure as struc
import biotite.structure.info as info
import msgpack
import numpy as np
from biotite.structure.info.ccd import get_ccd
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

# Molecules that appear is most structures
# Hence, there is a high importance to get the hydrogen conformations
# for these molecules right
# Fragments from these molecules will overwrite any existing fragments
# from the standard library
PROMINENT_MOLECULES = [
    # Amino acids
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    # histidine tautomers
    "HID",
    "HIE",
    "HIP",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    # Nucleotides
    "A",
    "C",
    "G",
    "T",
    "U",
    "DA",
    "DC",
    "DG",
    "DT",
    "DU",
    # Solvent
    "HOH",
]


original_wd = os.getcwd()
# Change directory to setup directory to ensure correct file identification
os.chdir(dirname(abspath(__file__)))
# Import is relative to working directory
from src.hydride import AtomNameLibrary, FragmentLibrary, __version__

# Compile Cython into C
try:
    cythonize("src/**/*.pyx", include_path=[np.get_include()], language_level=3)
except ValueError:
    # This is a source distribution and the directory already contains
    # only C files
    pass


def get_extensions():
    ext_sources = []
    for dirpath, _, filenames in os.walk(normpath(join("src", "hydride"))):
        for filename in fnmatch.filter(filenames, "*.c"):
            ext_sources.append(os.path.join(dirpath, filename))
    ext_names = [
        source.replace("src" + normpath("/"), "")
        .replace(".c", "")
        .replace(normpath("/"), ".")
        for source in ext_sources
    ]
    ext_modules = [
        Extension(ext_names[i], [ext_sources[i]], include_dirs=[np.get_include()])
        for i in range(len(ext_sources))
    ]
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

        molecule.coord[:, 0] = molecule_dict["coord_x"]
        molecule.coord[:, 1] = molecule_dict["coord_y"]
        molecule.coord[:, 2] = molecule_dict["coord_z"]

        molecule.bonds = struc.BondList(
            molecule.array_length(),
            bonds=np.stack(
                [
                    molecule_dict["bond_i"],
                    molecule_dict["bond_j"],
                    molecule_dict["bond_type"],
                ]
            ).T,
        )

        molecules.append(molecule)

    return molecules


def get_histidine_variants():
    """Get the CCD histidine protonation variants under PDB names.

    The protonation-variant file uses internal names for optimized
    structures. The three records below are chemically equivalent to the
    MoleculeKit variants, but their internal names must not leak into
    the generated name library.
    """
    variant_names = {
        "HIS_LEO2H_DHE2": "HID",  # ND1-H, NE2 unprotonated
        "HIS_LEO2H_DHD1": "HIE",  # ND1 unprotonated, NE2-H
        "HIS_LEO2H": "HIP",      # ND1-H and NE2-H, cation
    }
    variants = []
    for molecule in get_protonation_variants():
        source_name = molecule.res_name[0]
        target_name = variant_names.get(source_name)
        if target_name is None:
            continue
        molecule.res_name[:] = target_name
        variants.append(molecule)
    return variants


def get_mol_names_in_ccd():
    ccd = get_ccd()
    atom_category = ccd["chem_comp_atom"]
    return np.unique(atom_category["comp_id"].as_array()).tolist()


def fragment_library_needs_update(file_path):
    """Check whether a generated fragment library predates histidine variants."""
    if not isfile(file_path):
        return True

    try:
        with open(file_path, "rb") as file:
            fragment_dict = pickle.load(file)
    except (OSError, pickle.UnpicklingError, EOFError):
        return True

    return not {"HID", "HIE", "HIP"}.issubset(
        {value[0] for value in fragment_dict.values()}
    )


def name_library_needs_update(file_path):
    """Check whether a generated atom-name library predates histidine variants."""
    if not isfile(file_path):
        return True

    try:
        with open(file_path, "rb") as file:
            name_dict = pickle.load(file)
    except (OSError, pickle.UnpicklingError, EOFError):
        return True

    return any(
        not any(key[0] == res_name for key in name_dict)
        for res_name in ("HID", "HIE", "HIP")
    )


# Compile fragment library
fragment_file_path = join("src", "hydride", "fragments.pickle")
if fragment_library_needs_update(fragment_file_path):
    std_fragment_library = FragmentLibrary()
    # Add protonation variants at first because the bond lengths
    # for neutral fragments might vary slightly
    for mol in get_protonation_variants():
        std_fragment_library.add_molecule(mol)
    # Add the MoleculeKit histidine variants before the canonical HIS
    # reference. Their tautomer-specific local topology and formal charges
    # select distinct fragments; HIS remains the preferred reference for any
    # fragment shared with the canonical residue.
    for mol in get_histidine_variants():
        std_fragment_library.add_molecule(mol)
    mol_names = get_mol_names_in_ccd() + PROMINENT_MOLECULES
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, mol_name in enumerate(mol_names):
            if not i % 100:
                print(f"Compiling fragment library... ({i}/{len(mol_names)})", end="\r")
            try:
                mol = info.residue(mol_name)
            except ValueError:
                # Molecule has missing coordinates
                continue
            std_fragment_library.add_molecule(mol)
    print("Compiling fragment library... Done" + " " * 20)
    with open(fragment_file_path, "wb") as fragments_file:
        pickle.dump(std_fragment_library._frag_dict, fragments_file)


# Compile atom name library
names_file_path = join("src", "hydride", "names.pickle")
if name_library_needs_update(names_file_path):
    std_name_library = AtomNameLibrary()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # To reduce the size of the atom name library, only atom names
        # from prominent molecules are stored
        for i, mol_name in enumerate(PROMINENT_MOLECULES):
            print(
                f"Compiling atom name library... "
                f"({i+1}/{len(PROMINENT_MOLECULES)})",
                end="\r",
            )
            try:
                mol = info.residue(mol_name)
            except KeyError:
                continue
            std_name_library.add_molecule(mol)
        for mol in get_histidine_variants():
            std_name_library.add_molecule(mol)
    print("Compiling atom name library... Done" + " " * 20)
    with open(names_file_path, "wb") as names_file:
        pickle.dump(std_name_library._name_dict, names_file)


setup(
    version=__version__,
    zip_safe=False,
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=get_extensions(),
    # Include fragment and atom name libraries
    package_data={"hydride": ["*.pickle"]},
)


# Return to original directory
os.chdir(original_wd)
