# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import os
import pickle
import warnings
from os.path import abspath, dirname, isfile, join
import biotite.structure as struc
import biotite.structure.info as info
import msgpack
import numpy as np
from biotite.structure.info.ccd import get_ccd
from setuptools import find_packages, setup

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
from src.hydride import AtomNameLibrary, FragmentLibrary, __version__  # noqa: E402


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


def get_mol_names_in_ccd():
    ccd = get_ccd()
    atom_category = ccd["chem_comp_atom"]
    return np.unique(atom_category["comp_id"].as_array()).tolist()


# Compile fragment library
fragment_file_path = join("src", "hydride", "fragments.pickle")
if not isfile(fragment_file_path):
    std_fragment_library = FragmentLibrary()
    # Add protonation variants at first because the bond lengths
    # for neutral fragments might vary slightly
    for mol in get_protonation_variants():
        std_fragment_library.add_molecule(mol)
    mol_names = get_mol_names_in_ccd() + PROMINENT_MOLECULES
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, mol_name in enumerate(mol_names):
            if not i % 100:
                print(f"Compiling fragment library... ({i}/{len(mol_names)})", end="\r")
            try:
                mol = info.residue(mol_name)
            except KeyError:
                continue
            std_fragment_library.add_molecule(mol)
    print("Compiling fragment library... Done" + " " * 20)
    with open(fragment_file_path, "wb") as fragments_file:
        pickle.dump(std_fragment_library._frag_dict, fragments_file)


# Compile atom name library
names_file_path = join("src", "hydride", "names.pickle")
if not isfile(names_file_path):
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
    print("Compiling atom name library... Done" + " " * 20)
    with open(names_file_path, "wb") as names_file:
        pickle.dump(std_name_library._name_dict, names_file)


setup(
    version=__version__,
    zip_safe=False,
    packages=find_packages("src"),
    package_dir={"": "src"},
    # Include fragment and atom name libraries
    package_data={"hydride": ["*.pickle"]},
)


# Return to original directory
os.chdir(original_wd)
