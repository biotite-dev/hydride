# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "hydride"
__author__ = "Patrick Kunzmann"
__all__ = ["AtomNameLibrary"]

from os.path import join, dirname, abspath
import pickle
import string
import numpy as np
from biotite.structure.error import BadStructureError


class AtomNameLibrary:

    _std_library = None

    def __init__(self):
        self._name_dict = {}
    

    @staticmethod
    def standard_library():
        if AtomNameLibrary._std_library is None:
            AtomNameLibrary._std_library = AtomNameLibrary()
            file_name = join(dirname(abspath(__file__)), "names.pickle")
            with open(file_name, "rb") as names_file:
                AtomNameLibrary._std_library._name_dict \
                    = pickle.load(names_file)
        return AtomNameLibrary._std_library
    

    def add_molecule(self, molecule):
        if molecule.bonds is None:
            raise BadStructureError(
                "The input molecule must have an associated BondList"
            )
        
        all_bond_indices, _ = molecule.bonds.get_all_bonds()

        for i in np.where(molecule.element != "H")[0]:
            bonded_indices = all_bond_indices[i]
            # Remove padding values
            bonded_indices = bonded_indices[bonded_indices != -1]
            # Set atom names of bonded hydrogen atoms as values
            self._name_dict[(molecule.res_name[i], molecule.atom_name[i])] = [
                molecule.atom_name[j] for j in bonded_indices
                if molecule.element[j] == "H"
            ]

        
    def generate_hydrogen_names(self, heavy_res_name, heavy_atom_name):
        hydrogen_names = self._name_dict.get((heavy_res_name, heavy_atom_name))
        if hydrogen_names is not None:
            # Hydrogen names from library
            for i, hydrogen_name in enumerate(hydrogen_names):
                yield hydrogen_name
            base_name = hydrogen_name[:-1]
            number = int(hydrogen_name[-1])
            while True:
                # Proceed by increasing the atom number
                # e.g. CB -> HB1, HB2, HB3, ...
                number += 1
                yield f"{base_name}{number}"

        
        else:
            if heavy_atom_name[-1] in string.digits:
                # Atom name ends with number
                # -> assume ligand atom naming
                # C42 -> H42, H42A, H42B
                number = int(
                    ''.join([c for c in heavy_atom_name if c.isdigit()])
                )
                yield f"H{number}"
                i = 0
                while True:
                    yield f"H{number}{string.ascii_uppercase[i]}"
                    i += 1
            elif len(heavy_atom_name) > 1:
                # e.g. CA -> HA, HA2, HA3, ...
                suffix = heavy_atom_name[1:]
                yield f"H{suffix}"
                number = 1
                while True:
                    yield f"H{suffix}{number}"
                    number += 1

            else:
                # N -> H, H2, H3, ...
                yield "H"
                number = 1
                while True:
                    yield f"H{number}"
                    number += 1
        