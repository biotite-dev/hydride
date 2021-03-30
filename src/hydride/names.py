# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "hydride"
__author__ = "Patrick Kunzmann"
__all__ = ["AtomNameLibrary"]

from os.path import join, dirname, abspath
import pickle
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
            for i, hydrogen_name in enumerate(hydrogen_names):
                yield hydrogen_name
            i += 1
        else:
            i = 0
        
        # TODO: Better naming: C42-> H42, H42A, H42B
        # The generated hydrogen atom name is one character longer
        # than the heavy atom name, but the hydrogen atom name must
        # not exceed 4 characters (PDB limit)
        if len(heavy_atom_name) > 1 and len(heavy_atom_name) < 4:
            # e.g. CA -> HA, HA2, HA3, ...
            suffix = heavy_atom_name[1:]
            while True:
                yield f"H{suffix}{i+1}"
                i += 1

        else:
            # X -> H, H2, H3, ...
            while True:
                yield f"H{i+1}"
                i += 1
        