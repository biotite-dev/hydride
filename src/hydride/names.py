# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "hydride"
__author__ = "Patrick Kunzmann"
__all__ = ["AtomNameLibrary"]

import functools
import itertools
import pickle
import string
from os.path import abspath, dirname, join
import numpy as np
from biotite.structure.error import BadStructureError


class AtomNameLibrary:
    """
    A library for generating hydrogen atom names.

    For each molecule added to the :class:`AtomNameLibrary`,
    the hydrogen atom names are saved for each heavy atom in this
    molecule.

    If hydrogen atom names should be generated for a heavy atom,
    the library first looks for a corresponding entry in the library.
    If such entry is not found, since the molecule was never added to
    the library, the hydrogen atom names are guessed based on common
    hydrogen naming schemes.
    """

    def __init__(self):
        self._name_dict = {}

    @functools.cache
    @staticmethod
    def standard_library():
        """
        Get the standard :class:`AtomNameLibrary`.
        The library contains atom names for the most prominent molecules
        including amino acids and nucleotides.

        Returns
        -------
        library : AtomNameLibrary
            The standard library.
        """
        name_library = AtomNameLibrary()
        file_name = join(dirname(abspath(__file__)), "names.pickle")
        with open(file_name, "rb") as names_file:
            name_library._name_dict = pickle.load(names_file)
        return name_library

    def add_molecule(self, molecule):
        """
        Add the hydrogen atom names for each heavy atom in the molecule
        to the library.

        Parameters
        ----------
        molecule : AtomArray
            The molecule to use the hydrogen atom names from.
        """
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
            hydrogen_names = [
                molecule.atom_name[j]
                for j in bonded_indices
                if molecule.element[j] == "H"
            ]
            if len(hydrogen_names) > 0:
                self._name_dict[(molecule.res_name[i], molecule.atom_name[i])] = (
                    hydrogen_names
                )

    def generate_hydrogen_names(self, heavy_res_name, heavy_atom_name):
        """
        Generate hydrogen atom names for the given residue and heavy
        atom name.

        If the residue is not found in the library, the hydrogen atom
        name is guessed based on common hydrogen naming schemes.
        """
        hydrogen_names = self._name_dict.get((heavy_res_name, heavy_atom_name))
        if hydrogen_names is not None:
            # Hydrogen names from library
            for i, hydrogen_name in enumerate(hydrogen_names):
                yield hydrogen_name
            try:
                base_name = hydrogen_name[:-1]
                for number in itertools.count(int(hydrogen_name[-1]) + 1):
                    # Proceed by increasing the atom number
                    # e.g. CB -> HB1, HB2, HB3, ...
                    yield f"{base_name}{number}"
            except ValueError:
                # Atom name has no number at the end
                # -> simply append number
                for number in itertools.count(1):
                    yield f"{hydrogen_name}{number}"

        else:
            if len(heavy_atom_name) == 0:
                # Atom array has no atom names
                # (loaded e.g. from MOL file)
                # -> Also no atom names for hydrogen atoms
                while True:
                    yield ""
            if heavy_atom_name[-1] in string.digits:
                # Atom name ends with number
                # -> assume ligand atom naming
                # C1 -> H1, H1A, H1B
                number = int("".join([c for c in heavy_atom_name if c.isdigit()]))
                heavy_atom_name[0]
                # C1 -> H1, H1A, H1B
                yield f"H{number}"
                for i in itertools.count():
                    yield f"H{number}{string.ascii_uppercase[i]}"
            elif len(heavy_atom_name) > 1:
                # e.g. CA -> HA, HA2, HA3, ...
                suffix = heavy_atom_name[1:]
                yield f"H{suffix}"
                for number in itertools.count(1):
                    yield f"H{suffix}{number}"

            else:
                # N -> H, H2, H3, ...
                yield "H"
                for number in itertools.count(1):
                    yield f"H{number}"
