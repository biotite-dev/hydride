# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import glob
from os.path import join
import pytest
import numpy as np
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.mmtf as mmtf
import hydride
from hydride.relax import _find_rotatable_bonds
from .util import data_dir


def test_hydrogen_positions():
    """
    Check whether the relaxation algorithm is able to restore a high 
    percentage of the number of original hydrogen positions.
    """
    pass


def test_hydrogen_bonds():
    """
    Check whether the relaxation algorithm is able to restore most of
    the original hydrogen bonds.
    The number of bonds found without relaxation is handled as baseline.
    """
    PERCENTAGE = 0.8

    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), "1l2y.mmtf"))
    atoms = mmtf.get_structure(
        mmtf_file, model=1, include_bonds=True, extra_fields=["charge"]
    )
    ref_num = len(struc.hbond(atoms))
    
    atoms = atoms[atoms.element != "H"]
    atoms, _ = hydride.add_hydrogen(atoms)
    base_num = len(struc.hbond(atoms))

    atoms.coord = hydride.relax_hydrogen(atoms, iteration_number=1000)
    test_num = len(struc.hbond(atoms))

    if base_num == ref_num:
        ValueError(
            "Invalid test case, "
            "no further hydrogen bonds can be found via relaxation"
        )
    assert (test_num - base_num) / (ref_num - base_num) >= PERCENTAGE


@pytest.mark.parametrize(
    "res_name, ref_bonds",
    [
        ("FRU", [
            ( "O1",  "C1",  True, ("HO1",)),
            ( "O2",  "C2",  True, ("HO2",)),
            ( "O3",  "C3",  True, ("HO3",)),
            ( "O4",  "C4",  True, ("HO4",)),
            ( "O6",  "C6",  True, ("HO6",)),
        ]),
        ("ARG", [
            (  "N",  "CA",  True, ("H", "H2")),
            ("OXT",   "C",  True, ("HXT",)),
        ]),
        ("HOH", [
        ]),
    ]
)
def test_bond_identification(res_name, ref_bonds):
    """
    Test whether rotatable bonds for the relaxation are correctly
    identified based on known molecules.
    """
    molecule = info.residue(res_name)
    rotatable_bonds = _find_rotatable_bonds(molecule)

    ref_bonds = set(ref_bonds)

    assert len(rotatable_bonds) == len(ref_bonds)
    for center_atom_i, bonded_atom_i, is_free, h_indices in rotatable_bonds:
        bond_tuple = (
            molecule.atom_name[center_atom_i],
            molecule.atom_name[bonded_atom_i],
            is_free,
            tuple(np.sort(molecule.atom_name[h_indices]))
        )
        assert bond_tuple in ref_bonds