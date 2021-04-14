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
from .util import data_dir


def test_estimate_amino_acid_charges():
    """
    Test :func:`estimate_amino_acid_charges()` based on known charges
    at neutral pH.
    """

    ref_charges = {
        ("ARG", "NH2") :  1,
        ("ASP", "OD2") : -1,
        ("CYS",  "SG") :  0,
        ("GLU", "OE2") : -1,
        ("HIS", "ND1") :  0,
        ("LYS",  "NZ") :  1,
        ("TYR",  "OH") :  0,
    }
    
    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), "1l2y.mmtf"))
    atoms = mmtf.get_structure(
        mmtf_file, model=1, include_bonds=True, extra_fields=["charge"]
    )
    heavy_atoms = atoms[atoms.element != "H"]
    
    test_charges = hydride.estimate_amino_acid_charges(heavy_atoms, 7.0)

    assert test_charges[0] == 1
    assert test_charges[-1] == -1
    for res_name, atom_name, test_charge in zip(
        heavy_atoms.res_name[1:-1],
        heavy_atoms.atom_name[1:-1],
        test_charges[1:-1]
    ):
        ref_charge = ref_charges.get((res_name, atom_name), 0)
        assert test_charge == ref_charge