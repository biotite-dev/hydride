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


def test_molecule_without_hydrogens():
    """
    Test whether the :func:`add_hydrogen()` can handle molecules, where
    no hydrogen atom should be added.
    It is expected that simply the original structure is returned
    """
    # Chlordecone
    ref_molecule = info.residue("27E")

    test_molecule, original_mask = hydride.add_hydrogen(ref_molecule)

    assert np.all(original_mask)
    assert test_molecule == ref_molecule


@pytest.mark.parametrize(
    "path",
    glob.glob(join(data_dir(), "*.mmtf"))
)
def test_original_mask(path):
    """
    Check whether the returned original atom mask is correct
    by applying it to the hydrogenated structure and expect that the
    original structure is restored
    """
    mmtf_file = mmtf.MMTFFile.read(path)
    ref_model = mmtf.get_structure(
        mmtf_file, model=1, include_bonds=True, extra_fields=["charge"]
    )
    ref_model = ref_model[ref_model.element != "H"]
    
    hydrogenated_model, original_mask = hydride.add_hydrogen(ref_model)
    test_model = hydrogenated_model[original_mask]

    assert hydrogenated_model.array_length() > ref_model.array_length()
    assert test_model == ref_model