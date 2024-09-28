# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure.info as info
import numpy as np
import pytest
import hydride


@pytest.mark.parametrize(
    "heavy_atom_name, ref_hydrogen_names",
    [
        ("N", ["H", "H1", "H2"]),
        ("CB", ["HB", "HB1", "HB2"]),
        ("C42", ["H42", "H42A", "H42B"]),
    ],
)
def test_generated_names(heavy_atom_name, ref_hydrogen_names):
    """
    Test correct generation of hydrogen names for atoms that are not in
    the library, based on known examples.
    """
    name_lib = hydride.AtomNameLibrary()
    gen = name_lib.generate_hydrogen_names("", heavy_atom_name)
    test_hydrogen_names = [next(gen) for _ in range(len(ref_hydrogen_names))]

    assert test_hydrogen_names == ref_hydrogen_names


np.random.seed(0)
res_names = info.all_residues()


@pytest.mark.parametrize(
    "res_name",
    [
        res_names[i]
        for i in np.random.choice(np.arange(len(res_names)), size=1000, replace=False)
    ],
)
def test_names_from_library(res_name):
    """
    Test correct lookup of hydrogen names for atoms that are in the
    library.
    Use a random selection of molecules from the CCD,
    and check whether the created hydrogen names are equal to the
    existing ones.
    """
    try:
        residue = info.residue(res_name)
    except KeyError:
        pytest.skip("No structure available for residue name")
    name_lib = hydride.AtomNameLibrary()
    name_lib.add_molecule(residue)
    for i in np.where(residue.element != "H")[0]:
        bond_indices, _ = residue.bonds.get_bonds(i)
        ref_hydrogen_names = [
            residue.atom_name[j] for j in bond_indices if residue.element[j] == "H"
        ]

        gen = name_lib.generate_hydrogen_names(res_name, residue.atom_name[i])
        test_hydrogen_names = [next(gen) for _ in range(len(ref_hydrogen_names))]

        assert test_hydrogen_names == ref_hydrogen_names
