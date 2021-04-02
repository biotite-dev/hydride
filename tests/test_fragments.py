# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
from biotite.structure.atoms import Atom
import pytest
import numpy as np
import biotite.structure as struc
import biotite.structure.info as info
import hydride


@pytest.mark.parametrize("res_name", [
    "BNZ", # Benzene
    "BZF", # Benzofuran
    "IND", # indole
    "PZO", # Pyrazole
    "BZI", # Benzimidazole
    "LOM", # Thiazole
    "P1R", # Pyrimidine
    "ISQ", # Isoquinoline
    "NPY", # Naphthalene
    "AN3", # Anthracene
    "0PY", # Pyridine
    "4FT", # Phthalazine
    "URA", # Uracil
    "CHX", # Cyclohexane
    "CN",  # Hydrogen cyanide
    "11X"  # N-pyridin-3-ylmethylaniline
])
def test_hydrogen_positions(res_name):
    """
    Test whether the assigned hydrogen positions approximately match
    the original hydrogen positions for a given molecule.
    
    All chosen molecules consist completely of heavy atoms without
    rotational freedom for the bonded hydrogen atoms, such as aromatic
    or cyclic compounds.
    This is required to reproduce unambiguously the original hydrogen
    positions, as no relaxation is performed.
    The chosen molecules are also part of the standard fragment library
    themselves.
    However, due to the mere size of the CCD it is improbable that
    a fragment in the library actually originates from the molecule it
    is added to.
    """
    TOLERANCE = 0.1

    library = hydride.FragmentLibrary.standard_library()
    molecule = info.residue(res_name)
    # Perform translation of the molecule along the three axes
    ref_hydrogen_coord = molecule.coord[molecule.element == "H"]
    
    heavy_atoms = molecule[molecule.element != "H"]
    test_hydrogen_coord = library.calculate_hydrogen_coord(heavy_atoms)
    
    test_count = 0
    for coord in test_hydrogen_coord:
        test_count += len(coord)
    assert test_count == np.count_nonzero(molecule.element == "H")
    assert len(test_hydrogen_coord) == heavy_atoms.array_length()
    ref_index = 0
    for hydrogen_coord in test_hydrogen_coord:
        # The coord for all hydrogens bonded to the same heavy atom
        if len(hydrogen_coord) == 0:
            # No bonded hydrogen atoms -> nothing to compare
            continue
        elif len(hydrogen_coord) == 1:
            # Only a single hydrogen atom
            # -> unambiguous assignment to reference hydrogen coord
            assert np.max(struc.distance(
                hydrogen_coord,
                ref_hydrogen_coord[ref_index : ref_index+1]
            )) <= TOLERANCE
            ref_index += 1
        elif len(hydrogen_coord) == 2:
            # Heavy atom has 2 hydrogen atoms
            # -> Since the hydrogen atoms are indistinguishable,
            # there are two possible assignment to reference hydrogens
            try:
                assert np.max(struc.distance(
                    hydrogen_coord,
                    ref_hydrogen_coord[ref_index : ref_index+2]
                )) <= TOLERANCE
            except AssertionError:
                assert np.max(struc.distance(
                    hydrogen_coord,
                    ref_hydrogen_coord[ref_index : ref_index+2 : -1]
                )) <= TOLERANCE
            ref_index += 2
        else:
            # Heavy atom has 3 hydrogen atoms
            # -> there is rotational freedom
            # -> invalid test case
            raise ValueError("Invalid test case")


def test_missing_fragment():
    """
    If a molecule contains an unknown fragment, check if a warning is
    raised and all other atoms are still hydrogenated.
    """
    TOLERANCE = 0.1

    lib = hydride.FragmentLibrary.standard_library()

    ref_mol = info.residue("BZI") # Benzimidazole
    # It should not be possible to have a nitrogen at this position
    # with a positive charge
    ref_mol.charge[0] = 1
    ref_hydrogen_coord = ref_mol.coord[ref_mol.element == "H"]


    test_mol = test_mol = ref_mol[ref_mol.element != "H"]
    with pytest.warns(
        UserWarning, match="Missing fragment for atom 'N1' at position 0"
    ):
        hydrogen_coord = lib.calculate_hydrogen_coord(test_mol)
    flattend_coord = []
    for coord in hydrogen_coord:
        flattend_coord += coord.tolist()
    test_hydrogen_coord = np.array(flattend_coord)
    
    assert np.max(struc.distance(
        test_hydrogen_coord,
        # Expect missing first hydrogen due to missing fragment
        ref_hydrogen_coord[1:]
    )) <= TOLERANCE


@pytest.mark.parametrize(
    "lib_enantiomer, subject_enantiomer",
    itertools.product(
        # L-alanine and D-alanine
        ["ALA", "DAL"],
        ["ALA", "DAL"],
    )
)
def test_stereocenter(lib_enantiomer, subject_enantiomer):
    """
    Test whether one enantiomer in the library is sufficient to
    add hydrogen for both enatiomers
    """
    TOLERANCE = 0.1

    lib = hydride.FragmentLibrary()
    lib.add_molecule(info.residue(lib_enantiomer))

    ref_model = info.residue(subject_enantiomer)

    test_model = ref_model[ref_model.element != "H"]
    # As the test case is constructed that the exact same molecule
    # can be in the library, move the molecule to assure that
    # correct hydrogen position calculation is not an artifact
    np.random.seed(0)
    test_model = struc.rotate(test_model, np.random.rand(3))
    test_model = struc.translate(test_model, np.random.rand(3))
    test_model, _ = hydride.add_hydrogen(test_model, fragment_library=lib)

    test_model, _ = struc.superimpose(
        ref_model, test_model, atom_mask = (test_model.element != "H")
    )
    ref_stereo_h_coord = ref_model.coord[ref_model.atom_name == "HA"][0]
    test_stereo_h_coord = test_model.coord[test_model.atom_name == "HA"][0]

    assert struc.distance(test_stereo_h_coord, ref_stereo_h_coord) <= TOLERANCE