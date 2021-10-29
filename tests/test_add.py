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
    "CEJ", # 1,3-Cyclopentanedione
    "CN",  # Hydrogen cyanide
    "11X", # N-pyridin-3-ylmethylaniline
    "ANL", # Aniline
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
    ref_molecule = info.residue(res_name)

    test_molecule = ref_molecule[ref_molecule.element != "H"]
    test_molecule, _ = hydride.add_hydrogen(test_molecule)

    for category in ref_molecule.get_annotation_categories():
        if category == "atom_name":
            # Atom names are tested separately
            continue
        try:
            assert np.all(
                test_molecule.get_annotation(category) ==
                ref_molecule.get_annotation(category)
            )
        except AssertionError:
            print("Failing category:", category)
            raise
    # Atom names are only guessed
    # -> simply check if the atoms names are unique
    assert len(np.unique(test_molecule.atom_name)) \
        == len(test_molecule.atom_name)
    
    for heavy_i in np.where(ref_molecule.element != "H")[0]:
        ref_bond_i, _ = ref_molecule.bonds.get_bonds(heavy_i)
        ref_h_indices = ref_bond_i[ref_molecule.element[ref_bond_i] == "H"]
        test_bond_i, _ = test_molecule.bonds.get_bonds(heavy_i)
        test_h_indices = test_bond_i[test_molecule.element[test_bond_i] == "H"]
        # The coord for all hydrogens bonded to the same heavy atom
        if len(ref_h_indices) == 0:
            # No bonded hydrogen atoms -> nothing to compare
            continue
        elif len(ref_h_indices) == 1:
            # Only a single hydrogen atom
            # -> unambiguous assignment to reference hydrogen coord
            assert np.max(struc.distance(
                test_molecule.coord[test_h_indices],
                ref_molecule.coord[ref_h_indices],
            )) <= TOLERANCE
        elif len(ref_h_indices) == 2:
            # Heavy atom has 2 hydrogen atoms
            # -> Since the hydrogen atoms are indistinguishable,
            # there are two possible assignment to reference hydrogens
            best_distance = min([
                np.max(struc.distance(
                    test_molecule.coord[test_h_indices],
                    ref_molecule.coord[ref_h_indices][::order],
                ))
                for order in (1, -1)
            ])
            assert best_distance <= TOLERANCE
        else:
            # Heavy atom has 3 hydrogen atoms
            # -> there is rotational freedom
            # -> invalid test case
            raise ValueError("Invalid test case")


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


def test_atom_mask():
    """
    Test atom mask usage by hydrogenating in two steps:
    Firstly one random half is masked, secondly the other half is
    masked.
    After reordering the atoms in the standard way, the result should be
    equal to hydrogenating in a single step.
    """
    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), "1l2y.mmtf"))
    atoms = mmtf.get_structure(
        mmtf_file, model=1, include_bonds=True, extra_fields=["charge"]
    )
    heavy_atoms = atoms[atoms.element != "H"]

    ref_atoms, _ = hydride.add_hydrogen(heavy_atoms)
    ref_atoms = ref_atoms[info.standardize_order(ref_atoms)]

    # Hydrogenate the first random half of the molecule
    random_mask = np.random.choice([False, True], heavy_atoms.array_length())
    half_hydrogenated, orig_mask = hydride.add_hydrogen(
        heavy_atoms, random_mask
    )
    # Hydrogenate the second half by inverting mask
    # Special handling due to additional hydrogen atoms
    # after first hydrogenation
    inv_random_mask = np.zeros(half_hydrogenated.array_length(), bool)
    inv_random_mask[orig_mask] = ~random_mask
    test_atoms, _ = hydride.add_hydrogen(
        half_hydrogenated, inv_random_mask
    )
    test_atoms = test_atoms[info.standardize_order(test_atoms)]

    assert test_atoms == ref_atoms


@pytest.mark.parametrize("fill_value", [False, True])
def test_atom_mask_extreme_case(fill_value):
    """
    Check whether the input atom mask works properly by testing the
    cases, where no atom an each atom is masked, respectively.
    """
    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), "1l2y.mmtf"))
    atoms = mmtf.get_structure(
        mmtf_file, model=1, include_bonds=True, extra_fields=["charge"]
    )
    heavy_atoms = atoms[atoms.element != "H"]
    mask = np.full(heavy_atoms.array_length(), fill_value, dtype=bool)

    if fill_value is True:
        # All hydrogen atom are added
        ref_atoms,  ref_orig_mask  = hydride.add_hydrogen(heavy_atoms)
        test_atoms, test_orig_mask = hydride.add_hydrogen(heavy_atoms, mask)
        assert test_atoms == ref_atoms
        assert np.array_equal(test_orig_mask, ref_orig_mask)
    else:
        # No hydrogen atoms are added
        test_atoms, test_orig_mask = hydride.add_hydrogen(heavy_atoms, mask)
        assert test_atoms == heavy_atoms
        assert (test_orig_mask == True).all()


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