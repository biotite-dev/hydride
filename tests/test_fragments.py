# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import biotite.structure as struc
import biotite.structure.info as info
import numpy as np
import pytest
import hydride


@pytest.mark.parametrize(
    "res_name",
    [
        "BNZ",  # Benzene
        "BZF",  # Benzofuran
        "IND",  # indole
        "PZO",  # Pyrazole
        "BZI",  # Benzimidazole
        "LOM",  # Thiazole
        "P1R",  # Pyrimidine
        "ISQ",  # Isoquinoline
        "NPY",  # Naphthalene
        "AN3",  # Anthracene
        "0PY",  # Pyridine
        "4FT",  # Phthalazine
        "URA",  # Uracil
        "CHX",  # Cyclohexane
        "CEJ",  # 1,3-Cyclopentanedione
        "CN",  # Hydrogen cyanide
        "11X",  # N-pyridin-3-ylmethylaniline
        "ANL",  # Aniline
    ],
)
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
    for i, hydrogen_coord in enumerate(test_hydrogen_coord):
        try:
            # The coord for all hydrogens bonded to the same heavy atom
            if len(hydrogen_coord) == 0:
                # No bonded hydrogen atoms -> nothing to compare
                continue
            elif len(hydrogen_coord) == 1:
                # Only a single hydrogen atom
                # -> unambiguous assignment to reference hydrogen coord
                assert (
                    np.max(
                        struc.distance(
                            hydrogen_coord,
                            ref_hydrogen_coord[ref_index : ref_index + 1],
                        )
                    )
                    <= TOLERANCE
                )
                ref_index += 1
            elif len(hydrogen_coord) == 2:
                # Heavy atom has 2 hydrogen atoms
                # -> Since the hydrogen atoms are indistinguishable,
                # there are two possible assignment to reference
                # hydrogen atoms
                best_distance = min(
                    [
                        np.max(
                            struc.distance(
                                hydrogen_coord,
                                ref_hydrogen_coord[ref_index : ref_index + 2][::order],
                            )
                        )
                        for order in (1, -1)
                    ]
                )
                assert best_distance <= TOLERANCE
                ref_index += 2
            else:
                # Heavy atom has 3 hydrogen atoms
                # -> there is rotational freedom
                # -> invalid test case
                raise ValueError("Invalid test case")
        except AssertionError:
            print(f"Failing central atom: {heavy_atoms.atom_name[i]}")
            raise


def test_missing_fragment():
    """
    If a molecule contains an unknown fragment, check if a warning is
    raised and all other atoms are still hydrogenated.
    """
    TOLERANCE = 0.1

    lib = hydride.FragmentLibrary.standard_library()

    ref_mol = info.residue("BZI")  # Benzimidazole
    # It should not be possible to have a nitrogen at this position
    # with a positive charge
    ref_mol.charge[0] = 1
    ref_hydrogen_coord = ref_mol.coord[ref_mol.element == "H"]

    test_mol = test_mol = ref_mol[ref_mol.element != "H"]
    with pytest.warns(
        UserWarning, match="Missing fragment for atom 'N1' at position 0"
    ):
        hydrogen_coord = lib.calculate_hydrogen_coord(test_mol)
    print(hydrogen_coord)
    flattend_coord = []
    for coord in hydrogen_coord:
        flattend_coord += coord.tolist()
    test_hydrogen_coord = np.array(flattend_coord)

    assert (
        np.max(
            struc.distance(
                test_hydrogen_coord,
                # Expect missing first hydrogen due to missing fragment
                ref_hydrogen_coord[1:],
            )
        )
        <= TOLERANCE
    )


@pytest.mark.parametrize(
    "lib_enantiomer, subject_enantiomer",
    itertools.product(
        # L-alanine and D-alanine
        ["ALA", "DAL"],
        ["ALA", "DAL"],
    ),
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
    # As the test case is constructed with the exact same molecule
    # can be in the library, move the molecule to assure that
    # correct hydrogen position calculation is not an artifact
    np.random.seed(0)
    test_model = struc.rotate(test_model, np.random.rand(3))
    test_model = struc.translate(test_model, np.random.rand(3))
    test_model, _ = hydride.add_hydrogen(test_model, fragment_library=lib)

    test_model, _ = struc.superimpose(
        ref_model, test_model, atom_mask=(test_model.element != "H")
    )
    ref_stereo_h_coord = ref_model.coord[ref_model.atom_name == "HA"][0]
    test_stereo_h_coord = test_model.coord[test_model.atom_name == "HA"][0]

    assert struc.distance(test_stereo_h_coord, ref_stereo_h_coord) <= TOLERANCE


@pytest.mark.parametrize(
    "res_name, nitrogen_index",
    [
        ("ARG", 7),  # Arginine NE
        ("ARG", 9),  # Arginine NH1
        ("ARG", 10),  # Arginine NH2
        ("PRO", 0),  # Proline N
        ("ASN", 7),  # Asparagine N
        ("T", 15),  # Thymidine N3
    ],
)
def test_partial_double_bonds(res_name, nitrogen_index):
    """
    It is difficult to assign hydrogen atoms to nitrogen properly,
    as the the geometry can be trigonal planar instead of tetrahedral,
    due to partial double bonds.

    This test checks whether the correct geomtry is found for selected
    nitrogen atoms in chosen examples.
    The chosen examples have no intersect with
    :func:`test_hydrogen_positions()`, as those molecules were already
    tested.

    As in :func:`test_hydrogen_positions()`, it is important that the
    respective nitrogen atom has no rotational freedom.
    """
    TOLERANCE = 0.1

    library = hydride.FragmentLibrary.standard_library()

    molecule = info.residue(res_name)
    if molecule.element[nitrogen_index] != "N":
        raise ValueError("Invalid test case")

    np.random.seed(0)
    molecule = struc.rotate(molecule, np.random.rand(3))
    molecule = struc.translate(molecule, np.random.rand(3))

    bond_indices, _ = molecule.bonds.get_bonds(nitrogen_index)
    bond_h_indices = bond_indices[molecule.element[bond_indices] == "H"]
    ref_hydrogen_coord = molecule.coord[bond_h_indices]

    molecule = molecule[molecule.element != "H"]
    hydrogen_coord = library.calculate_hydrogen_coord(molecule)
    test_hydrogen_coord = hydrogen_coord[nitrogen_index]

    # The coord for all hydrogens bonded to the same heavy atom
    if len(test_hydrogen_coord) == 0:
        # No bonded hydrogen atoms -> nothing to compare
        raise ValueError("Invalid test case")
    elif len(test_hydrogen_coord) == 1:
        # Only a single hydrogen atom
        # -> unambiguous assignment to reference hydrogen coord
        assert (
            np.max(struc.distance(test_hydrogen_coord, ref_hydrogen_coord)) <= TOLERANCE
        )
    elif len(test_hydrogen_coord) == 2:
        # Heavy atom has 2 hydrogen atoms
        # -> Since the hydrogen atoms are indistinguishable,
        # there are two possible assignment to reference
        # hydrogen atoms
        best_distance = min(
            [
                np.max(struc.distance(test_hydrogen_coord, ref_hydrogen_coord[::order]))
                for order in (1, -1)
            ]
        )
        assert best_distance <= TOLERANCE
    else:
        # Heavy atom has 3 hydrogen atoms
        # -> there is rotational freedom
        # -> invalid test case
        raise ValueError("Invalid test case")


def test_undefined_bond_type():
    """
    Check if the :class:`FragmentLibrary` raises a warning, if atoms
    are bonded with :attr:`BondType.ANY`, and if the corresponding
    fragments are ignored.
    """
    molecule = info.residue("BNZ")
    # Retain the bonds, but remove the bond type
    molecule.bonds = struc.BondList(
        molecule.array_length(),
        # Remove bond type from existing bonds
        molecule.bonds.as_array()[:, :2],
    )

    # Test handing of 'BondType.ANY' in 'add_molecule()'
    library = hydride.FragmentLibrary()
    with pytest.warns(UserWarning) as record:
        library.add_molecule(molecule)
    assert any(
        [
            True if "undefined bond type" in warning.message.args[0] else False
            for warning in record
        ]
    )
    # No fragment should be added
    assert len(library._frag_dict) == 0

    # Test handing of 'BondType.ANY' in 'calculate_hydrogen_coord()'
    heavy_atoms = molecule[molecule.element != "H"]
    library = hydride.FragmentLibrary.standard_library()
    with pytest.warns(UserWarning) as record:
        hydrogen_coord = library.calculate_hydrogen_coord(heavy_atoms)
    assert any(
        [
            True if "undefined bond type" in warning.message.args[0] else False
            for warning in record
        ]
    )
    assert len(hydrogen_coord) == heavy_atoms.array_length()
    for coord in hydrogen_coord:
        # For each heavy atom there should be no added hydrogen
        assert len(coord) == 0
