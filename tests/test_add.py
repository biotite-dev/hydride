# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pytest
import numpy as np
from numpy.linalg import norm
from numpy import random
from biotite.structure import rotate_centered, rotate, rotate_about_axis
import biotite.structure.io.mmtf as mmtf
from biotite.structure.info import residue
from hydride import add_hydrogen
from .util import struc_for_test_dir


@pytest.mark.parametrize("molecule_id", [
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
    "CN", # Hydrogen cyanide
    "11X" # N-pyridin-3-ylmethylaniline
])
def test_hyd_position(molecule_id):
    """
    Load test molecules from the Chemical Components Dictionary (CCD),
    subject AtomArray to rotation and translation by a
    randomly chosen degree respectively distance, extract hydrogen
    coordinates and remove hydrogen.
    Then, the `add_hydrogen` function is applied and it is verified
    whether the coordinates of the newly added hydrogen atoms are in
    accordance with the original coordinates within an appropriate
    tolerance range.
    Rotation and translation of the primary AtomArray from the CCD by
    randomly chosen values is performed for two reasons:
    On the one hand, the AtomArrays from the CCD themselves serve as
    reference for hydrogen addition. Hence, AtomArrays loaded from the
    CCD without any further spatial modification possess exactly the
    same coordinates as the reference array, which is regularly not the
    case with residues the user obtained e. g. from a molecular dynamics
    simulation or x-ray cristallography. This implies that the
    implementation's capacity to rotate the two molecules once they are
    centered such that congruence is achieved between them to a
    satisfying extent is crucial. Therefore, the need for congruence by
    rotation is artificially created.
    On the other hand, a fragment consisting of at least three atoms is
    required in order to definitely determine the position of hydrogen
    to add as three points create a plane and a fragment consisting
    e. g. of only two atoms leaves a rotational degree of freedom that
    makes it impossible to reproduce the original positions of the
    hydrogen atoms. In biomolecules, this scenario is commonly
    encountered in terminal amino groups or methyl groups or terminal
    groups in general. Proper conformation of hydrogen bound to terminal
    groups with rotational freedom however is tested in the test
    belonging to the relaxation function.
    For this reason, only molecules enabling the construction of
    fragments with at least three atoms such as aromatic or cyclic
    compounds are chosen in this test.
    """
    molecule = residue(molecule_id)
    # Perform translation of the molecule along the three axes
    translation = random.uniform(-100, 100, 3)
    molecule.coord = molecule.coord + np.array(
        [translation[0], translation[1], translation[2]]
    )
    # Perform rotation around x, y and z axis
    angles = random.uniform(0, 2 * np.pi, 3)
    molecule = rotate(
        molecule, np.array([angles[0], angles[1], angles[2]])
    )
    molecule_hyd_coord = molecule.coord[molecule.element == "H"]
    molecule_without_hyd = molecule[molecule.element != "H"]
    hyd_indices = np.where(molecule.element == "H")[0]
    start_index = hyd_indices[0]
    end_index = hyd_indices[len(hyd_indices) - 1] + 1
    molecule_with_newly_added_hyd = add_hydrogen(molecule_without_hyd)
    added_hyd_atoms = molecule_with_newly_added_hyd[
        start_index:end_index
    ]
    assert added_hyd_atoms.coord == pytest.approx(
        molecule_hyd_coord, abs=1e-3
    )


@pytest.mark.parametrize("molecule_id", [
    "CH3", # Methane
    "EHN", # Ethane
    "TME" # Propane
])
def test_rotational_freedom(molecule_id):
    """
    This test's purpose is to prove the thoughts elaborated in the
    DocString of the previous test, namely that definite identification
    of the position of hydrogen atoms to add is not possible if the
    constructed fragment allows rotational freedom, i. e. consists of
    less than three atoms.
    For this purpose, molecules with possess terminal groups or are made
    up of less than three non-hydrogen atoms are chosen for this test.
    First, is is proven that reproduction of the hydrogen coordinates is
    indeed possible if the molecule does not undergo spatial
    modification, but is not possible if the molecule is subjected to
    spatial modification (rotation and translation).
    """
    molecule = residue(molecule_id)
    molecule_hyd_coord = molecule.coord[molecule.element == "H"]
    molecule_without_hyd = molecule[molecule.element != "H"]
    hyd_indices = np.where(molecule.element == "H")[0]
    start_index = hyd_indices[0]
    end_index = hyd_indices[len(hyd_indices) - 1] + 1
    molecule_with_newly_added_hyd = add_hydrogen(molecule_without_hyd)
    added_hyd_atoms = molecule_with_newly_added_hyd[
        start_index:end_index
    ]
    assert added_hyd_atoms.coord == pytest.approx(
        molecule_hyd_coord, abs=1e-3
    )
    # Now subjecting molecule to spatial modification
    # Perform translation of the molecule along the three axes
    translation = random.uniform(-100, 100, 3)
    molecule.coord = molecule.coord + np.array(
        [translation[0], translation[1], translation[2]]
    )
    # Perform rotation around x, y and z axis
    angles = random.uniform(0, 2 * np.pi, 3)
    molecule = rotate(
        molecule, np.array([angles[0], angles[1], angles[2]])
    )
    molecule_hyd_coord = molecule.coord[molecule.element == "H"]
    molecule_without_hyd = molecule[molecule.element != "H"]
    molecule_with_newly_added_hyd = add_hydrogen(molecule_without_hyd)
    added_hyd_atoms = molecule_with_newly_added_hyd[
        start_index:end_index
    ]
    assert added_hyd_atoms.coord != pytest.approx(
        molecule_hyd_coord, abs=1e-3
    )


def test_terminal_double_bond():
    """
    This test's purpose is to verify whether the hydrogen atoms added to
    a terminal double bond, as it is encountered in the amino acid
    arginine's residue (cf. guanidino group), are located within the
    sigma bond plane.
    The necessity to explicitly test this arises from the fact that
    fragments consisting of less than three atoms entail rotational
    freedom during the process of superimposition, as explained above,
    enabling the addition of hydrogen at coordinates not in accordance
    with the orbital model according to which a double bond is made up
    of two atoms with sp2 hybridisation; the three sp2 hybrid orbitals
    of one atom, which are responsible for the formation of sigma bonds,
    form an equilateral triangle with angles of 120° between them and
    are thus located within one plane.
    The test is performed with the amino acid arginine.
    """
    arginine = residue("ARG")
    # Subjecting the molecule to translation and rotation
    # Perform translation of the molecule along the three axes
    translation = random.uniform(-100, 100, 3)
    arginine.coord = arginine.coord + np.array(
        [translation[0], translation[1], translation[2]]
    )
    # Perform rotation around x, y and z axis
    angles = random.uniform(0, 2 * np.pi, 3)
    arginine = rotate(
        arginine, np.array([angles[0], angles[1], angles[2]])
    )
    # The hydrogen atoms of interest, i. e. the hydrogen atoms bound to
    # terminal nitrogen atoms, are located at the indices 22 to 25
    # As delocalisation across the whole guanidino group is given due to
    # the charged nitogen atom with double bond, the hydrogen atoms
    # bound to the nitrogen atom with single bond are considered as well
    guanidino_hyd_coord = arginine.coord[22:26]
    arginine_without_hyd = arginine[arginine.element != "H"]
    arginine_with_newly_added_hyd = add_hydrogen(arginine_without_hyd)
    added_guanidino_hyd_atoms = arginine_with_newly_added_hyd[22:26]
    added_guanidino_hyd_coords = added_guanidino_hyd_atoms.coord
    assert added_guanidino_hyd_coords == pytest.approx(
        guanidino_hyd_coord, abs=1e-5
    )


def test_no_entry_in_ccd():
    """
    This test's purpose is to verify whether the case of a residue which
    does not possess an entry in the CCD raises the expected
    UserWarning.
    Therefore, urea is retrieved from the CCD and afterwards, its
    residue name is changed to the non-existent residue name 'XYX'.
    Moreover, it is verified that residues not possessing an entry in
    the CCD still are added to the output AtomArray.
    """
    warning_message_entry_not_available = (
        "Entries in the Chemical Components Dictionary are not "
        "available for the following residues: \n"
        "XYX. \n"
        "Therefore, hydrogen addition for those residues is omitted."
    )
    ref_urea = residue("URE")
    ref_urea.res_name = np.array(["XYX"] * ref_urea.array_length())
    with pytest.warns(None) as record:
        test_urea = add_hydrogen(ref_urea)
    assert len(record) == 1
    warning = record[0]
    assert issubclass(warning.category, UserWarning)
    assert str(warning.message) == warning_message_entry_not_available
    assert ref_urea == test_urea


def test_hydrogen_at_peptide_bond():
    """
    This test's purpose is to verify that hydrogen atoms belonging to
    the nitrogen atom of an amino group involved in a peptide bond are
    placed correctly within a certain tolerance range. The necessity for
    this test arises from the fact that hydrogen at a peptide bond has a
    different bond angle than hydrogen of a free amino group.
    For this reason, the mini protein 1l2y consisting of 20 residues is
    used.
    """
    # The mmtf format already contains bond information
    mmtf_file = mmtf.MMTFFile.read(
        struc_for_test_dir("1l2y.mmtf")
    )
    mini_protein = mmtf.get_structure(
        mmtf_file, model=1, include_bonds=True
    )
    # The step of subjecting the protein to translation and rotation is
    # omitted as in proteins the individual residues generally do not
    # possess the same coordinates as the reference residues
    # Hydrogen at peptide bonds, i. e. bound to amino groups, possess
    # the atom name 'H'
    hyd_at_peptide_bonds = mini_protein[mini_protein.atom_name == "H"]
    hyd_at_peptide_coord = hyd_at_peptide_bonds.coord
    mini_protein_without_hyd = mini_protein[mini_protein.element != "H"]
    mini_protein_with_added_hyd = add_hydrogen(mini_protein_without_hyd)
    newly_added_hyd_at_peptide_bonds = mini_protein_with_added_hyd[
        mini_protein_with_added_hyd.atom_name == "H"
    ]
    # The first hydrogen atom must be removed as it belongs to the amino
    # terminus and is therefore not involved in a peptide bond
    # In case of the mini protein's original hydrogen atoms, this step
    # is omitted as the hydrogen atoms belonging to the amino terminus
    # both possess numbers in their names, making them distinguishable
    # from hydrogen atoms of amino groups involved in peptide bonds
    newly_added_hyd_at_peptide_bonds = newly_added_hyd_at_peptide_bonds[
        1:
    ]
    newly_added_hyd_at_peptide_coord = \
        newly_added_hyd_at_peptide_bonds.coord
    deviations = np.array([])
    for i in range(hyd_at_peptide_coord.shape[0]):
        original_pos = hyd_at_peptide_coord[i]
        added_pos = newly_added_hyd_at_peptide_coord[i]
        distance = norm(original_pos - added_pos)
        deviations = np.append(deviations, distance)
    mean_deviation = np.sum(deviations) / deviations.shape[0]
    largest_deviation = np.amax(deviations)
    assert mean_deviation == pytest.approx(6e-2, abs=1e-2)
    assert largest_deviation <= 2e-1


def test_missing_heavy_atoms():
    """
    This test's purpose is to verify that hydrogen-carrying heavy atoms
    occurring in the reference array but not in the subject array do not
    cause erroneous addition of hydrogen atoms.
    To be more precise, this test's aim is to ensure that above named
    scenario does not lead to the addition of wrong hydrogen atoms,
    i. e. hydrogen atoms that are not supposed to appear in the subject
    due to the missing heavy atom they are bound to.
    Verification is performed by regarding the coordinates of hydrogen
    atoms.
    Benzene is chosen as test molecule. Hydrogen addition is correct
    even at the carbon atoms neighbouring the removed carbon atom as
    the bond type is BondType.AROMATIC, preventing that the rotational
    freedom terminal groups usually entail leads to erroneous results.
    """
    benzene = residue("BNZ")
    translation = random.uniform(-100, 100, 3)
    benzene.coord = benzene.coord + np.array(
        [translation[0], translation[1], translation[2]]
    )
    angles = random.uniform(0, 2 * np.pi, 3)
    benzene = rotate(
        benzene, np.array([angles[0], angles[1], angles[2]])
    )
    # One carbon atom occurring before the last atom must be removed in
    # order to enable effects on the following hydrogen addition
    # The first carbon atom of benzene is removed in order to enable
    # effects on the largest possible amount of subsequent hydrogen
    # addition
    benzene = benzene[benzene.atom_name != "C1"]
    # Hydrogen bound to the first carbon ('H1') must be removed as well
    benzene = benzene[benzene.atom_name != "H1"]
    ref_hyd_coord = benzene[benzene.element == "H"].coord
    benzene_without_hyd = benzene[benzene.element != "H"]
    benzene_with_added_hyd = add_hydrogen(benzene_without_hyd)
    test_hyd_coord = benzene_with_added_hyd[
        benzene_with_added_hyd.element == "H"
    ].coord
    assert test_hyd_coord == pytest.approx(ref_hyd_coord, abs=1e-3)


def test_adoption_of_optional_annotations():
    """
    This test's purpose is to verify that optional annotation categories
    present in the input array are adopted for the output array.
    Therefore, is is taken advantage of the fact that molecules
    retrieved from the CCD already posses a charge annotation. Benzene
    is chosen as test molecule. Moreover, a occupancy annotation with
    default values (1) is added to the AtomArray.
    """
    ref_benzene = residue("BNZ")
    ref_benzene.set_annotation(
        "occupancy", np.array([1] * ref_benzene.array_length())
    )
    ref_benzene_without_hyd = ref_benzene[ref_benzene.element != "H"]
    test_benzene = add_hydrogen(ref_benzene_without_hyd)
    anno_cats = ref_benzene.get_annotation_categories()
    for category in anno_cats:
        assert np.all(
            ref_benzene.get_annotation(category) == \
                test_benzene.get_annotation(category)
        )


def test_addition_of_hyd_to_bond_list():
    """
    This test's purpose is to verify that the bonds between heavy atoms
    and the newly added hydrogen atoms also appear in the BondList of
    the output array.
    Benzene is chosen as test molecule. Again, it is taken advantage of
    the fact that AtomArrays retrieved from the CCD already possess an
    associated BondList.
    """
    ref_benzene = residue("BNZ")
    ref_bond_list = ref_benzene.bonds
    benzene_without_hyd = ref_benzene[ref_benzene.element != "H"]
    test_benzene = add_hydrogen(benzene_without_hyd)
    test_bond_list = test_benzene.bonds
    assert ref_bond_list == test_bond_list


def test_order_differing_from_ccd():
    """
    This test's purpose is to verify that an atom order differing from
    that in the Chemical Components Dictionary does not lead to
    erroneous results. The amino acid lysine is chosen as test molecule.
    In order to achieve a deviation of the atom order to that of the
    CCD, lysine is first retrieved from the CCD and subsequently, the
    atom order is pseudorandomly shuffled. Rotation and translation of
    the molecule are performed in order to prevent that the coordinates
    of the hydrogen atoms constructed by the function are identical with
    the initial hydrogen coordinates because superimposition effectively
    does not take place and the reference coordinates are directly
    adapted. Moreover, in addition to translation and rotation of the
    molecule as a whole, the dihedral angles between the carbon atoms of
    lysine's side chain are pseudorandomly altered as well, which has
    the following background: If merely the molecule as a whole is
    spatially modified, i. e. translated and rotated, always translation
    and rotation by the same amount is required in order to achieve
    congruence between a specific pair of fragments (fixed and mobile
    fragment). Thus, although the algorithm internally assigns a
    fragment pair to wrong hydrogen atoms, the hydrogen atoms are added
    at the correct positions. In order to evade this artefact, internal
    spatial modification of the molecule is required as well so that
    translation and rotation by different amounts is required in order
    to achieve congruence between different fragment pairs and a wrong
    assignment of a fragment pair to hydrogen atoms becomes evident.
    """
    lysine = residue("LYS")
    lysine_coord_1 = lysine.coord[lysine.element == "H"]
    translation = random.uniform(-100, 100, 3)
    lysine.coord = lysine.coord + np.array(
        [translation[0], translation[1], translation[2]]
    )
    angles = random.uniform(0, 2 * np.pi, 3)
    lysine = rotate(
        lysine, np.array([angles[0], angles[1], angles[2]])
    )
    lysine_coord_2 = lysine.coord[lysine.element == "H"]
    assert lysine_coord_1 != pytest.approx(lysine_coord_2, abs=1e-1)
    # Creating list containing 'packages' of heavy atoms in the side
    # chain with the respective hydrogen atoms bound to them
    side_chain_list = [
        [4,13,14], [5,15,16], [6,17,18], [7,19,20], [8,21,22,23]
    ]
    # Randomly rotating bonds between the carbon atoms of the residue in
    # a loop
    # All atoms behind the considered carbon atom, i. e. the rest of the
    # residue, are rotated by the respective angle as well in order to
    # provide stereochemical consistency
    for i in range(len(side_chain_list)):
        # Stop the loop as soon as the amino group of the residue is
        # reached
        if i == 4:
            break
        # Identifying rotation axis
        carbon_index = side_chain_list[i][0]
        bonds, _ = lysine.bonds.get_bonds(carbon_index)
        bond_elements = lysine.element[bonds]
        # Filter carbon binding partners
        bonds = bonds[bond_elements == "C"]
        # Preceding carbon atoms always have a lower index than
        # successive ones
        preceding_carbon_index = np.amin(bonds)
        considered_carb_coord = lysine[carbon_index].coord
        preceding_carb_coord = lysine[preceding_carbon_index].coord
        axis = considered_carb_coord - preceding_carb_coord
        support = considered_carb_coord
        # Fusioning individual packages for collective rotation
        collective_indices = []
        for j in range(i, len(side_chain_list)):
            collective_indices += side_chain_list[j]
        dihedral_angle = random.uniform(0, 2 * np.pi)
        lysine.coord[collective_indices] = rotate_about_axis(
            lysine.coord[collective_indices], axis, dihedral_angle,
            support
        )
    lysine_coord_3 = lysine.coord
    assert lysine_coord_3 != pytest.approx(lysine_coord_2, abs=1e-1)
    # Only the atoms whose coordinates are affected by the
    # conformational change are considered for the verification
    affected_ref_hyd_coord = lysine.coord[
        (lysine.element == "H")
        &
        ~np.isin(
            lysine.atom_name,
            ["H", "H2", "HA", "HZ1", "HZ2", "HZ3", "HXT"]
        )
    ]
    lysine = lysine[lysine.element != "H"]
    lys_indices = np.arange(lysine.shape[0])
    random.shuffle(lys_indices)
    shuffled_lysine = lysine[lys_indices]
    shuffled_lysine_with_hyd = add_hydrogen(shuffled_lysine)
    affected_test_hyd_coord = shuffled_lysine_with_hyd.coord[
        (shuffled_lysine_with_hyd.element == "H")
        &
        ~np.isin(
            shuffled_lysine_with_hyd.atom_name,
            ["H", "H2", "HA", "HZ1", "HZ2", "HZ3", "HXT"]
        )
    ]
    # Hydrogen atoms are added in the same sequence as in the CCD so
    # that further steps for comparison of the coordinates are not
    # necessary
    assert affected_ref_hyd_coord == pytest.approx(
        affected_test_hyd_coord, abs=1e-3
    )