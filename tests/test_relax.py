# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import pytest
import numpy as np
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.mmtf as mmtf
import hydride
from hydride.relax import _find_rotatable_bonds
from .util import data_dir


@pytest.mark.parametrize("seed", np.arange(10))
def test_staggered(seed):
    """
    :func:`relax_hydrogen()` should be able to restore a staggered
    conformation of ethane from any other conformation.
    """
    # Construct ethane in staggered conformation
    ethane = struc.AtomArray(8)
    ethane.element = np.array(["C", "C", "H", "H", "H", "H", "H", "H"])
    ethane.coord = np.array([
        [-0.756,  0.000,  0.000],
        [ 0.756,  0.000,  0.000],
        [-1.140,  0.659,  0.7845],
        [-1.140,  0.350, -0.9626],
        [-1.140, -1.009,  0.1781],
        [ 1.140, -0.350,  0.9626],
        [ 1.140,  1.009, -0.1781],
        [ 1.140, -0.659, -0.7845],
    ])
    ethane.bonds = struc.BondList(
        8,
        np.array([
            [0, 1, 1],
            [0, 2, 1],
            [0, 3, 1],
            [0, 4, 1],
            [1, 5, 1],
            [1, 6, 1],
            [1, 7, 1],
        ])
    )

    # Check if created ethane is in optimal staggered conformation
    # -> Dihedral angle of 60 degrees
    dihed = struc.dihedral(ethane[2], ethane[0], ethane[1], ethane[5])
    assert np.rad2deg(dihed) % 120 == pytest.approx(60, abs=1)

    # Move the ethane molecule away
    # from the optimal staggered conformation
    np.random.seed(seed)
    angle = np.random.rand() * 2 * np.pi
    ethane.coord[5:] = struc.rotate_about_axis(
        ethane.coord[5:],
        angle = angle,
        axis = ethane.coord[1] - ethane.coord[0],
        support = ethane.coord[0]
    )

    # Check if new conformation ethane is not staggered anymore
    dihed = struc.dihedral(ethane[2], ethane[0], ethane[1], ethane[5])
    assert np.rad2deg(dihed) % 120 != pytest.approx(60, abs=1)

    # Try to restore staggered conformation via relax_hydrogen()
    ethane.coord = hydride.relax_hydrogen(
        ethane,
        # The angle increment must be smaller
        # than the expected accuracy (abs=1)
        angle_increment = np.deg2rad(0.5),
        # High number of iterations due to small increment
        iterations = 1000
    )

    # Check if staggered conformation is restored
    dihed = struc.dihedral(ethane[2], ethane[0], ethane[1], ethane[5])
    assert np.rad2deg(dihed) % 120 == pytest.approx(60, abs=1)


def test_hydrogen_bonds():
    """
    Check whether the relaxation algorithm is able to restore most of
    the original hydrogen bonds.
    The number of bonds found without relaxation is handled as baseline.
    The residues at the biotin binding pocket of streptavidin (including
    biotin itself) are used as test case.
    """
    PERCENTAGE = 1.0
    RES_IDS = (27, 43, 45, 47, 90, 300)

    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), "2rtg.mmtf"))
    atoms = mmtf.get_structure(
        mmtf_file, model=1, include_bonds=True, extra_fields=["charge"]
    )
    atoms = atoms[atoms.chain_id == "B"]
    #biotin = atoms[atoms.res_name == "BTN"]
    mask = np.isin(atoms.res_id, RES_IDS)
    ref_num = len(struc.hbond(atoms, mask, mask))
    
    atoms = atoms[atoms.element != "H"]
    atoms, _ = hydride.add_hydrogen(atoms)
    mask = np.isin(atoms.res_id, RES_IDS)
    base_num = len(struc.hbond(atoms, mask, mask))

    atoms.coord = hydride.relax_hydrogen(atoms, iterations=100)
    mask = np.isin(atoms.res_id, RES_IDS)
    test_num = len(struc.hbond(atoms, mask, mask))

    if base_num == ref_num:
        ValueError(
            "Invalid test case, "
            "no further hydrogen bonds can be found via relaxation"
        )
    assert (test_num - base_num) / (ref_num - base_num) >= PERCENTAGE


@pytest.mark.parametrize(
    "res_name, ref_bonds",
    [   
        # Fructopyranose
        ("FRU", [
            ( "O1",  "C1",  True,  ("HO1",)),
            ( "O2",  "C2",  True,  ("HO2",)),
            ( "O3",  "C3",  True,  ("HO3",)),
            ( "O4",  "C4",  True,  ("HO4",)),
            ( "O6",  "C6",  True,  ("HO6",)),
        ]),
        # Arginine with positive side chain
        ("ARG", [
            (  "N",  "CA",  True,  ("H", "H2")),
            ("OXT",   "C",  True,  ("HXT",)),
        ]),
        # Isoleucine
        ("ILE", [
            (  "N",  "CA",  True,  ("H", "H2")),
            ("OXT",   "C",  True,  ("HXT",)),
            ("CG2",  "CB",  True,  ("HG21", "HG22", "HG23")),
            ("CD1", "CG1",  True,  ("HD11", "HD12", "HD13")),
        ]),
        # 1-phenylguanidine
        ("PL0", [
            ( "N3",  "C7",  False, ("HN3",)),
        ]),
        # Water
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