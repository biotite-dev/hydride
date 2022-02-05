# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import shutil
import tempfile
import itertools
import os
from os.path import join, splitext
import subprocess
import pytest
import numpy as np
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.info as info
import biotite.structure.io.mmtf as mmtf
import biotite.structure.io.mol as mol
import hydride
from hydride.cli import main as run_cli
from .util import data_dir


PDB_ID = "1l2y"
PH = 7.0


@pytest.fixture(scope="module", params=["pdb", "pdbx", "cif", "mmtf"])
def input_file(request):
    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), f"{PDB_ID}.mmtf"))
    model = mmtf.get_structure(
        mmtf_file, model=1, include_bonds=True, extra_fields=["charge"]
    )
    model = model[model.element != "H"]
    temp_file = tempfile.NamedTemporaryFile(
        "w", suffix=f".{request.param}", delete=False
    )
    strucio.save_structure(temp_file.name, model)
    temp_file.close()

    yield temp_file.name

    os.remove(temp_file.name)


@pytest.fixture(params=["pdb", "pdbx", "cif", "mmtf"])
def output_file(request):
    temp_file = tempfile.NamedTemporaryFile(
        "r", suffix=f".{request.param}", delete=False
    )
    temp_file.close()

    yield temp_file.name

    os.remove(temp_file.name)


@pytest.fixture()
def dummy_output_file():
    """
    This file should never be written, as the CLI run should fail
    before.
    """
    temp_file = tempfile.NamedTemporaryFile(
        "r", suffix=f".pdb", delete=False
    )

    yield temp_file.name

    temp_file.close()
    os.remove(temp_file.name)


def assert_hydrogen_addition(test_file):
    """
    Test that all hydrogen atoms were added with the correct name.

    The hydrogen positions are not tested since the other test modules
    already focus on this task.
    """
    mmtf_file = mmtf.MMTFFile.read(join(data_dir(), f"{PDB_ID}.mmtf"))
    ref_model = mmtf.get_structure(mmtf_file, model=1, include_bonds=True)
    ref_model.charge = hydride.estimate_amino_acid_charges(ref_model, PH)
    # Both naming conventions 'H' and 'H1' are used in PDB
    # However, AtomNameLibrary uses 'H'; it is changed for consistency
    ref_model.atom_name[ref_model.atom_name == "H1"] = "H"

    test_model = strucio.load_structure(test_file)

    assert test_model.array_length() == ref_model.array_length()
    assert test_model.atom_name.tolist() == ref_model.atom_name.tolist()




def test_simple(input_file, output_file):
    """
    Test CLI run without optional parameters.
    """
    run_cli([
        "-v",
        "-i", input_file,
        "-o", output_file,
        "-c", str(PH)
    ])

    assert_hydrogen_addition(output_file)


def test_molfile():
    """
    Test usage of MOL/SDF files for input and output.
    """
    mol_file = mol.MOLFile.read(join(data_dir(), "TYR.sdf"))
    ref_model = mol_file.get_structure()
    model = ref_model[ref_model.element != "H"]

    input_file = tempfile.NamedTemporaryFile(
        "w", suffix=".mol", delete=False
    )
    strucio.save_structure(input_file.name, model)
    input_file.close()

    output_file = tempfile.NamedTemporaryFile(
        "r", suffix=".mol", delete=False
    )
    output_file.close()

    run_cli([
        "-v",
        "-i", input_file.name,
        "-o", output_file.name,
    ])

    mol_file = mol.MOLFile.read(output_file.name)
    test_model = mol_file.get_structure()

    os.remove(input_file.name)
    os.remove(output_file.name)

    assert test_model.array_length() == ref_model.array_length()
    assert test_model.element.tolist() == ref_model.element.tolist()


def test_std_in_out(input_file, output_file):
    """
    Test CLI run with input from STDIN and output to STDOUT.
    """
    in_format  = splitext(input_file )[-1][1:]
    out_format = splitext(output_file)[-1][1:]
    with open(input_file, "rb") as file:
        input_file_content = file.read()

    # Use subprocess instead of API function call
    # to be able to use STDIN/STDOUT
    completed_process = subprocess.run(
        [
            "hydride",
            "-v",
            "-I", in_format,
            "-O", out_format,
            "-c", str(PH)
        ],
        input=input_file_content,
        capture_output=True
    )
    stdout = completed_process.stderr.decode("UTF-8")
    if "ModuleNotFoundError: No module named 'hydride.relax'" in stdout:
        pytest.skip("Relax module is only compiled via pyximport")
    assert completed_process.returncode == 0

    with open(output_file, "wb") as file:
        file.write(completed_process.stdout)
    
    assert_hydrogen_addition(output_file)


@pytest.mark.parametrize("res_name", ["URA"])
def test_extra_fragments(output_file, res_name):
    """
    If the fragments are given for the molecule, where hydrogens atoms
    should be added, the atom names should fit exactly and the hydrogen
    coordinates should fit almost exactly.
    """
    TOLERANCE = 0.01

    ref_molecule = info.residue(res_name)
    frag_temp_file = tempfile.NamedTemporaryFile(
        "w", suffix=f".mmtf", delete=False
    )
    # As the test case is constructed with the exact same molecule
    # can be in the library, move the molecule to assure that
    # correct hydrogen position calculation is not an artifact
    np.random.seed(0)
    frag_molecule = struc.rotate(ref_molecule, np.random.rand(3))
    frag_molecule = struc.translate(frag_molecule, np.random.rand(3))
    strucio.save_structure(frag_temp_file.name, frag_molecule)

    heavy_atoms = ref_molecule[ref_molecule.element != "H"]
    input_temp_file = tempfile.NamedTemporaryFile(
        "w", suffix=f".mmtf", delete=False
    )
    strucio.save_structure(input_temp_file.name, heavy_atoms)

    run_cli([
        "-v",
        "-i", input_temp_file.name,
        "-f", frag_temp_file.name,
        "-o", output_file,
    ])

    frag_temp_file.close()
    os.remove(frag_temp_file.name)
    input_temp_file.close()
    os.remove(input_temp_file.name)

    test_molecule = strucio.load_structure(output_file)
    
    assert test_molecule.array_length() == ref_molecule.array_length()
    # Atoms are added to AtomNameLibrary
    # -> atom names should fit exactly
    assert test_molecule.atom_name.tolist() == ref_molecule.atom_name.tolist()
    # The fragments are from the same molecule
    # -> The hydrogen positions should be almost fit exactly
    assert np.max(struc.distance(test_molecule, ref_molecule)) <= TOLERANCE


def test_ignored_residues(input_file, output_file):
    """
    Test if ignored residues have no associated hydrogen atoms.
    """
    ignored_residues = (5, 13)
    run_cli(
        [
            "-v",
            "-i", input_file,
            "-g", "A", "5",
            "-g", "A", "13",
            "-o", output_file,
        ]
        + list(itertools.chain(
            *[("-g", "A", str(res_id)) for res_id in ignored_residues]
        ))
    )

    test_model = strucio.load_structure(output_file)

    # There should be added hydrogen atoms in the file...
    assert (test_model.element == "H").any()
    # ... but not for the ignored residues
    for res_id in ignored_residues:
        assert not (
            (test_model.element == "H") & (test_model.res_id == res_id)
        ).any()


def test_limited_iterations(input_file, output_file):
    """
    Test CLI run with ``--iterations`` parameter.
    """
    run_cli([
        "-v",
        "-i", input_file,
        "-o", output_file,
        "-c", str(PH),
        "--iterations", str(100)
    ])

    assert_hydrogen_addition(output_file)


def test_angle_increment(input_file, output_file):
    """
    Test CLI run with ``--angle_increment`` parameter.
    """
    run_cli([
        "-v",
        "-i", input_file,
        "-o", output_file,
        "-c", str(PH),
        "--angle-increment", str(5)
    ])

    assert_hydrogen_addition(output_file)


def test_pbc(input_file, output_file):
    """
    Test CLI run with ``--pbc`` parameter.
    """
    run_cli([
        "-v",
        "-i", input_file,
        "-o", output_file,
        "-c", str(PH),
        "--pbc"
    ])

    assert_hydrogen_addition(output_file)


def test_invalid_iteratons(input_file, dummy_output_file):
    with pytest.raises(SystemExit) as wrapped_exception:
        run_cli([
            "-i", input_file,
            "-o", dummy_output_file,
            "-n", str(-1)
        ])
    assert wrapped_exception.value.code == 1


@pytest.mark.parametrize("model", [-10, -1, 0, 39, 100])
def test_invalid_model(input_file, dummy_output_file, model):
    with pytest.raises(SystemExit) as wrapped_exception:
        run_cli([
            "-i", input_file,
            "-o", dummy_output_file,
            "-m", str(model)
        ])
    assert wrapped_exception.value.code == 1


def test_unknown_input_format(input_file, dummy_output_file):
    with pytest.raises(SystemExit) as wrapped_exception:
        run_cli([
            "-i", input_file,
            "-o", dummy_output_file,
            "-I", "abc"
        ])
    assert wrapped_exception.value.code == 2


def test_unknown_output_format(input_file, dummy_output_file):
    with pytest.raises(SystemExit) as wrapped_exception:
        run_cli([
            "-i", input_file,
            "-o", dummy_output_file,
            "-O", "abc"
        ])
    assert wrapped_exception.value.code == 2


def test_unknown_extension(input_file, dummy_output_file):
    temp_file = tempfile.NamedTemporaryFile(
        "w", suffix=".abc", delete=False
    )
    shutil.copy(input_file, temp_file.name)
    
    with pytest.raises(SystemExit) as wrapped_exception:
        run_cli([
            "-i", temp_file.name,
            "-o", dummy_output_file,
        ])
    assert wrapped_exception.value.code == 1

    temp_file.close()
    os.remove(temp_file.name)


def test_noexisting_file(dummy_output_file):
    with pytest.raises(SystemExit) as wrapped_exception:
        run_cli([
            "-i", "/not/a/file.pdb",
            "-o", dummy_output_file,
        ])
    assert wrapped_exception.value.code == 1


def test_invalid_file_data(input_file, dummy_output_file):
    in_format = splitext(input_file)[-1]
    temp_file = tempfile.NamedTemporaryFile(
        "wb", suffix=in_format, delete=False
    )
    temp_file.write(b"ATOM Invalid data")

    with pytest.raises(SystemExit) as wrapped_exception:
        run_cli([
            "-i", temp_file.name,
            "-o", dummy_output_file,
        ])
    assert wrapped_exception.value.code == 1


def test_missing_ignore_residue(input_file, dummy_output_file):
    with pytest.raises(SystemExit) as wrapped_exception:
        run_cli([
            "-i", input_file,
            "-o", dummy_output_file,
            "-g", "A", "42"
        ])
    assert wrapped_exception.value.code == 1