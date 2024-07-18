# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "hydride"
__author__ = "Patrick Kunzmann"

import argparse
import sys
import warnings
from os.path import splitext
import biotite.structure as struc
import biotite.structure.io.mol as mol
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
from hydride.add import add_hydrogen
from hydride.charge import estimate_amino_acid_charges
from hydride.fragments import FragmentLibrary
from hydride.names import AtomNameLibrary
from hydride.relax import relax_hydrogen


class UserInputError(Exception):
    pass


def main(args=None):
    parser = argparse.ArgumentParser(
        description="This program adds hydrogen atoms to molecular "
        "structures where these are missing.\n"
        "For more information, please visit "
        "https://hydride.biotite-python.org/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--infile",
        "-i",
        metavar="FILE",
        help="The path to the input structure file containing the model "
        "without hydrogen atoms. "
        "If omitted, the file is read from STDOUT.",
    )
    parser.add_argument(
        "--outfile",
        "-o",
        metavar="FILE",
        help="The path to the output structure file where the model "
        "with added hydrogen atoms should be written to. "
        "If omitted, the file is written to STDOUT. "
        "Any existing hydrogen atoms will be removed the model.",
    )
    parser.add_argument(
        "--informat",
        "-I",
        choices=["pdb", "pdbx", "cif", "bcif", "sdf", "mol"],
        help="The file format of the input file. "
        "Must be specified if input file is read from STDIN. "
        "If omitted, the file format is guessed from the suffix "
        "of the file.",
    )
    parser.add_argument(
        "--outformat",
        "-O",
        choices=["pdb", "pdbx", "cif", "bcif", "mol"],
        help="The file format of the output file. "
        "Must be specified if output file is written to STDOUT."
        "If omitted, the file format is guessed from the suffix "
        "of the file.",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose display of errors."
    )

    parser.add_argument(
        "--no-relax",
        action="store_true",
        help="Omit the relaxation step. "
        "Note bond lengths and angles will still be correct. "
        "However clashes or electrostatically unfavorable "
        "conformations will not be resolved.",
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        metavar="NUMBER",
        help="The maximum number of relaxation iterations. "
        "The runtime of the relaxation scales approximately "
        "linear with this value, if the relaxation does not "
        "terminate before. "
        "By default, the relaxation runs until a local optimum "
        "has been reached.",
    )
    parser.add_argument(
        "--angle-increment",
        "-a",
        type=float,
        metavar="NUMBER",
        default=10.0,
        help="The angle in degrees that a freely rotatable bond is rotated "
        "in each relaxation step."
        "Lower values increase the accuracy of hydrogen positioning, "
        "but increase the required number of steps until an optimum "
        "is found.",
    )
    parser.add_argument(
        "--fragments",
        "-f",
        metavar="FILE",
        action="append",
        help="Additional structure file to containing fragments for the "
        "fragment library. "
        "This can be used supply fragments for molecules with uncommon "
        "groups, if the standard fragment library does not contain such "
        "fragments, yet. "
        "May be supplied multiple times.",
    )
    parser.add_argument(
        "--fragformat",
        "-F",
        choices=["pdb", "pdbx", "cif", "bcif", "sdf", "mol"],
        help="The file format of the additional structure files. "
        "If omitted, the file format is guessed from the suffix "
        "of the file.",
    )
    parser.add_argument(
        "--ignore",
        "-g",
        metavar="RESIDUE",
        action="append",
        nargs=2,
        help="No hydrogen atoms are added to the specified residue. "
        "The format is '{chain} {residue}', e.g. 'A 123'. "
        "May be supplied multiple times, if multiple residues should be "
        "ignored.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=int,
        metavar="NUMBER",
        default=1,
        help="The model number, if the input structure file contains multiple "
        "models.",
    )
    parser.add_argument(
        "--charges",
        "-c",
        type=float,
        metavar="PH",
        help="Recalculate the charges of atoms in canonical amino acids based "
        "on the given pH value. "
        "This estimation does not take the surrounding amino acids into "
        "account.",
    )
    parser.add_argument(
        "--pbc",
        "-p",
        action="store_true",
        help="Set hydrogen addition and relaxation aware to periodic boundary "
        "conditions. "
        "The box is read from the input structure file.",
    )

    args = parser.parse_args(args=args)

    try:
        run(args)
    except UserInputError as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            raise
        else:
            sys.exit(1)
    except Exception:
        print("An unexpected error occured:\n", file=sys.stderr)
        raise


def run(args):
    frag_library = FragmentLibrary.standard_library()
    name_library = AtomNameLibrary.standard_library()
    if args.fragments is not None:
        for frag_path in args.fragments:
            try:
                model = read_structure(frag_path, args.fragformat, 1)
            except UserInputError:
                raise
            except PermissionError:
                raise UserInputError(
                    f"Missing file permission for reading '{frag_path}'"
                )
            except FileNotFoundError:
                raise UserInputError(f"Input file '{args.infile}' cannot be found")
            except Exception:
                raise UserInputError(
                    f"Input file '{args.infile}' contains invalid data"
                )
            frag_library.add_molecule(model)
            name_library.add_molecule(model)

    try:
        model = read_structure(args.infile, args.informat, args.model)
    except UserInputError:
        raise
    except PermissionError:
        raise UserInputError(f"Missing file permission for reading '{args.infile}'")
    except FileNotFoundError:
        raise UserInputError(f"Input file '{args.infile}' cannot be found")
    except Exception:
        if args.infile is None:
            raise UserInputError("Input file contains invalid data")
        else:
            raise UserInputError(f"Input file '{args.infile}' contains invalid data")

    heavy_mask = model.element != "H"
    if not heavy_mask.all():
        warnings.warn("Existing hydrogen atoms were removed")
        model = model[heavy_mask]
        pass

    if args.charges:
        aa_mask = struc.filter_amino_acids(model)
        charges = estimate_amino_acid_charges(model, args.charges)
        model.charge[aa_mask] = charges[aa_mask]

    input_mask = np.ones(model.array_length(), dtype=bool)
    if args.ignore is not None:
        for chain_id, res_id in args.ignore:
            res_id = int(res_id)
            removal_mask = (model.chain_id == chain_id) & (model.res_id == res_id)
            if not removal_mask.any():
                raise UserInputError(
                    f"Cannot find '{chain_id} {res_id}' " "in the input structure"
                )
            input_mask &= ~removal_mask

    if args.pbc:
        if model.box is None:
            raise UserInputError(
                "The input structure file does not provide box vectors "
                "required for handling periodic boundary conditions"
            )
        box = True
    else:
        box = None

    model, _ = add_hydrogen(model, input_mask, frag_library, name_library, box)
    if not args.no_relax:
        if args.iterations is not None and args.iterations < 0:
            raise UserInputError("The number of iterations must be positive")
        model.coord = relax_hydrogen(
            model,
            args.iterations,
            angle_increment=np.deg2rad(args.angle_increment),
            box=box,
        )

    try:
        write_structure(args.outfile, args.outformat, model)
    except UserInputError:
        raise
    except PermissionError:
        raise UserInputError(f"Missing file permission for writing '{args.outfile}'")
    except:
        raise


def read_structure(path, format, model_number):
    if format is None:
        if path is None:
            raise UserInputError(
                "The input file format must be given, "
                "if the input file is read from STDIN"
            )
        format = guess_format(path)

    if model_number < 1:
        raise UserInputError("Model number must be positive")

    if path is None:
        path = sys.stdin

    match format:
        case "pdb":
            pdb_file = pdb.PDBFile.read(path)
            model_count = pdb.get_model_count(pdb_file)
            if model_number > model_count:
                raise UserInputError(
                    f"Model number {model_number} is out of range "
                    f"for the input structure with {model_count} models"
                )
            model = pdb.get_structure(
                pdb_file,
                model=model_number,
                extra_fields=["charge"],
                include_bonds=True,
            )
            # Expect that all ANY bonds are actually SINGLE bonds
            bond_array = model.bonds.as_array()
            unknown_order_mask = bond_array[:, 2] == struc.BondType.ANY
            if unknown_order_mask.any():
                warnings.warn(
                    "For some bonds the bond order is unknown, "
                    "hence single bonds are assumed"
                )
                bond_array[unknown_order_mask, 2] = struc.BondType.SINGLE
                model.bonds = struc.BondList(model.array_length(), bond_array)
        case "pdbx" | "cif" | "bcif":
            if format == "bcif":
                if path == sys.stdin:
                    # Special handling for binary input
                    pdbx_file = pdbx.BinaryCIFFile.read(sys.stdin.buffer)
                else:
                    pdbx_file = pdbx.BinaryCIFFile.read(path)
            else:
                pdbx_file = pdbx.CIFFile.read(path)
            model_count = pdbx.get_model_count(pdbx_file)
            if model_number > model_count:
                raise UserInputError(
                    f"Model number {model_number} is out of range "
                    f"for the input structure with {model_count} models"
                )
            model = pdbx.get_structure(
                pdbx_file,
                model=model_number,
                extra_fields=["charge"],
                include_bonds=True,
            )
            model.bonds = struc.connect_via_residue_names(model)
        case "mol":
            mol_file = mol.MOLFile.read(path)
            if model_number > 1:
                raise UserInputError(
                    f"Model number {model_number} is out of range "
                    f"for the input structure with 1 models"
                )
            model = mol_file.get_structure()
            model.res_name[:] = mol_file.header.mol_name
            model.atom_name[:] = struc.create_atom_names(model)
        case "sdf":
            sd_file = mol.SDFile.read(path)
            mol_name, sd_block = next(iter(sd_file.items()))
            if model_number > 1:
                raise UserInputError(
                    f"Model number {model_number} is out of range "
                    f"for the input structure with 1 models"
                )
            model = sd_block.get_structure()
            model.res_name[:] = mol_name
            model.atom_name[:] = struc.create_atom_names(model)
        case _:
            raise UserInputError(f"Unknown file format '{format}'")

    return model


def write_structure(path, format, model):
    if format is None:
        if path is None:
            raise UserInputError(
                "The output file format must be given, "
                "if the output written to STDOUT"
            )
        format = guess_format(path)

    if path is None:
        path = sys.stdout

    match format:
        case "pdb":
            pdb_file = pdb.PDBFile()
            pdb.set_structure(pdb_file, model)
            pdb_file.write(path)
        case "pdbx" | "cif":
            pdbx_file = pdbx.CIFFile()
            pdbx.set_structure(
                pdbx_file, model, data_block="STRUCTURE", include_bonds=True
            )
            pdbx_file.write(path)
        case "bcif":
            pdbx_file = pdbx.BinaryCIFFile()
            pdbx.set_structure(
                pdbx_file, model, data_block="STRUCTURE", include_bonds=True
            )
            if path == sys.stdout:
                # Special handling for binary output
                pdbx_file.write(sys.stdout.buffer)
            else:
                pdbx_file.write(path)
        case "mol":
            mol_file = mol.MOLFile()
            mol_file.set_structure(model)
            mol_file.header.mol_name = model.res_name[0]
            mol_file.write(path)
        case "sdf":
            sd_file = mol.SDFile()
            mol_name = model.res_name[0]
            mol.set_structure(sd_file, model, record_name=mol_name)
            sd_file.write(path)
        case _:
            raise UserInputError(f"Unknown file format '{format}'")


def guess_format(path):
    suffix = splitext(path)[-1].lower()
    match suffix:
        case ".pdb":
            return "pdb"
        case ".pdbx" | ".cif" | ".mmcif":
            return "cif"
        case ".bcif":
            return "bcif"
        case ".mol":
            return "mol"
        case ".sdf":
            return "sdf"
        case _:
            raise UserInputError(f"Unknown file extension '{suffix}'")
