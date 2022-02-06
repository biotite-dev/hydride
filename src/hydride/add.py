# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "hydride"
__author__ = "Patrick Kunzmann, Jacob Marcel Anter"
__all__ = ["add_hydrogen"]

import numpy as np
import biotite.structure as struc
from .fragments import FragmentLibrary
from .names import AtomNameLibrary


def add_hydrogen(atoms, mask=None, fragment_library=None, name_library=None,
                 box=None):
    """
    Add hydrogen atoms to a structure.

    The hydrogen atoms for each residue are placed directly behind the
    atoms from this residue.
    The function also tries to assign the correct atom name to each
    added hydrogen atom.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,)
        The structure, where hydrogen atoms should be added.
        The structure must have an associated :class:`BondList`.
        The structure must also include the *charge* annotation array,
        depicting the formal charge for each atom.
        The structure must not have any annotated hydrogen atoms
        (in the region covered by `mask`), yet.
    mask : ndarray, shape=(n,), dtype=bool, optional
        A boolean mask that is true for each atom, where hydrogen atoms
        should be added.
        By default, hydrogen atoms are added to all applicable atoms.
    fragment_library : FragmentLibrary
        The fragment library to use for hydrogen position estimation.
        By default :meth:`FragmentLibrary.standard_library()` is used,
        containing fragments from all molecules in the
        *RCSB* *Chemical Component Dictionary*.
    name_library : AtomNameLibrary
        The atom name library to use for hydrogen naming.
        By default :meth:`AtomNameLibrary.standard_library()` is used,
        containing atom names for the most prominent molecules including
        amino acids and nucleotides.
        For all other molecules the hydrogen atom names are guessed.
    box : bool or array-like, shape=(3,3), dtype=float, optional
        If this parameter is set, periodic boundary conditions are
        taken into account (minimum-image convention), based on
        the box vectors given with this parameter.
        If `box` is set to true, the box vectors are taken from the
        ``box`` attribute of `atoms` instead.

    Returns
    -------
    hydrogenated_atoms : AtomArray, shape=(p,)
        The atoms from the input `atoms` with additional hydrogen atoms.
        Although, the hydrogen positions are meaningful with respect to
        bond lengths and angles, the dihedral angles are not optimized.
    original_atom_mask : ndarray, shape=(p,)
        A boolean mask that is true for each atom in
        `hydrogenated_atoms` that originates from the input `atoms`.
        That means, that if the mask is applied to `hydrogenated_atoms`,
        the input structure is restored.
    """
    if mask is None:
        mask = np.ones(atoms.array_length(), dtype=bool)

    if fragment_library is None:
        fragment_library = FragmentLibrary.standard_library()
    if name_library is None:
        name_library = AtomNameLibrary.standard_library()

    if (atoms.element[mask] == "H").any():
        raise struc.BadStructureError(
           "Input structure already contains hydrogen atoms"
        )

    hydrogen_coord = fragment_library.calculate_hydrogen_coord(
        atoms, mask, box
    )

    # Count number of hydrogen atoms to be added
    count = 0
    p = 0
    for coord in hydrogen_coord:
        count += len(coord)

    # Create new empty AtomArray with an appropriate length for all
    # heavy and hydrogen atoms
    hydrogenated_atoms = struc.AtomArray(atoms.array_length() + count)
    original_atom_mask = np.zeros(
        hydrogenated_atoms.array_length(), dtype=bool
    )
    # Add all annotation categories of the original AtomArray
    for category in atoms.get_annotation_categories():
        if category not in hydrogenated_atoms.get_annotation_categories():
            hydrogenated_atoms.add_annotation(
               category, dtype=atoms.get_annotation(category).dtype
            )
    if atoms.box is not None:
        hydrogenated_atoms.box = atoms.box.copy()

    # Fill the combined AtomArray residue for residue
    # Stores covalent bonds between a heavy atom and its hydrogen atoms
    hydrogen_bonds = []
    residue_starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    # Maps atom indices of the input AtomArray
    # to indices of the hydrogenated AtomArray
    index_mapping = np.zeros(atoms.array_length(), dtype=np.uint32)
    p = 0
    for i in range(len(residue_starts) - 1):
        # Set annotation and coordinates from input AtomArray
        start = residue_starts[i]
        stop = residue_starts[i+1]
        res_length = stop - start
        index_mapping[start : stop] = np.arange(p, p + res_length)
        original_atom_mask[p : p + res_length] = True
        hydrogenated_atoms.coord[p : p + res_length] = atoms.coord[start:stop]
        for category in atoms.get_annotation_categories():
            hydrogenated_atoms.get_annotation(category)[p : p + res_length] \
                       = atoms.get_annotation(category)[start : stop]
        p += res_length
        # Set annotation and coordinates for hydrogen atoms
        for j in range(start, stop):
            hydrogen_coord_for_atom = hydrogen_coord[j]
            hydrogen_name_generator = name_library.generate_hydrogen_names(
               atoms.res_name[j], atoms.atom_name[j]
            )
            for coord in hydrogen_coord_for_atom:
                hydrogenated_atoms.coord[p] = coord
                hydrogenated_atoms.chain_id[p] = atoms.chain_id[j]
                hydrogenated_atoms.res_id[p] = atoms.res_id[j]
                hydrogenated_atoms.ins_code[p] = atoms.ins_code[j]
                hydrogenated_atoms.res_name[p] = atoms.res_name[j]
                hydrogenated_atoms.hetero[p] = atoms.hetero[j]
                hydrogenated_atoms.atom_name[p] = next(hydrogen_name_generator)
                hydrogenated_atoms.element[p] = "H"
                heavy_index = index_mapping[j]
                hydrogen_index = p
                hydrogen_bonds.append((heavy_index, hydrogen_index))
                p += 1

    # Add bonds to combined AtomArray
    original_bonds = atoms.bonds.as_array()
    bond_indices = index_mapping[original_bonds[:,:2]]
    heavy_bonds = np.stack(
       [
          bond_indices[:, 0],
          bond_indices[:, 1],
          # The bond types
          original_bonds[:,2]
       ],
       axis=-1
    )
    hydrogen_bonds = np.array(hydrogen_bonds, dtype=np.uint32).reshape(-1, 2)
    # All bonds to hydrogen atoms are single bonds
    hydrogen_bonds = np.stack(
       [
          hydrogen_bonds[:, 0],
          hydrogen_bonds[:, 1],
          np.ones(len(hydrogen_bonds), dtype=np.uint32)
       ],
       axis=-1
    )
    hydrogenated_atoms.bonds = struc.BondList(
       hydrogenated_atoms.array_length(),
       np.concatenate([heavy_bonds, hydrogen_bonds])
    )

    return hydrogenated_atoms, original_atom_mask
