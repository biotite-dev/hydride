# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "hydride"
__author__ = "Patrick Kunzmann, Jacob Marcel Anter"
__all__ = ["relax_hydrogen"]

cimport cython
cimport numpy as np

import numpy as np
import biotite.structure as struc
import biotite.structure.info as info


ctypedef np.uint8_t uint8
ctypedef np.int8_t int8
ctypedef np.int32_t int32
ctypedef np.float32_t float32


cdef int SINGLE = struc.BondType.SINGLE
cdef int DOUBLE = struc.BondType.SINGLE
cdef int AROMATIC_DOUBLE = struc.BondType.AROMATIC_DOUBLE


class EnergyFunction:

    def __init__(self, atoms, relevant_mask,
                 force_cutoff=10.0, partial_charges=None):
        cdef int i, j, pair_i
        cdef int32 atom_i, atom_j


        cell_list = struc.CellList(
            atoms, cell_size=force_cutoff
        )
        cdef int32[:] relevant_indices = np.where(relevant_mask)[0] \
                                         .astype(np.int32)
        cdef int32[:,:] adj_indices = cell_list.get_atoms(
            atoms.coord[np.asarray(relevant_indices)],
            radius=force_cutoff
        )

        # Calculate which pairs of atoms interact with each other
        # with respect to the cutoff distance
        # Array allocation with pessimistic size assumption.
        # In the worst case, there are no padded -1 values
        cdef int32[:,:] interaction_pairs = np.zeros(
            (adj_indices.shape[0] * adj_indices.shape[1], 2),
            dtype=np.int32
        )
        cdef uint8[:] mask = relevant_mask.astype(np.uint8)
        pair_i = 0
        for i in range(adj_indices.shape[0]):
            for j in range(adj_indices.shape[1]):
                if adj_indices[i, j] == -1:
                    # Padding values
                    continue
                atom_i = relevant_indices[i]
                atom_j = adj_indices[i, j]
                # Do not include H-H-interaction duplicates
                # or an interaction of an atom to itself
                if not mask[atom_j] or atom_j > atom_i:
                    interaction_pairs[pair_i, 0] = atom_i
                    interaction_pairs[pair_i, 1] = atom_j
                    pair_i += 1
        # Trim to correct size
        self._interaction_pairs = np.asarray(interaction_pairs)[:pair_i]


        # Calculate Coulomb parameters for interaction pairs
        cdef float32[:] charges 
        if partial_charges is None:
            charges = struc.partial_charges(atoms)
        else:
            charges = partial_charges.astype(np.float32, copy=False)
        cdef float32[:] coulomb_param  = np.zeros(pair_i, dtype=np.float32)
        for i in range(pair_i):
            coulomb_param[i] = (
                charges[interaction_pairs[i, 0]] * 
                charges[interaction_pairs[i, 1]]
            )
        self._coulomb_param  = np.asarray(coulomb_param)


        # Calculate Lennart-Jones parameters for interaction pairs
        cdef float32[:] vdw_radii = np.array(
            [info.vdw_radius_single(element) for element in atoms.element],
            dtype=np.float32
        )
        cdef float32[:] r_6  = np.zeros(pair_i, dtype=np.float32)
        cdef float32[:] r_12 = np.zeros(pair_i, dtype=np.float32)
        for i in range(pair_i):
            r_6[i] = (-0.5 * (
                vdw_radii[interaction_pairs[i, 0]] + 
                vdw_radii[interaction_pairs[i, 1]]
            ))**6
            r_12[i] = r_6[i]**2
        self._r_6  = np.asarray(r_6)
        self._r_12 = np.asarray(r_12)


    def __call__(self, coord):
        distances = struc.index_distance(coord, self._interaction_pairs) \
                    .astype(np.float32, copy=False)
        
        return np.sum(
            # Electrostatic interaction
            self._coulomb_param / distances
            # Lennart-Jones interaction
            + self._r_6 * distances**6 + self._r_12 * distances**12
        )


def relax_hydrogen(atoms, iteration_number=1):
    coord = atoms.coord.copy()

    rotatable_bonds = _find_rotatable_bonds(atoms)
    if len(rotatable_bonds) == 0:
        # No bond to optimize
        return coord

    rotation_axes = np.zeros(
        (len(rotatable_bonds), 2, 3), dtype=np.float32
    )
    matrix_indices = np.full(
        atoms.array_length(), -1, dtype=np.int32
    )
    cdef uint8[:] full_freedom = np.zeros(
        len(rotatable_bonds), dtype=np.uint8
    )
    hydrogen_mask = np.zeros(atoms.array_length(), dtype=bool)
    for i, (
        central_atom_index, bonded_atom_index, is_free, hydrogen_indices
    ) in enumerate(rotatable_bonds):
        rotation_axes[i, 0] = coord[central_atom_index]
        rotation_axes[i, 1] = coord[bonded_atom_index]
        matrix_indices[hydrogen_indices] = i
        full_freedom[i] = is_free
        hydrogen_mask[hydrogen_indices] = True
    
    cdef float32[:,:,:] rotation_matrices = np.zeros(
        (len(rotatable_bonds), 3, 3), dtype=np.float32
    )
    cdef int32[:] mat_indices = matrix_indices
    cdef float32[:,:] axes = rotation_axes[:, 1] - rotation_axes[:, 0]
    cdef float32[:,:] support = rotation_axes[:, 0]

    energy_function = EnergyFunction(atoms, hydrogen_mask)
    
    for _ in range(iteration_number):
        energy = energy_function(coord)
        print(energy)

    return coord


def _find_rotatable_bonds(atoms):
    cdef int i, j, h_i, bonded_i

    if atoms.bonds is None:
        raise struc.BadStructureError(
            "The input structure must have an associated BondList"
        )
    cdef int32[:,:] all_bond_indices
    cdef int8[:,:] all_bond_types
    all_bond_indices, all_bond_types = atoms.bonds.get_all_bonds()

    cdef uint8[:] is_hydrogen = (atoms.element == "H").astype(np.uint8)
    cdef uint8[:] is_nitrogen = (atoms.element == "N").astype(np.uint8)
    cdef uint8[:] is_oxygen   = (atoms.element == "O").astype(np.uint8)
    
    cdef list rotatable_bonds = []

    cdef int32[:] hydrogen_indices = np.zeros(4, np.int32)
    cdef bint is_free
    cdef int bonded_heavy_index
    cdef int bonded_heavy_btype
    cdef int rem_index
    cdef int rem_btype
    cdef bint is_rotatable
    for i in range(all_bond_indices.shape[0]):
        if is_hydrogen[i]:
            continue

        bonded_heavy_index = -1
        bonded_heavy_btype = -1
        is_rotatable = True
        h_i = 0
        
        # Check for number of bonded heavy atoms
        # and store bonded hydrogen atoms
        for j in range(all_bond_indices.shape[1]):
            bonded_i = all_bond_indices[i, j]
            if bonded_i == -1:
                # -1 is a padding value that appears after all actual
                # values have been interated through
                break
            if is_hydrogen[bonded_i]:
                hydrogen_indices[h_i] = bonded_i
                h_i += 1
            elif bonded_heavy_index == -1:
                bonded_heavy_index = bonded_i
                bonded_heavy_btype = all_bond_types[i, j]
            else:
                # There is already a bonded heavy atom
                # -> there are at least two bonded heavy atom
                # -> no rotational freedom
                is_rotatable = False
                break

        # The rotation freedom might be restricted to 180 degrees
        # in a (partially) double bonded heavy atom
        if is_rotatable:
            if bonded_heavy_btype == SINGLE:
                is_free = True
                # Check for partial double bonds
                # Nitrogen is the only relevant atom for this case
                if is_nitrogen[i]:
                    for j in range(all_bond_indices.shape[1]):
                        rem_index = all_bond_indices[i, j]
                        rem_btype = all_bond_types[i, j]
                        # If the adjacent atom has a double bond to
                        # either a nitrogen or oxygen atom or is part of
                        # an aromatic system, the partial double bond
                        # condition is fulfilled
                        if rem_btype == AROMATIC_DOUBLE or (
                            rem_btype == DOUBLE 
                            and (is_oxygen[rem_index] or is_nitrogen[rem_index])
                        ):
                            is_free = False
                            break
            elif bonded_heavy_btype == DOUBLE:
                is_free = False
            else:
                # Triple bond etc. -> no rotational freedom
                is_rotatable = False
                is_free = False
        
        # 180 degrees rotation makes only sense if there is only one
        # hydrogen atom, as two hydrogen atoms would simply replace each
        # other after rotation
        if is_rotatable and not is_free and h_i > 1:
            is_rotatable = False

        # Add a rotatable bond to list of rotatable bonds
        if is_rotatable:
            h_indices_array = np.asarray(hydrogen_indices)
            h_indices_array = h_indices_array[:h_i]
            rotatable_bonds.append(
                (i, bonded_heavy_index, is_free, h_indices_array.copy())
            )

    return rotatable_bonds