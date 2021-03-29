# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "hydride"
__author__ = "Patrick Kunzmann, Jacob Marcel Anter"
__all__ = ["FragmentLibrary"]

import warnings
import numpy as np


class FragmentLibrary:

    def __init__(self):
        self._frag_dict = {}
    

    def add_molecule(self, molecule):
        fragments = _fragment(molecule)
        for fragment in fragments:
            if fragment is None:
                continue
            (
                central_element, central_charge, stereo, bond_types,
                center_coord, heavy_coord, hydrogen_coord
            ) = fragment
            # Translate the coordinates,
            # so the central heavy atom is at origin
            centered_heavy_coord = heavy_coord - center_coord
            centered_hydrogen_coord = hydrogen_coord - center_coord
            # TODO Maybe use multiple coordinate sets
            # representing different conformations
            self._frag_dict[(
                central_element, central_charge, stereo, tuple(bond_types)
            )] = (centered_heavy_coord, centered_hydrogen_coord)
            if stereo != 0:
                pass
                # Also include the opposite enantiomer in the library
                # by reflecting the coordinates along an arbitrary axis
                stereo *= -1
                refl_centered_heavy_coord = centered_heavy_coord.copy()
                refl_centered_hydrogen_coord = centered_hydrogen_coord.copy()
                refl_centered_heavy_coord[..., 0] *= -1
                refl_centered_hydrogen_coord[..., 0] *= -1
                self._frag_dict[(
                    central_element, central_charge, stereo, tuple(bond_types)
                )] = (refl_centered_heavy_coord, refl_centered_hydrogen_coord)
        
    
    def get_fragment_coord(self, central_element, central_charge,
                           bonded_elements, bonded_charges, bond_types):
        """
        Returns
        -------
        heavy_coord : ndarray, shape=(3,3), dtype=np.float32
        hydrogen_coord : ndarray, shape=(k,3), dtype=np.float32
        """


    def calculate_hydrogen_coord(self, structure):
        """
        Returns
        -------
        hydrogen_coord : ndarray, shape=(n,4,3), dtype=np.float32
            Padded with *NaN* values.
        """
        # The subject and reference heavy atom coordinates for each fragment
        sub_frag_center_coord = np.full(
            (structure.array_length(), 3), np.nan, dtype=np.float32
        )
        sub_frag_heavy_coord = np.full(
            (structure.array_length(), 3, 3), np.nan, dtype=np.float32
        )
        ref_frag_heavy_coord = np.full(
            (structure.array_length(), 3, 3), np.nan, dtype=np.float32
        )
        # The amount of hydrogens varies for each fragment
        # -> padding with NaN
        ref_frag_hydrogen_coord = np.full(
            (structure.array_length(), 4, 3), np.nan, dtype=np.float32
        )

        # Fil the coordinate arrays
        fragments = _fragment(structure)
        for i, fragment in enumerate(fragments):
            (
                central_element, central_charge, stereo, bond_types,
                center_coord, heavy_coord, _
            ) = fragment
            sub_frag_center_coord[i] = center_coord
            sub_frag_heavy_coord[i] = heavy_coord
            # The hydrogen_coord can be ignored:
            # In the subject structure are no hydrogen atoms
            hit = self._frag_dict[
                (central_element, central_charge, stereo, tuple(bond_types))
            ]
            if hit is None:
                warnings.warn(f"Missing fragment for atom at position {i}")
                ref_hydrogen_coord[i] = np.zeros(0, dtype=np.float32)
            ref_heavy_coord, ref_hydrogen_coord = hit
            ref_frag_heavy_coord[i] = ref_heavy_coord
            ref_frag_hydrogen_coord[i, :len(ref_hydrogen_coord)] \
                = ref_hydrogen_coord

        # Translate the subject coordinates,
        # so the central heavy atom is at origin
        # This has already been done for the reference atoms
        # in the 'add_molecule()' method
        sub_frag_heavy_coord -= sub_frag_center_coord[:, np.newaxis, :]
        # Get the rotation matrix required for superimposition of
        # the reference coord to the subject coord 
        matrices = _get_rotation_matrices(
            sub_frag_heavy_coord, ref_frag_heavy_coord
        )
        # Rotate the reference hydrogen atoms, so they fit the
        # subject heavy atoms
        sub_frag_hydrogen_coord = _rotate(ref_frag_hydrogen_coord, matrices)
        # Translate hydrogen atoms to the position of the
        # non-centered central heavy subject atom
        sub_frag_hydrogen_coord += sub_frag_center_coord[:, np.newaxis, :]
        
        # Turn into list ad remove NaN paddings
        sub_frag_hydrogen_coord = [
            # If the x-coordinate is NaN it is expected that
            # y and z are also NaN
            coord[~np.isnan(coord[:, 0])] for coord in sub_frag_hydrogen_coord
        ]

        return sub_frag_hydrogen_coord


def _fragment(structure):
    fragments = [None] * structure.array_length()
    
    all_bond_indices, all_bond_types = structure.bonds.get_all_bonds()
    elements = structure.element
    charges = structure.charge
    coord = structure.coord

    for i in range(structure.array_length()):
        if elements[i] == "H":
            # Only create fragments for heavy atoms
            continue
        bond_indices = all_bond_indices[i]
        bond_types = all_bond_types[i]
        bond_indices = bond_indices[bond_indices != -1]
        bond_types = bond_types[bond_types != -1]

        heavy_mask = (elements[bond_indices] != "H")
        heavy_indices = bond_indices[heavy_mask]
        heavy_types = bond_types[heavy_mask]

        # Order the bonded atoms by their bond types
        # to remove atom order dependency in the matching step 
        order = np.argsort(heavy_types)
        heavy_indices = heavy_indices[order]
        heavy_types = heavy_types[order]

        hydrogen_mask = ~heavy_mask
        hydrogen_coord = coord[bond_indices[hydrogen_mask]]

        n_heavy_bonds = np.count_nonzero(heavy_mask)
        if n_heavy_bonds == 0:
            # The orientation is arbitrary
            # -> The fragment coord is the coord of the central atom
            # 4 times repeated
            heavy_coord = np.repeat(coord[np.newaxis, i, :], 3, axis=0)
            stereo = 0
        elif n_heavy_bonds == 1:
            # Include one atom further away
            # to get an unambiguous fragment
            remote_index = heavy_indices[0]
            rem_bond_indices = all_bond_indices[remote_index]
            rem_bond_indices = rem_bond_indices[rem_bond_indices != -1]
            rem_heavy_mask = (elements[rem_bond_indices] != "H")
            rem_heavy_indices = rem_bond_indices[rem_heavy_mask]
            # Use the coord of any heavy atom bonded to the remote
            # atom
            rem_rem_index = rem_bond_indices[0]
            # Include the directly bonded atom two times, to give it a
            # greater weight in superimposition
            heavy_coord = coord[[remote_index, remote_index, rem_rem_index]]
            stereo = 0
        elif n_heavy_bonds == 2:
            heavy_coord = coord[[heavy_indices[0], heavy_indices[1], i]]
            stereo = 0
        elif n_heavy_bonds == 3:
            heavy_coord = coord[heavy_indices]
            center = coord[i]
            # Determine the enantiomer of this stereocenter
            # For performance reasons, the result does not follow the
            # R/S nomenclature, but a custom -1/1 based one, which also
            # unambiguously identifies the enantiomer
            n = np.cross(heavy_coord[0] - center, heavy_coord[1] - center)
            stereo = int(np.sign(np.dot(heavy_coord[2] - center, n)))
        elif n_heavy_bonds == 4:
            # The fragment is irrelevant, as there is no bonded hydrogen
            # -> The fragment coord is the coord of the central atom
            # 4 times repeated
            heavy_coord = np.repeat(coord[np.newaxis, i, :], 3, axis=0)
            stereo = 0
        central_coord = coord[i]
        fragments[i] = (
            elements[i], charges[i], stereo, heavy_types,
            central_coord, heavy_coord, hydrogen_coord
        )
    return fragments


def _get_rotation_matrices(fixed, mobile):
    # TODO Use proper vectorization
    matrices = []
    for x, y in zip(fixed, mobile):
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            matrices.append(np.full((3, 3), np.nan))
            continue
        # Calculate covariance matrix
        cov = np.dot(x.T, y)
        v, s, w = np.linalg.svd(cov)
        # Remove possibility of reflected atom coordinates
        if np.linalg.det(v) * np.linalg.det(w) < 0:
            v[:,-1] *= -1
        matrix = np.dot(v,w)
        matrices.append(matrix)
    return np.array(matrices)


def _rotate(coord, matrices):
    # TODO Use proper vectorization
    rotated = []
    for c, matrix in zip(coord, matrices):
        rotated.append(np.dot(matrix, c.T).T)
    return np.array(rotated)

    coord = np.transpose(coord, (0, 2, 1))
    print(matrices.shape)
    print(coord.shape)
    rotated =  np.dot(matrices, coord)
    print(rotated.shape)
    return np.transpose(rotated, (0, 2, 1))