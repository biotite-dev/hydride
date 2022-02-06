# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "hydride"
__author__ = "Patrick Kunzmann"
__all__ = ["FragmentLibrary"]

from os.path import join, dirname, abspath
import warnings
import pickle
from biotite.structure.error import BadStructureError
from biotite.structure import BondType, displacement
import numpy as np


class FragmentLibrary:
    """
    A molecule fragment library for estimation of hydrogen positions.

    For each molecule added to the :class:`FragmentLibrary`,
    the molecule is split into fragments.
    Each fragment consists of

        - A central heavy atom,
        - bond order and position of its bonded heavy atoms and
        - and positions of bonded hydrogen atoms.
    
    The properties of the fragment (central atom element,
    central atom charge, order of connected bonds) are stored in
    a dictionary mapping these properties to heavy and hydrogen atom
    positions.

    If hydrogen atoms should be added to a target structure,
    the target structure is also split into fragments.
    Now the corresponding reference fragment in the library dictionary
    is accessed for each fragment.
    The corresponding atom coordinates of the reference fragment
    are superimposed [1]_ [2]_ onto the target fragment to obtain the
    hydrogen coordinates for the heavy atom.

    The constructor of this class creates an empty library.

    References
    ----------
    
    .. [1] W Kabsch,
       "A solution for the best rotation to relate two sets of vectors."
       Acta Cryst, 32, 922-923 (1976).
       
    .. [2] W Kabsch,
       "A discussion of the solution for the best rotation to relate
       two sets of vectors."
       Acta Cryst, 34, 827-828 (1978).
    """

    _std_library = None

    def __init__(self):
        self._frag_dict = {}
    

    @staticmethod
    def standard_library():
        """
        Get the standard :class:`FragmentLibrary`.
        The library contains fragments from all molecules in the
        *RCSB* *Chemical Component Dictionary*.

        Returns
        -------
        library : FragmentLibrary
            The standard library.
        """
        if FragmentLibrary._std_library is None:
            FragmentLibrary._std_library = FragmentLibrary()
            file_name = join(dirname(abspath(__file__)), "fragments.pickle")
            with open(file_name, "rb") as fragments_file:
                FragmentLibrary._std_library._frag_dict \
                    = pickle.load(fragments_file)
        return FragmentLibrary._std_library
    

    def add_molecule(self, molecule):
        """
        Add the fragments of a molecule to the library.

        Parameters
        ----------
        molecule : AtomArray
            A molecule, whose fragments should be added to the library.
            The structure must contain hydrogen atoms for each
            applicable heavy atom.
            The molecule must have an associated :class:`BondList`.
            The molecule must also include the *charge* annotation
            array, depicting the formal charge for each atom.
        """
        fragments = _fragment(molecule)
        for i, fragment in enumerate(fragments):
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
            self._frag_dict[
                (central_element, central_charge, stereo, tuple(bond_types))
            ] = (
                # Information about the origin of the fragment
                # # for debugging purposes 
                molecule.res_name[i], molecule.atom_name[i],
                # The interesting information
                centered_heavy_coord, centered_hydrogen_coord
            )
            if stereo != 0:
                # Also include the opposite enantiomer in the library
                # by reflecting the coordinates along an arbitrary axis
                stereo *= -1
                refl_centered_heavy_coord = centered_heavy_coord.copy()
                refl_centered_hydrogen_coord = centered_hydrogen_coord.copy()
                refl_centered_heavy_coord[..., 0] *= -1
                refl_centered_hydrogen_coord[..., 0] *= -1
                self._frag_dict[(
                    central_element, central_charge, stereo, tuple(bond_types)
                )] = (
                    molecule.res_name[i], molecule.atom_name[i],
                    refl_centered_heavy_coord, refl_centered_hydrogen_coord
                )


    def calculate_hydrogen_coord(self, atoms, mask=None, box=None):
        """
        Estimate the hydrogen coordinates for each atom in a given
        structure/molecule.

        Parameters
        ----------
        atoms : AtomArray, shape=(n,)
            The structure to get the hydrogen atom positions for.
            The structure must not contain any hydrogen atoms.
            The structure must have an associated :class:`BondList`.
            The structure must also include the *charge* annotation
            array, depicting the formal charge for each atom.
        mask : ndarray, shape=(n,), dtype=bool
            A boolean mask that is true for each heavy atom, where
            corresponsing hydrogen atom positions should be calculated.
            By default, hydrogen atoms are calculated for all applicable
            atoms.
        box : bool or array-like, shape=(3,3), dtype=float, optional
            If this parameter is set, periodic boundary conditions are
            taken into account (minimum-image convention), based on
            the box vectors given with this parameter.
            If `box` is set to true, the box vectors are taken from the
            ``box`` attribute of `atoms` instead.

        Returns
        -------
        hydrogen_coord : list of (ndarray, shape=(k,3), dtype=np.float32), length=n
            A list of hydrogen coordinates for each atom in the input
            `atoms`.
            *k* is the number of hydrogen atoms for this atom.
            Each atom, that is not included in the input `mask` or which
            has no fitting fragment in the library,
            has no corresponding hydrogen coordinates, i.e. *k* is 0.
        """
        if mask is None:
            mask = np.ones(atoms.array_length(), dtype=bool)

        # The target and reference heavy atom coordinates
        # for each fragment
        tar_frag_center_coord = np.zeros(
            (atoms.array_length(), 3), dtype=np.float32
        )
        tar_frag_heavy_coord = np.zeros(
            (atoms.array_length(), 3, 3), dtype=np.float32
        )
        ref_frag_heavy_coord = np.zeros(
            (atoms.array_length(), 3, 3), dtype=np.float32
        )
        # The amount of hydrogens varies for each fragment
        # -> padding with NaN
        # The maximum number of bond hydrogen atoms is 4
        ref_frag_hydrogen_coord = np.full(
            (atoms.array_length(), 4, 3), np.nan, dtype=np.float32
        )

        # Fill the coordinate arrays
        fragments = _fragment(atoms, mask)
        for i, fragment in enumerate(fragments):
            if fragment is None:
                # This atom is not in mask
                continue
            (
                central_element, central_charge, stereo, bond_types,
                center_coord, heavy_coord, _
            ) = fragment
            tar_frag_center_coord[i] = center_coord
            tar_frag_heavy_coord[i] = heavy_coord
            # The hydrogen_coord can be ignored:
            # In the target structure are no hydrogen atoms
            hit = self._frag_dict.get(
                (central_element, central_charge, stereo, tuple(bond_types))
            )
            if hit is None:
                warnings.warn(
                    f"Missing fragment for atom '{atoms.atom_name[i]}' "
                    f"at position {i}"
                )
            else:
                _, _, ref_heavy_coord, ref_hydrogen_coord = hit
                ref_frag_heavy_coord[i] = ref_heavy_coord
                ref_frag_hydrogen_coord[i, :len(ref_hydrogen_coord)] \
                    = ref_hydrogen_coord

        # Translate the target coordinates,
        # so the central heavy atom is at origin
        # This has already been done for the reference atoms
        # in the 'add_molecule()' method
        if box is not None:
            # Find shortest possible displacement vector for each heavy
            # atom according to minimum image convention
            if box is True:
                if atoms.box is None:
                    raise ValueError("Input structure has no associated box")
                box = atoms.box
            else:
                # Box vectors are given as array-like object
                box = np.asarray(box)
        tar_frag_heavy_coord = displacement(
            tar_frag_center_coord[:, np.newaxis, :], tar_frag_heavy_coord,
            box
        )

        # Get the rotation matrix required for superimposition of
        # the reference coord to the target coord 
        matrices = _get_rotation_matrices(
            tar_frag_heavy_coord, ref_frag_heavy_coord
        )
        # Rotate the reference hydrogen atoms, so they fit the
        # target heavy atoms
        tar_frag_hydrogen_coord = _rotate(ref_frag_hydrogen_coord, matrices)
        # Translate hydrogen atoms to the position of the
        # non-centered central heavy target atom
        tar_frag_hydrogen_coord += tar_frag_center_coord[:, np.newaxis, :]
        
        # Turn into list and remove NaN paddings
        tar_frag_hydrogen_coord = [
            # If the x-coordinate is NaN it is expected that
            # y and z are also NaN
            coord[~np.isnan(coord[:, 0])] for coord in tar_frag_hydrogen_coord
        ]

        return tar_frag_hydrogen_coord


def _fragment(atoms, mask=None):
    """
    Create fragments for the input structure/molecule.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,)
        The structure to be fragmented.
        The structure must have an associated :class:`BondList`.
        The structure must also include the *charge* annotation
        array, depicting the formal charge for each atom.
    mask : ndarray, shape=(n,), dtype=bool
        A boolean mask that is true for each heavy atom for which a
        fragment should be created.
    
    Returns
    -------
    fragments : list of tuple(str, int, int, ndarray, ndarray, ndarray), length=n
        The fragments.
        The tuple elements are

            #. the central atom element,
            #. the central atom charge,
            #. the enantiomer for stereocenter
               (``-1`` and ``1`` based on a custom nomenclature),
            #. the :class:`BondType` for each bonded heavy atom,
            #. the coordinates of the central atom,
            #. 3 coordinates of bonded heavy atoms (includes padding
               values, if there are not enough heavy atoms),
            #. the coordinates of bonded hydrogen atoms.
        
        ``None`` for each atom not included by the `mask`.
    """
    if mask is None:
        mask = np.ones(atoms.array_length(), dtype=bool)

    if atoms.bonds is None:
        raise BadStructureError(
            "The input structure must have an associated BondList"
        )

    fragments = [None] * atoms.array_length()
    
    all_bond_indices, all_bond_types = atoms.bonds.get_all_bonds()
    elements = atoms.element
    charges = atoms.charge
    coord = atoms.coord

    for i in range(atoms.array_length()):
        if not mask[i]:
            continue
        
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
        if (heavy_types == BondType.ANY).any():
            warnings.warn(
                f"Atom '{atoms.atom_name[i]}' in '{atoms.res_name[i]}' has an "
                f"undefined bond type and is ignored"
            )
            continue

        # Order the bonded atoms by their bond types
        # to remove atom order dependency in the matching step 
        order = np.argsort(heavy_types)
        heavy_indices = heavy_indices[order]
        heavy_types = heavy_types[order]

        hydrogen_mask = ~heavy_mask
        hydrogen_coord = coord[bond_indices[hydrogen_mask]]

        # Special handling of nitrogen as central atom:
        # There are cases where the free electron pair can form
        # a partial double bond.
        # Although the bond order is formally 1 in this case,
        # it would enforce planar hydrogen positionioning
        # Therefore, a partial double bond is handled as bond type 7
        if elements[i] == "N":
            for j, remote_index in enumerate(heavy_indices):
                if heavy_types[j] != 1:
                    # This handling only applies to single bonds
                    continue
                rem_bond_indices = all_bond_indices[remote_index]
                rem_bond_indices = rem_bond_indices[rem_bond_indices != -1]
                rem_bond_types = all_bond_types[remote_index]
                rem_bond_types = rem_bond_types[rem_bond_types != -1]
                for rem_rem_index, bond_type in zip(
                    rem_bond_indices, rem_bond_types
                ):
                    # If the adjacent atom has a double bond
                    # the partial double bond condition is fulfilled
                    if bond_type == BondType.AROMATIC_DOUBLE or \
                       bond_type == BondType.DOUBLE:
                            heavy_types[j] = 7

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
            rem_rem_index = rem_heavy_indices[0]
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
        else:
            warnings.warn(
                f"Atom '{atoms.atom_name[i]}' in "
                f"'{atoms.res_name[i]}' has more than 4 bonds to "
                f"heavy atoms ({n_heavy_bonds}) and is ignored"
            )
            heavy_coord = np.repeat(coord[np.newaxis, i, :], 3, axis=0)
            hydrogen_coord = np.zeros((0, 3), dtype=np.float32)
            stereo = 0
        central_coord = coord[i]
        fragments[i] = (
            elements[i], charges[i], stereo, heavy_types,
            central_coord, heavy_coord, hydrogen_coord
        )
    return fragments


def _get_rotation_matrices(fixed, mobile):
    """
    Get the rotation matrices to superimpose the given mobile
    coordinates into the given fixed coordinates, minimizing the RMSD.

    Uses the *Kabsch* algorithm.

    Parameters
    ----------
    fixed : ndarray, shape=(m,n,3), dtype=np.float32
        The fixed coordinates.
    mobile : ndarray, shape=(m,n,3), dtype=np.float32
        The mobile coordinates.
    
    Returns
    -------
    matrices : ndarray, shape=(m,3,3), dtype=np.float32
        The rotation matrices.
    """
    # Calculate cross-covariance matrices
    cov = np.sum(fixed[:,:,:,np.newaxis] * mobile[:,:,np.newaxis,:], axis=1)
    v, s, w = np.linalg.svd(cov)
    # Remove possibility of reflected atom coordinates
    reflected_mask = (np.linalg.det(v) * np.linalg.det(w) < 0)
    v[reflected_mask, :, -1] *= -1
    matrices = np.matmul(v, w)
    return matrices


def _rotate(coord, matrices):
    """
    Apply a rotation on given coordinates.

    Parameters
    ----------
    coord : ndarray, shape=(m,n,3), dtype=np.float32
        The coordinates.
    matrices : ndarray, shape=(m,3,3), dtype=np.float32
        The rotation matrices.
    """
    return np.transpose(
        np.matmul(matrices, np.transpose(coord, axes=(0, 2, 1))),
        axes=(0, 2, 1)
    )