# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "hydride"
__author__ = "Patrick Kunzmann, Jacob Marcel Anter"
__all__ = ["relax_hydrogen"]

from libc.math cimport sin, cos
cimport cython
cimport numpy as np

import warnings
import numpy as np
import biotite.structure as struc


ctypedef np.uint8_t uint8
ctypedef np.int8_t int8
ctypedef np.int32_t int32
ctypedef np.float32_t float32


cdef int ANY = struc.BondType.ANY
cdef int SINGLE = struc.BondType.SINGLE
cdef int DOUBLE = struc.BondType.DOUBLE
cdef int AROMATIC_DOUBLE = struc.BondType.AROMATIC_DOUBLE


# Values are taken from
# Rappé et al.
# "UFF, a Full Periodic Table Force Field
# for Molecular Mechanics and Molecular Dynamics Simulations"
# J Am Chem Soc, 114, 10024-10035 (1992)
# https://doi.org/10.1021/ja00051a040
cdef dict NB_VALUES = {
    "H" : (2.886, 0.044),
    "HE": (2.362, 0.056),
    "LI": (2.451, 0.025),
    "BE": (2.745, 0.085),
    "B" : (4.083, 0.180),
    "C" : (3.851, 0.105),
    "N" : (3.660, 0.069),
    "O" : (3.500, 0.060),
    "F" : (3.364, 0.050),
    "NE": (3.243, 0.042),
    "NA": (2.983, 0.030),
    "MG": (3.021, 0.111),
    "AL": (4.499, 0.505),
    "SI": (4.295, 0.402),
    "P": (4.147, 0.305),
    "S": (4.035, 0.274),
    "CL": (3.947, 0.227),
    "AR": (3.868, 0.185),
    "K" : (3.812, 0.035),
    "CA": (3.399, 0.238),
    "SC": (3.295, 0.019),
    "TI": (3.175, 0.017),
    "V" : (3.144, 0.016),
    "CR": (3.023, 0.015),
    "MN": (2.961, 0.013),
    "FE": (2.912, 0.013),
    "CO": (2.872, 0.014),
    "NI": (2.834, 0.015),
    "CU": (3.495, 0.005),
    "ZN": (2.763, 0.124),
    "GA": (4.383, 0.415),
    "GE": (4.280, 0.379),
    "AS": (4.230, 0.309),
    "SE": (4.205, 0.291),
    "BR": (4.189, 0.251),
    "KR": (4.141, 0.220),
    "RB": (4.114, 0.040),
    "SR": (3.641, 0.235),
    "Y" : (3.345, 0.072),
    "ZR": (3.124, 0.069),
    "NB": (3.165, 0.059),
    "MO": (3.052, 0.056),
    "TC": (2.998, 0.048),
    "RU": (2.963, 0.056),
    "RH": (2.929, 0.053),
    "PD": (2.899, 0.048),
    "AG": (3.148, 0.036),
    "CD": (2.848, 0.228),
    "IN": (4.463, 0.599),
    "SN": (4.392, 0.567),
    "SB": (4.420, 0.449),
    "TE": (4.470, 0.398),
    "I" : (4.500, 0.339),
    "XE": (4.404, 0.332),
    "CS": (4.517, 0.045),
    "BA": (3.703, 0.364),
    "LA": (3.522, 0.017),
    "CE": (3.556, 0.013),
    "PR": (3.606, 0.010),
    "ND": (3.575, 0.010),
    "PM": (3.547, 0.009),
    "SM": (3.520, 0.008),
    "EU": (3.493, 0.008),
    "GD": (3.368, 0.009),
    "TB": (3.451, 0.007),
    "DY": (3.428, 0.007),
    "HO": (3.409, 0.007),
    "ER": (3.391, 0.007),
    "TM": (3.374, 0.006),
    "YB": (3.355, 0.228),
    "LU": (3.640, 0.041),
    "HF": (3.141, 0.072),
    "TA": (3.170, 0.081),
    "W" : (3.069, 0.067),
    "RE": (2.954, 0.066),
    "OS": (3.120, 0.037),
    "IR": (2.840, 0.073),
    "PT": (2.754, 0.080),
    "AU": (3.293, 0.039),
    "HG": (2.705, 0.385),
    "TL": (4.347, 0.680),
    "PB": (4.297, 0.663),
    "BI": (4.370, 0.518),
    "PO": (4.709, 0.325),
    "AT": (4.750, 0.284),
    "RN": (4.765, 0.248),
    "FR": (4.900, 0.050),
    "RA": (3.677, 0.404),
    "AC": (3.478, 0.033),
    "TH": (3.396, 0.026),
    "PA": (3.424, 0.022),
    "U" : (3.395, 0.022),
    "NP": (3.424, 0.019),
    "PU": (3.424, 0.016),
    "AM": (3.381, 0.014),
    "CM": (3.326, 0.013),
    "BK": (3.339, 0.013),
    "CF": (3.313, 0.013),
    "ES": (3.299, 0.012),
    "FM": (3.286, 0.012),
    "MD": (3.274, 0.011),
    "NO": (3.248, 0.011),
    "LW": (3.236, 0.011),
}

HBOND_ELEMENTS = ("N", "O", "F", "S", "CL")

DEF HBOND_FACTOR = 0.79


class MinimumFinder:
    r"""
    __init__(self, atoms, groups, partial_charges=None, force_cutoff=10.0)

    This class evaluates an energy function based on given atom
    coordinates and selects coordinates that perform better with respect
    to the energy function than previous ones.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,)
        The structure to calculate the energy values for.
        The *topology* of this :class:`AtomArray`
        (bonds, charges, elements, etc.) is used to calculate the
        interacting atoms pairs as well as the force field parameters.
        The coordinates are used as initial reference set of positions
        for :meth:`select_minimum()`.
    groups : ndarray, shape=(n,), dtype=np.int32
        Groups of hydrogen atoms in `atoms`, whose positions are changed
        via the same rotatable bond.
        Each positive integer (including ``0``) represents one group,
        i.e. all atoms with the same group integer are connected to the
        same heavy atom.
        ``-1`` indicates atoms that cannot be rotated
        (haevy atoms, or non-rotatable hydrogen atoms).
    partial_charges : ndarray, shape=(n,), dtype=float, optional
        The partial charges for each atom used to calculate
        :math:`V_\text{el}`.
        By default the charges are calculated using
        :func:`biotite.structure.partial_charges()`.
    force_cutoff : float, optional
        The force cutoff distance in Å.
        If the initial distance between two atoms exceeds this value,
        their interaction
        (:math:`V_\text{el}` and :math:`V_\text{nb}`) is not
        calculated.
    
    See also
    --------
    relax_hydrogen
        For explanation of the energy function.
    """

    def __init__(self, atoms, int32[:] groups, partial_charges=None,
                 float32 force_cutoff=10.0):
        cdef int i, j, k, pair_i
        cdef int32 atom_i, atom_j, bonded_atom_i
        
        if atoms.array_length() != groups.shape[0]:
            raise ValueError(
                f"There are {atoms.array_length()} atoms, "
                f"but {groups.shape[0]} group indicators"
            )
        
        self._prev_group_energies = None
        self._prev_coord = atoms.coord.copy()
        self._groups = np.asarray(groups)
        self._n_groups = np.max(np.asarray(groups)) + 1
        if self._n_groups < 1:
            raise ValueError("Expected at least one movable group")


        # Find proximate atoms for calcualtion of interacting pairs
        cell_list = struc.CellList(
            atoms, cell_size=force_cutoff
        )
        relevant_mask = (np.asarray(groups) != -1)
        cdef int32[:] relevant_indices = np.where(relevant_mask)[0] \
                                         .astype(np.int32)
        cdef int32[:,:] adj_indices = cell_list.get_atoms(
            atoms.coord[np.asarray(relevant_indices)],
            radius=force_cutoff
        )

        
        cdef int32[:,:] bond_indices = atoms.bonds.get_all_bonds()[0]


        # Calculate which pairs of atoms interact with each other
        # with respect to the cutoff distance
        # and to which movable atom group those pairs belong to
        # Array allocation with pessimistic size assumption.
        # In the worst case, there are no padded -1 values
        cdef int32[:,:] interaction_pairs = np.zeros(
            (adj_indices.shape[0] * adj_indices.shape[1], 2),
            dtype=np.int32
        )
        cdef int32[:] interaction_groups = np.zeros(
            (adj_indices.shape[0] * adj_indices.shape[1]),
            dtype=np.int32
        )
        cdef uint8[:] mask = relevant_mask.astype(np.uint8)
        pair_i = 0
        for i in range(relevant_indices.shape[0]):
            atom_i = relevant_indices[i]
            for j in range(adj_indices.shape[1]):
                atom_j = adj_indices[i, j]
                if atom_j == -1:
                    # Padding values
                    continue
                # Do not include interaction between hydrogen atoms from
                # the same group
                if groups[atom_i] == groups[atom_j]:
                    continue
                # Do not include interaction to directly bonded atoms
                # Hydrogen atoms only have a single bond partner
                # -> it is sufficient to check the first entry
                if bond_indices[atom_i, 0] == atom_j:
                    continue
                interaction_pairs[pair_i, 0] = atom_i
                interaction_pairs[pair_i, 1] = atom_j
                interaction_groups[pair_i] = groups[atom_i]
                pair_i += 1
        # Trim to correct size
        self._interaction_pairs  = np.asarray(interaction_pairs )[:pair_i]
        self._interaction_groups = np.asarray(interaction_groups)[:pair_i]
        # H-H-Interaction are included twice in the standard interaction
        # pairs, which would give wrong global energy
        # -> Deduplicate these pairs 
        self._dedup_interaction_mask = self._deduplicate_pairs()


        # Calculate electrostatic parameters for interaction pairs
        cdef float32[:] charges
        if partial_charges is None:
            charges = struc.partial_charges(atoms)
        else:
            charges = partial_charges.astype(np.float32, copy=False)
        cdef float32[:] elec_param  = np.zeros(pair_i, dtype=np.float32)
        for i in range(pair_i):
            elec_param[i] = 332.0673 * (
                charges[interaction_pairs[i, 0]] * 
                charges[interaction_pairs[i, 1]]
            )
        self._elec_param  = np.asarray(elec_param)


        # Calculate nonbonded (LJ) parameters for interaction pairs
        nb_values = np.array(
            [NB_VALUES[element] for element in atoms.element],
            dtype=np.float32
        )
        cdef float32[:] radii = nb_values[:,0]
        cdef float32[:] scales = nb_values[:,1]
        cdef float32[:] r_6  = np.zeros(pair_i, dtype=np.float32)
        cdef float32[:] r_12 = np.zeros(pair_i, dtype=np.float32)
        cdef float32[:] sq_eps  = np.zeros(pair_i, dtype=np.float32)
        # Special handlding for potential hydrogen bonds:
        # If hydrogen in bound to a donor element the optimal distance 
        # to the possible acceptor is decreased
        cdef uint8[:] hbond_mask = np.isin(atoms.element, HBOND_ELEMENTS) \
                                   .astype(np.uint8)
        cdef float32 hbond_factor
        # Calculate parameters for each H-X interaction
        for i in range(pair_i):
            atom_i = interaction_pairs[i, 0]
            atom_j = interaction_pairs[i, 1]
            # Hydrogen atoms only have a single bond partner
            # -> it is sufficient to check the first entry
            bonded_atom_i = bond_indices[atom_i, 0]
            # Check if hydrogen has a bonded atom and this atom is a
            # hydrogen bond donor
            # and if the interaction partner is a hydrogen acceptor
            if bonded_atom_i != -1 and hbond_mask[bonded_atom_i] \
               and hbond_mask[atom_j]:
                    # Potential hydrogen bond interaction
                    # -> apply correction factor
                    hbond_factor = HBOND_FACTOR
            else:
                    hbond_factor = 1.0
            r_6[i] = (hbond_factor * 0.5 * (
                radii[atom_i] + 
                radii[atom_j]
            ))**6
            r_12[i] = r_6[i]**2
            sq_eps[i] = scales[atom_i] * scales[atom_j]
        self._r_6  = np.asarray(r_6)
        self._r_12 = np.asarray(r_12)
        self._eps  = np.sqrt(np.asarray(sq_eps))
    

    def calculate_global_energy(self, coord):
        """
        calculate_global_energy(self, coord)

        Calculate the global result of the energy function.

        Parameters
        ----------
        coord : ndarray, shape=(n,3), dtype=np.float32
            The coordinates of the atoms to calculate the energy for.

        Returns
        -------
        energies : float
            The energy calculated from the input atom coordinates.
        """
        energies = self._calculate__energies(coord, coord)
        return np.sum(energies[self._dedup_interaction_mask])


    def select_minimum(self, float32[:,:] next_coord):
        """
        select_minimum(self, next_coord)

        From a given set of updated coordinates select those
        coordinates, that decrease the result of the energy function
        compared to the current set of coordinates.

        For each atom group, there are two available choices:
        Either accepting the given `next_coord` or rejecting them and
        keeping the current coordinates.
        The current coordinates are the accepted coordinates from
        the last call of :meth:`select_minimum()` or the input
        coordinates of the constructor, if :meth:`select_minimum()`
        was not called yet.

        Parameters
        ----------
        next_coord : ndarray, shape=(n,3), dtype=np.float32
            Updated coordinates to choose from.

        Returns
        -------
        accepted_coord : ndarray, shape=(n,3), dtype=np.float32
            The accepted coordinates.
            For each atom group it contains either the respective
            coordinates from `next_coord` or the current coordinates.
        global_energy : float
            The global energy of the accepted conformation, according
            to the underlying energy function.
        any_accepted : bool
            True, if any of the coordinates from `next_coord` were
            accepted, false otherwise.
        """
        cdef int i

        if self._prev_group_energies is None:
            prev_energies = self._calculate_energies(
                self._prev_coord, self._prev_coord
            )
            self._prev_group_energies = self._sum_for_groups(prev_energies)
        next_energies = self._calculate_energies(
            self._prev_coord, np.asarray(next_coord)
        )
        next_group_energies = self._sum_for_groups(next_energies)

        cdef uint8[:] accept_next = (
            next_group_energies < self._prev_group_energies
        ).astype(np.uint8)
        cdef int32[:] groups = self._groups
        # The accepted next coordinates are the new prev coordinates
        # for the next function call
        cdef float32[:,:] prev_coord = self._prev_coord
        for i in range(prev_coord.shape[0]):
            if accept_next[groups[i]]:
                prev_coord[i, 0] = next_coord[i, 0]
                prev_coord[i, 1] = next_coord[i, 1]
                prev_coord[i, 2] = next_coord[i, 2]
        
        # Prepare the reference energies for the next call of
        # 'select_minimum()'
        # Do this after this call, instead of the beginning of the next
        # call, to be able to return the global energies for this step
        prev_energies = self._calculate_energies(
            self._prev_coord, self._prev_coord
        )
        self._prev_group_energies = self._sum_for_groups(prev_energies)
        global_energy = np.sum(prev_energies[self._dedup_interaction_mask])
        
        return self._prev_coord, global_energy, np.asarray(accept_next).any()
    

    def _calculate_energies(self, prev_coord, next_coord):
        """
        Calculate the result of the energy function for each group,
        based on the given atom coordinates.

        Parameters
        ----------
        prev_coord, next_coord : ndarray, shape=(n,3), dtype=np.float32
            For a group *i* each pairwise atom distance required for the
            energy function is calculated between the `next_coord` of
            an atom in group *i* and the `prev_coord` of the respective
            interacting atoms.
            To calculate the group energies for a single conformation,
            `prev_coord` and `next_coord` can be given the same array.

        Returns
        -------
        group_energies : ndarray, shape=(g,), dtype=np.float32
            The energies for each group.
            The group integer is used to get the energy for the
            corresponding atom group.
        """
        cdef int i
        
        diff = next_coord[self._interaction_pairs[:,0]] - \
               prev_coord[self._interaction_pairs[:,1]]
        distances = np.sqrt((diff*diff).sum(axis=-1)) \
                    .astype(np.float32, copy=False)
                    
        return self._pairwise_energy_function(distances)
        
        
    def _sum_for_groups(self, float32[:] values):
        """
        Sum up values (e.g. energies) for interaction pairs into
        values for the respective atom groups.

        Parameters
        ----------
        values : ndarray, shape=(p,), dtype=np.float32
            The input values.
        
        Returns
        -------
        values : ndarray, shape=(g,), dtype=np.float32
            The summed values.
        """
        cdef int i
        
        cdef float32[:] group_values = np.zeros(
            self._n_groups, dtype=np.float32
        )
        cdef int32[:] groups = self._interaction_groups
        for i in range(values.shape[0]):
            group_values[groups[i]] += values[i]
        
        return np.asarray(group_values)


    def _pairwise_energy_function(self, distances):
        """
        The energy function, containing an electrostatic and a
        non-bonded interaction term.

        Parameters
        ----------
        distances : ndarray, shape(p,), dtype=np.float32
            The distances for each pair of interacting atoms.
        
        Returns
        -------
        energy : ndarray, shape(p,), dtype=np.float32
            The energy of each interaction pair.
        """
        return (
            # Electrostatic interaction
            self._elec_param / distances
            # nonbonded interaction
            + self._eps * (
                -2 * self._r_6 / distances**6 + self._r_12 / distances**12
            )
        )
    

    def _deduplicate_pairs(self):
        """
        Remove duplicate interaction pairs.

        Returns
        -------
        mask : ndarray, shape(p,), dtype=bool
            A boolean mask, that is true for all elements that should
            persist.
        """
        cdef int i

        cdef int32[:,:] dup_pairs = self._interaction_pairs
        cdef int32[:] groups = self._groups

        cdef uint8[:] mask = np.zeros(
            dup_pairs.shape[0], dtype=np.uint8
        )
        for i in range(dup_pairs.shape[0]):
            # Only use one atom pair order
            # for the interaction of two hydrogen atoms
            if dup_pairs[i, 1] > dup_pairs[i, 0] \
               or groups[dup_pairs[i, 1]] == -1:
                    mask[i] = True
        return np.asarray(mask).astype(bool)



def relax_hydrogen(atoms, iterations=None, angle_increment=np.deg2rad(10),
                   return_trajectory=False, return_energies=False,
                   partial_charges=None):
    r"""
    relax_hydrogen(atoms, iterations=None, angle_increment=np.deg2rad(10), return_trajectory=False, return_energies=False, partial_charges=None)

    Optimize the hydrogen atom positions by rotating about terminal
    bonds.
    The relaxation uses hill climbing based on an electrostatic and
    a nonbonded potential [1]_.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,)
        The structure, whose hydrogen atoms should be relaxed.
        Must have an associated :class:`BondList`.
        Note that :attr:`BondType.ANY` bonds are not considered
        for rotation
    iterations : int, optional
        Limit the number of relaxation iterations.
        The runtime scales approximately linearly with the number of
        iterations.
        By default, the relaxation runs until a local optimum has been
        found, i.e. the hydrogen coordinates do not change anymore.
        If this parameter is set, the relaxation terminates before this
        point after the given number of interations.
    angle_increment : float, optional
        The angle in radians by which a bond can be rotated in each
        iteration.
        Smaller values increase the accurucy, but increase the number of
        required iterations.
    return_trajectory : bool, optional
        If set to true, the resulting coordinates for each relaxation
        step are returned, instead of the coordinates of the final
        step.
    return_energies : bool, optional
        If set to true, also the calculated energy for the conformation
        of each relaxation step is returned.
        This parameter can be useful for monitoring and debugging.
    
    Returns
    -------
    relaxed_coord : ndarray, shape=(n,) or shape=(m,n), dtype=np.float32
        The optimized coordinates.
        The coordinates for all heavy atoms remain unchanged.
        if `return_trajectory` is set to true, not only the coordinates
        after relaxation, but the coordinates from each step are
        returned.
    energies : ndarray, shape=(m,), dtype=np.float32
        The energy for each step.
        Only returned, if `return_energies` is set to true

    Notes
    -----
    The potential consists of the follwong terms:
    
    .. math::

        V = V_\text{el} + V_\text{nb}
        
        V_\text{el} = 332.067
        \sum_i^\text{H}  \sum_j^\text{All}
        \frac{q_i q_j}{D_{ij}}

        E_\text{nb} = \epsilon_{ij}
        \sum_i^\text{H}  \sum_j^\text{All}
        \left(
            \frac{r_{ij}^{12}}{D_{ij}^{12}} - 2\frac{r_{ij}^6}{D_{ij}^6}
        \right)
    
    where :math:`D_{ij}` is the distance between the atoms :math:`i`
    and :math:`j`. :math:`\epsilon_{ij}` and :math:`r_{ij}` are the
    well depth and optimal distance between these atoms, respectively,
    and are calculated as

    .. math::

         \epsilon_{ij} = \sqrt{ \epsilon_i  \epsilon_j},
         
         r_{ij} = \frac{r_i + r_j}{2}.
    
    :math:`\epsilon_{i/j}` and :math:`r_{i/j}` are taken from the
    *Universal Force Field* [1]_ [2]_.

    References
    ----------
    
    .. [1] AK Rappé, CJ Casewit, KS Colwell, WA Goddard III and WM Skiff,
       "UFF, a full periodic table force field for molecular mechanics
       and molecular dynamics simulations."
       J Am Chem Soc, 114, 10024-10035 (1992).
   
    .. [2] T Ogawa and T Nakano,
       "The Extended Universal Force Field (XUFF): Theory and applications."
       CBIJ, 10, 111-133 (2010)
    """
    cdef int i, j, mat_i

    coord = atoms.coord

    if iterations is not None and iterations < 0:
        raise ValueError("The number of iterations must be positive")

    rotatable_bonds = _find_rotatable_bonds(atoms)
    if len(rotatable_bonds) == 0:
        # No bond to optimize
        return coord.copy()

    rotation_axes = np.zeros(
        (len(rotatable_bonds), 2, 3), dtype=np.float32
    )
    matrix_indices = np.full(
        atoms.array_length(), -1, dtype=np.int32
    )
    rotation_freedom = np.zeros(
        len(rotatable_bonds), dtype=bool
    )
    for i, (
        central_atom_index, bonded_atom_index, is_free, hydrogen_indices
    ) in enumerate(rotatable_bonds):
        rotation_axes[i, 0] = coord[central_atom_index]
        rotation_axes[i, 1] = coord[bonded_atom_index]
        matrix_indices[hydrogen_indices] = i
        rotation_freedom[i] = is_free
    
    cdef float32[:,:,:] rot_mat_v = np.zeros(
        (len(rotatable_bonds), 3, 3), dtype=np.float32
    )
    cdef int32[:] matrix_indices_v = matrix_indices
    axes = rotation_axes[:, 1] - rotation_axes[:, 0]
    axes /= np.linalg.norm(axes, axis=-1)[:, np.newaxis]
    cdef float32[:,:] axes_v = axes
    cdef float32[:,:] support_v = rotation_axes[:, 0]


    minimum_finder = MinimumFinder(atoms, matrix_indices, partial_charges)

    if return_trajectory:
        trajectory = []
    energies = []
    prev_coord = atoms.coord.copy()
    next_coord = np.zeros(prev_coord.shape, dtype=np.float32)
    # Helper variable for the support-subtracted vector
    center_coord = np.zeros(3, dtype=np.float32)
    # Variables for saving whether any changes were accepted in a step
    cdef bint curr_accepted, prev_accepted = True
    cdef float curr_energy, prev_energy = np.nan
    cdef float32[:,:] prev_coord_v
    cdef float32[:,:] next_coord_v
    cdef float32[:] center_coord_v = center_coord
    cdef float32[:] angles_v
    cdef float32 angle
    cdef float32 sin_a, cos_a, icos_a
    cdef float32 x, y, z
    n = 0
    # Loop terminates via break if result converges
    # or iteration number is exceeded
    while True:
        if iterations is not None and n >= iterations:
            break
        
        # Generate next hydrogen conformation
        n_free_rotations = np.count_nonzero(rotation_freedom)
        angles = np.zeros(len(rotatable_bonds), dtype=np.float32)
        # Rotate bonds with rotation freedom
        # alternatingly either clockwise or counterclockwise
        angles[rotation_freedom] = angle_increment if n % 2 else -angle_increment
        # There is only one way
        # to rotate a bond without rotation freedom
        angles[~rotation_freedom] = np.pi
        angles_v = angles
        # Calculate rotation matrices for these angles
        for mat_i in range(angles_v.shape[0]):
            x = axes_v[mat_i, 0]
            y = axes_v[mat_i, 1]
            z = axes_v[mat_i, 2]
            angle = angles_v[mat_i]
            sin_a = sin(angle)
            cos_a = cos(angle)
            icos_a = 1 - cos_a
            # The roation matrix:
            #  cos_a + icos_a*x**2  icos_a*x*y - z*sin_a  icos_a*x*z + y*sin_a
            # icos_a*x*y + z*sin_a   cos_a + icos_a*y**2  icos_a*y*z - x*sin_a
            # icos_a*x*z - y*sin_a  icos_a*y*z + x*sin_a   cos_a + icos_a*z**2
            rot_mat_v[mat_i, 0, 0] = cos_a + icos_a*x**2
            rot_mat_v[mat_i, 0, 1] = icos_a*x*y - z*sin_a
            rot_mat_v[mat_i, 0, 2] = icos_a*x*z + y*sin_a
            rot_mat_v[mat_i, 1, 0] = icos_a*x*y + z*sin_a
            rot_mat_v[mat_i, 1, 1] = cos_a + icos_a*y**2
            rot_mat_v[mat_i, 1, 2] = icos_a*y*z - x*sin_a
            rot_mat_v[mat_i, 2, 0] = icos_a*x*z - y*sin_a
            rot_mat_v[mat_i, 2, 1] = icos_a*y*z + x*sin_a
            rot_mat_v[mat_i, 2, 2] = cos_a + icos_a*z**2
        # Apply matrices
        next_coord = prev_coord.copy()
        prev_coord_v = prev_coord
        next_coord_v = next_coord
        for i in range(matrix_indices_v.shape[0]):
            mat_i = matrix_indices_v[i]
            if mat_i == -1:
                # Atom should not be rotated
                continue
            # Subtract support vector
            center_coord_v[0] = prev_coord_v[i, 0] - support_v[mat_i, 0]
            center_coord_v[1] = prev_coord_v[i, 1] - support_v[mat_i, 1]
            center_coord_v[2] = prev_coord_v[i, 2] - support_v[mat_i, 2]
            # Rotate using the matrix
            # Iterate over each vector row
            # to perform the matrix-vector multiplication
            for j in range(3):
                next_coord_v[i,j] = rot_mat_v[mat_i,j,0] * center_coord_v[0] + \
                                   rot_mat_v[mat_i,j,1] * center_coord_v[1] + \
                                   rot_mat_v[mat_i,j,2] * center_coord_v[2]
            # Readd support vector
            next_coord_v[i, 0] += support_v[mat_i, 0]
            next_coord_v[i, 1] += support_v[mat_i, 1]
            next_coord_v[i, 2] += support_v[mat_i, 2]

        
        # Calculate next conformation based on energy
        curr_coord, curr_energy, curr_accepted \
            = minimum_finder.select_minimum(next_coord)
        if not curr_accepted and not prev_accepted:
            # No coordinates were accepted from the current and previous
            # step -> Relaxation converged -> Early termination
            # If only no coordinates from the current were accepted,
            # this would not be sufficient due to the alternating
            # bond rotation (see above)
            break
        if not np.isnan(prev_energy) and curr_energy > prev_energy:
            # The relaxation algorithm allows the case, that the energy
            # oscillates between two almost-minimum energies due to its 
            # discrete nature and so convergence is never reached
            # To prevent this, the relaxation terminates, if the energy
            # of the accepted is higher than the one before
            break
        prev_coord = curr_coord
        prev_energy = curr_energy
        prev_accepted = curr_accepted
        if return_trajectory:
            trajectory.append(prev_coord.copy())
        energies.append(curr_energy)
        
        n += 1

    if return_trajectory:
        return_coord = np.stack(trajectory)
    else:
        return_coord = prev_coord
    if return_energies:
        return return_coord, np.array(energies)
    else:
        return return_coord


def _find_rotatable_bonds(atoms):
    """
    Identify rotatable bonds between two heavy atoms, where one atom
    has only one heavy bond partner and one or multiple hydrogen
    partners.
    These bonds are used to create new conformations for the relaxation
    algorithm.

    Parameters
    ----------
    atoms : AtomArray
        The structure to find rotatable bonds in.
    
    Returns
    -------
    rotatable_bonds : list of tuple(int, int, bool, ndarray)
        The rotatable bonds.
        The tuple elements are

            #. Atom index of heavy atom with bond hydrogen atoms.
            #. Atom index of bonded heavy atom.
            #. If false, the bond can only be rotated by 180°.
            #. Atom indices of bonded hydrogen atoms
    """
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
        
        # There must be at least one bonded hydrogen atom
        if h_i == 0:
            is_rotatable = False

        # The rotation freedom might be restricted to 180 degrees
        # in a (partially) double bonded heavy atom
        if is_rotatable:
            if bonded_heavy_btype == SINGLE:
                is_free = True
                # Check for partial double bonds
                # Nitrogen is the only relevant atom for this case
                if is_nitrogen[i]:
                    for j in range(all_bond_indices.shape[1]):
                        rem_index = all_bond_indices[bonded_heavy_index, j]
                        if rem_index == -1:
                            # padding value
                            break
                        rem_btype = all_bond_types[bonded_heavy_index, j]
                        # If the adjacent atom has a double bond
                        # the condition is fulfilled
                        if rem_btype == AROMATIC_DOUBLE or rem_btype == DOUBLE:
                            is_free = False
                            break
            elif bonded_heavy_btype == DOUBLE:
                is_free = False
            elif bonded_heavy_btype == ANY:
                warnings.warn(
                    "The given structure contains 'BondType.ANY' bonds, "
                    "which cannot be rotated about"
                )
                is_rotatable = False
                is_free = False
            else:
                # Triple bond etc. -> no rotational freedom
                is_rotatable = False
                is_free = False
        else:
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