# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "hydride"
__author__ = "Patrick Kunzmann, Jacob Marcel Anter"
__all__ = ["relax_hydrogen", "EnergyFunction"]

from libc.math cimport sin, cos
cimport cython
cimport numpy as np

import numpy as np
import biotite.structure as struc


ctypedef np.uint8_t uint8
ctypedef np.int8_t int8
ctypedef np.int32_t int32
ctypedef np.float32_t float32


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

DEF DIELECTRIC_CONSTANT = 1.0


class EnergyFunction:
    r"""
    __init__(atom_array, atoms, relevant_mask, force_cutoff=10.0, partial_charges=None)

    This class represents the energy function used for relaxation.

    After construction, the energy and energy gradient dependent on the
    dihedral angles is obtained by calling the object, with the
    current atom coordinates.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,)
        The structure to calculate the energy values for.
    relevant_mask : ndarray, shape=(n,), dtype=bool
        This boolean mask specifies the atoms, whose positions are
        variable and hence whose interactions with other atoms are
        relevant.
        Usually this includes all hydrogen atoms with rotational
        freedom.
    force_cutoff : float, optional
        The force cutoff distance in Å.
        If the initial distance between two atoms exceeds this value,
        their interaction
        (:math:`V_\text{el}` and :math:`V_\text{nb}`) is not
        calculated.
    partial_charges : ndarray, shape=(n,), dtype=float, optional
        The partial charges for each atom used to calculate
        :math:`V_\text{el}`.
        By default the charges are calculated using
        :func:`biotite.structure.partial_charges()`.
    """

    def __init__(self, atoms, relevant_mask,
                 force_cutoff=10.0, partial_charges=None):
        cdef int i, j, k, pair_i
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
        cdef int32[:,:] bond_indices = atoms.bonds.get_all_bonds()[0]
        pair_i = 0
        for i in range(relevant_indices.shape[0]):
            atom_i = relevant_indices[i]
            for j in range(adj_indices.shape[1]):
                if adj_indices[i, j] == -1:
                    # Padding values
                    continue
                atom_j = adj_indices[i, j]
                # Do not include H-H-interaction duplicates
                # or an interaction of an atom to itself
                if atom_j <= atom_i:
                    continue
                # Do not include directly bonded atoms
                # Hydrogen atoms only have a single bond partner
                # -> it is sufficient to check the first entry
                if bond_indices[atom_i, 0] == atom_j:
                    continue
                interaction_pairs[pair_i, 0] = atom_i
                interaction_pairs[pair_i, 1] = atom_j
                pair_i += 1
        # Trim to correct size
        self._interaction_pairs = np.asarray(interaction_pairs)[:pair_i]


        # Calculate electrostatic parameters for interaction pairs
        cdef float32[:] charges 
        if partial_charges is None:
            charges = struc.partial_charges(atoms)
        else:
            charges = partial_charges.astype(np.float32, copy=False)
        cdef float32[:] elec_param  = np.zeros(pair_i, dtype=np.float32)
        for i in range(pair_i):
            elec_param[i] = 332.0673 / DIELECTRIC_CONSTANT * (
                charges[interaction_pairs[i, 0]] * 
                charges[interaction_pairs[i, 1]]
            )
        self._elec_param  = np.asarray(elec_param)


        # Calculate nonbonded parameters for interaction pairs
        nb_values = np.array(
            [NB_VALUES[element] for element in atoms.element],
            dtype=np.float32
        )
        cdef float32[:] radii = nb_values[:,0]
        cdef float32[:] scales = nb_values[:,1]
        cdef float32[:] r_6  = np.zeros(pair_i, dtype=np.float32)
        cdef float32[:] r_12 = np.zeros(pair_i, dtype=np.float32)
        cdef float32[:] eps  = np.zeros(pair_i, dtype=np.float32)
        for i in range(pair_i):
            r_6[i] = (0.5 * (
                radii[interaction_pairs[i, 0]] + 
                radii[interaction_pairs[i, 1]]
            ))**6
            r_12[i] = r_6[i]**2
            eps[i] = scales[interaction_pairs[i, 0]] * \
                     scales[interaction_pairs[i, 1]]
        self._r_6  = np.asarray(r_6)
        self._r_12 = np.asarray(r_12)
        self._eps  = np.sqrt(np.asarray(eps))


    def __call__(self, coord):
        distances = struc.index_distance(coord, self._interaction_pairs) \
                    .astype(np.float32, copy=False)
        
        #print(np.sum(self._elec_param / distances))
        #print(np.sum(self._eps * (
        #        -2 * self._r_6 / distances**6 + self._r_12 / distances**12
        #    )))
        #print()
        #exit()
        
        return np.sum(
            # Electrostatic interaction
            self._elec_param / distances
            # nonbonded interaction
            + self._eps * (
                -2 * self._r_6 / distances**6 + self._r_12 / distances**12
            )
        )


def relax_hydrogen(atoms, iteration_number=1):
    r"""
    relax_hydrogen(atoms, iteration_number=1)

    Optimize the hydrogen atom positions using gradient descent
    based on an electrostatic and a nonbonded potential.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,)
        The structure, whose hydrogen atoms should be relaxed.
    iteration_number : int, optional
        The number of gradient descent iterations.
        The runtime scales approximately linearly with the number of
        iterations.
    
    Returns
    -------
    relaxed_coord : ndarray, shape=(n,), dtype=np.float32
        The optimized coordinates.
        The coordinates for all heavy atoms remain unchanged.

    Notes
    -----
    The potential consists of the follwong terms:
    
    .. math::

        V = V_\text{el} + V_\text{nb}
        
        V_\text{el} = \epsilon_\text{el}
        \sum_i^\text{H}  \sum_j^\text{All}
        \frac{q_i q_j}{D_{ij}}

        E_\text{nb} = \epsilon_\text{nb}
        \sum_i^\text{H}  \sum_j^\text{All}
        \left(
            \frac{r_{ij}^{12}}{D_{ij}^{12}} - \frac{r_{ij}^6}{D_{ij}^6}
        \right)
    
    where :math:`D_{ij}` is the distance between the atoms :math:`i`
    and :math:`j` and :math:`r_{ij}` is calculated from the
    *Van-der-Waals* radii :math:`R` as

    .. math::

        r_{ij} = \frac{R_i + R_j}{2}.
    """
    cdef int i, j, mat_i

    coord = atoms.coord

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
    hydrogen_mask = np.zeros(atoms.array_length(), dtype=bool)
    for i, (
        central_atom_index, bonded_atom_index, is_free, hydrogen_indices
    ) in enumerate(rotatable_bonds):
        rotation_axes[i, 0] = coord[central_atom_index]
        rotation_axes[i, 1] = coord[bonded_atom_index]
        matrix_indices[hydrogen_indices] = i
        rotation_freedom[i] = is_free
        hydrogen_mask[hydrogen_indices] = True
    
    cdef float32[:,:,:] rot_mat_v = np.zeros(
        (len(rotatable_bonds), 3, 3), dtype=np.float32
    )
    cdef int32[:] matrix_indices_v = matrix_indices
    axes = rotation_axes[:, 1] - rotation_axes[:, 0]
    axes /= np.linalg.norm(axes, axis=-1)[:, np.newaxis]
    cdef float32[:,:] axes_v = axes
    cdef float32[:,:] support_v = rotation_axes[:, 0]


    energy_function = EnergyFunction(atoms, hydrogen_mask)
    prev_energy = energy_function(coord)
    print(prev_energy)

    np.random.seed(0)
    #seen_cord = np.zeros(coord.shape + (iteration_number,), dtype=np.float32)
    #energies = np.zeros(iteration_number, dtype=np.float32)
    prev_coord = atoms.coord.copy()
    new_coord = np.zeros(prev_coord.shape, dtype=np.float32)
    # Helper variable for the support-subtracted vector
    center_coord = np.zeros(3, dtype=np.float32)
    cdef float32[:,:] prev_coord_v
    cdef float32[:,:] new_coord_v
    cdef float32[:] center_coord_v = center_coord
    cdef float32[:] angles_v
    cdef float32 angle
    cdef float32 sin_a, cos_a, icos_a
    cdef float32 x, y, z
    for _ in range(iteration_number):
        # Generate new hydrogen conformation
        # Get random angles
        n_free_rotations = np.count_nonzero(rotation_freedom)
        angles = np.zeros(len(rotatable_bonds), dtype=np.float32)
        angles[rotation_freedom] = np.random.choice(
            np.array([-0.05, 0, 0.05]) * 2 * np.pi,
            size = n_free_rotations,
            p = (0.2, 0.6, 0.2)
        )
        angles[~rotation_freedom] = np.random.choice(
            (0, np.pi),
            size = len(rotatable_bonds) - n_free_rotations,
            p = (0.8, 0.2)
        ) 
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
        new_coord = prev_coord.copy()
        prev_coord_v = prev_coord
        new_coord_v = new_coord
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
                new_coord_v[i,j] = rot_mat_v[mat_i,j,0] * center_coord_v[0] + \
                                   rot_mat_v[mat_i,j,1] * center_coord_v[1] + \
                                   rot_mat_v[mat_i,j,2] * center_coord_v[2]
            # Readd support vector
            new_coord_v[i, 0] += support_v[mat_i, 0]
            new_coord_v[i, 1] += support_v[mat_i, 1]
            new_coord_v[i, 2] += support_v[mat_i, 2]

        # Calculate energy for new conformation
        energy = energy_function(new_coord)
        if energy < prev_energy:
            print(_, energy)
            prev_coord = new_coord
            prev_energy = energy

    return prev_coord


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