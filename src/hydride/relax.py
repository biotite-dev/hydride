# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "hydride"
__author__ = "Patrick Kunzmann, Jacob Marcel Anter"
__all__ = ["relax_hydrogen"]

import warnings
from dataclasses import dataclass
import biotite.structure as struc
import jax
import jax.numpy as jnp
import numpy as np
import optax

# Values are taken from
# Rappé et al.
# "UFF, a Full Periodic Table Force Field
# for Molecular Mechanics and Molecular Dynamics Simulations"
# J Am Chem Soc, 114, 10024-10035 (1992)
# https://doi.org/10.1021/ja00051a040
NB_VALUES = {
    "H": (2.886, 0.044),
    "HE": (2.362, 0.056),
    "LI": (2.451, 0.025),
    "BE": (2.745, 0.085),
    "B": (4.083, 0.180),
    "C": (3.851, 0.105),
    "N": (3.660, 0.069),
    "O": (3.500, 0.060),
    "F": (3.364, 0.050),
    "NE": (3.243, 0.042),
    "NA": (2.983, 0.030),
    "MG": (3.021, 0.111),
    "AL": (4.499, 0.505),
    "SI": (4.295, 0.402),
    "P": (4.147, 0.305),
    "S": (4.035, 0.274),
    "CL": (3.947, 0.227),
    "AR": (3.868, 0.185),
    "K": (3.812, 0.035),
    "CA": (3.399, 0.238),
    "SC": (3.295, 0.019),
    "TI": (3.175, 0.017),
    "V": (3.144, 0.016),
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
    "Y": (3.345, 0.072),
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
    "I": (4.500, 0.339),
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
    "W": (3.069, 0.067),
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
    "U": (3.395, 0.022),
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
HBOND_FACTOR = 0.79


@dataclass
class InteractionPairs:
    """
    Parameters for the interaction energy calculation.

    Attributes
    ----------
    heavy_indices, hydrogen_indices : ndarray, shape=(n,), dtype=int
        The indices of the interacting atoms.
        Contains corresponding indices of heavy and hydrogen atoms, respectively.
    electrostatic_factors : ndarray, shape=(n,), dtype=float
        The product of the partial charges of the interacting atoms
        multiplied with the electrostatic constant.
        Can be directly used as factor in front of the distance.
    epsilon_2 : ndarray, shape=(n,), dtype=float
        Epsilon squared value for the Lennard-Jones potential.
    radii_6, radii_12 : ndarray, shape=(n,), dtype=float
        The VdW-radii to the power of 6 and 12 for the Lennard-Jones potential.
    """

    heavy_indices: ...
    hydrogen_indices: ...
    electrostatic_factors: ...
    epsilon_2: ...
    radii_6: ...
    radii_12: ...

    @staticmethod
    def from_structure(
        atoms,
        groups,
        partial_charges=None,
        force_cutoff=10.0,
    ):
        r"""
        Create a :class:`InteractionPairs` instance from a structure.

        Parameters
        ----------
        atoms : AtomArray, shape=(n,)
            The structure to calculate the energy term parameters for.
            The *topology* of this :class:`AtomArray`
            (bonds, charges, elements, etc.) is used to calculate the
            interacting atoms pairs as well as the force field parameters.
        groups : ndarray, shape=(n,), dtype=np.int32
            Groups of hydrogen atoms in `atoms`, whose positions are changed
            via the same rotatable bond.
            Each positive integer (including ``0``) represents one group,
            i.e. all atoms with the same group integer are connected to the
            same heavy atom.
            ``-1`` indicates atoms that cannot be rotated
            (heavy atoms, or non-rotatable hydrogen atoms).
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

        Returns
        -------
        pairs : InteractionPairs
            The created :class:`InteractionPairs`
        """
        bond_indices, _ = atoms.bonds.get_all_bonds()
        # Hydrogen atoms only have a single bond partner
        # -> it is sufficient to check the first entry
        hydrogen_bond_partner_indices = bond_indices[:, 0]
        heavy_indices, hydrogen_indices = _get_interaction_pairs(
            atoms, groups, hydrogen_bond_partner_indices, force_cutoff
        )

        # Calculate electrostatic parameters for interaction pairs
        if partial_charges is None:
            partial_charges = struc.partial_charges(atoms)
        # Handle NaN charges as neutral charges
        partial_charges[np.isnan(partial_charges)] = 0
        electrostatic_factors = jnp.asarray(
            332.0673
            * (
                partial_charges[heavy_indices[:, 0]]
                * partial_charges[hydrogen_indices[:, 1]]
            ),
            dtype=np.float32,
        )

        # Calculate LJ parameters for interaction pairs
        nb_values = np.array(
            [NB_VALUES.get(element, (np.nan, np.nan)) for element in atoms.element],
            dtype=np.float32,
        )
        radii = nb_values[:, 0]
        scales = nb_values[:, 1]
        # Special handlding for potential hydrogen bonds:
        # If hydrogen in bound to a donor element the optimal distance
        # to the possible acceptor is decreased
        hbond_element_mask = np.isin(atoms.element, HBOND_ELEMENTS)
        bonded_heavy_indices = hydrogen_bond_partner_indices[hydrogen_indices]
        # Consider a pair as a hydrogen bond,
        # if both involved heavy atoms are hydrogen bond donor/acceptor elements
        hbond_mask = (
            bonded_heavy_indices
            != -1
            & hbond_element_mask[bonded_heavy_indices]
            & hbond_element_mask[hydrogen_indices]
        )
        hbond_factor = np.where(hbond_mask, HBOND_FACTOR, 1.0)

        radii_6 = jnp.asarray(
            (hbond_factor * 0.5 * (radii[heavy_indices] + radii[hydrogen_indices]))
            ** 6,
            dtype=np.float32,
        )
        radii_12 = radii_6**2
        epsilon_2 = jnp.sqrt(
            scales[heavy_indices] * scales[hydrogen_indices], dtype=np.float32
        )

        return InteractionPairs(
            jnp.asarray(heavy_indices, dtype=np.int32),
            jnp.asarray(hydrogen_indices, dtype=np.int32),
            electrostatic_factors=electrostatic_factors,
            epsilon_2=epsilon_2,
            radii_6=radii_6,
            radii_12=radii_12,
        )


def relax_hydrogen(
    atoms,
    iterations=None,
    mask=None,
    return_trajectory=False,
    return_energies=False,
    partial_charges=None,
    optimizer=None,
):
    # Copy to avoid altering the coordinates of the input
    atoms = atoms.copy()
    coord = atoms.coord

    if iterations is not None and iterations < 0:
        raise ValueError("The number of iterations must be positive")

    rotatable_bonds = _find_rotatable_bonds(atoms, mask)
    if len(rotatable_bonds) == 0:
        # No bond to relax -> Can return without relaxation
        if return_trajectory:
            return_coord = coord[np.newaxis, ...]
        else:
            return_coord = coord
        if return_energies:
            return return_coord, np.zeros(0)
        else:
            return return_coord
    # TODO: When `box` is reenabled, use minimum image convention for hydrogen coord
    # in hydrogen-heavy connections

    # The rotation axis is defined by the coordinates of connected atoms
    rotation_axes = np.zeros((len(rotatable_bonds), 2, 3), dtype=np.float32)
    # Each bond gets its own integer as group identifier
    # This array gives this identifier for each rotatable hydrogen atom
    groups = np.full(atoms.array_length(), -1, dtype=np.int32)
    # Whether the bond can be rotated freely or is constrained to 180 degree increments
    rotation_freedom = np.zeros(len(rotatable_bonds), dtype=bool)
    for i, (
        central_atom_index,
        bonded_atom_index,
        is_free,
        hydrogen_indices,
    ) in enumerate(rotatable_bonds):
        rotation_axes[i, 0] = coord[central_atom_index]
        rotation_axes[i, 1] = coord[bonded_atom_index]
        groups[hydrogen_indices] = i
        rotation_freedom[i] = is_free

    axes_support = jnp.asarray(rotation_axes[:, 0])
    # TODO  When `box` is reenabled, add it as parameter here
    axes_direction = jnp.asarray(
        struc.displacement(rotation_axes[:, 1], rotation_axes[:, 0])
    )
    axes_direction /= jnp.linalg.norm(axes_direction, axis=-1)[:, np.newaxis]
    rotation_freedom = jnp.asarray(rotation_freedom)
    groups = jnp.asarray(groups)
    interactions = InteractionPairs.from_structure(atoms, groups, partial_charges)
    coord = jnp.asarray(coord, dtype=jnp.float32)

    if optimizer is None:
        optimizer = optax.adam(learning_rate=1e-1)
    # Initially no rotation of the dihedrals is applied
    dihedrals = jnp.zeros(len(rotatable_bonds), dtype=jnp.float32)
    opt_state = optimizer.init(dihedrals)
    for _ in range(1000):
        grads = jax.grad(_compute_energy)(
            dihedrals,
            coord,
            axes_support,
            axes_direction,
            is_free,
            groups,
            interactions,
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        dihedrals = optax.apply_updates(dihedrals, updates)

    _compute_energy(
        dihedrals, coord, axes_support, axes_direction, is_free, groups, interactions
    )


def _get_interaction_pairs(atoms, groups, hydrogen_bond_partner_indices, force_cutoff):
    """
    Get pairs of atom indices that interact with each other based on the cutoff
    distance and other criteria.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,)
        The structure to get the interacting pairs for.
    groups : ndarray, shape=(n,), dtype=np.int32
        Groups of hydrogen atoms in `atoms`, whose positions are changed
        via the same rotatable bond.
        Each positive integer (including ``0``) represents one group,
        i.e. all atoms with the same group integer are connected to the
        same heavy atom.
        ``-1`` indicates atoms that cannot be rotated
        (heavy atoms, or non-rotatable hydrogen atoms).
    hydrogen_bond_partner_indices : ndarray, shape=(n,), dtype=int
        The indices of the heavy atom that a hydrogen atom is bound to.
        If the hydrogen atom is not bound to any heavy atom, the value is
        ``-1``.
        Contains undefined values, if indexed with non-hydrogen atom indices.
    force_cutoff : float, optional
        The force cutoff distance in Å.
        If the initial distance between two atoms exceeds this value,
        their interaction
        (:math:`V_\text{el}` and :math:`V_\text{nb}`) is not
        calculated.

    """
    if atoms.array_length() != groups.shape[0]:
        raise ValueError(
            f"There are {atoms.array_length()} atoms, "
            f"but {groups.shape[0]} group indicators"
        )

    # Filter atoms to rotatable hydrogen atoms
    relevant_mask = np.asarray(groups) != -1
    # Find proximate atoms for calculation of interacting pairs
    # TODO: When `box` is reenabled, add it as parameter here
    cell_list = struc.CellList(atoms, cell_size=force_cutoff, mask=relevant_mask)
    # Get pairs of atoms that interact with each other with respect to the cutoff
    # First column: All atoms
    # Second column: All rotatable hydrogen atoms
    interaction_indices = _to_sparse_indices(
        cell_list.get_atoms(atoms.coord, radius=force_cutoff)
    )

    # Filter these pairs to only include pairs that:
    # ... are not rotated by the same rotatable bond, as rotation of that bond would
    # not change their distance to each other
    interaction_indices = interaction_indices[
        groups[interaction_indices[:, 0]] != groups[interaction_indices[:, 1]]
    ]
    # ... are not directly bonded to each other (same reason as above)
    interaction_indices = interaction_indices[
        hydrogen_bond_partner_indices[interaction_indices[:, 1]]
        != interaction_indices[:, 0]
    ]
    # ... do not contain atoms with an unknown e.g. 'placeholder' elements
    known_element_mask = np.isin(
        atoms.element[interaction_indices[:, 0]], list(NB_VALUES.keys())
    )
    if not np.all(known_element_mask):
        warnings.warn(
            "Atoms with following unknown elements will be ignored: "
            f"{list(set(atoms.element[interaction_indices[~known_element_mask, 0]]))}"
        )
        interaction_indices = interaction_indices[known_element_mask]
    # Finally remove duplicate H-H interactions
    interaction_indices = _remove_duplicate_pairs(interaction_indices)

    heavy_indices = interaction_indices[:, 0]
    hydrogen_indices = interaction_indices[:, 1]
    return heavy_indices, hydrogen_indices


def _to_sparse_indices(contacts):
    """
    Create tuples of indices that would mark the non-zero elements in a dense
    contact matrix.
    """
    # Find rows where a query atom has at least one contact
    non_empty_indices = np.where(np.any(contacts != -1, axis=1))[0]
    # Take those rows and flatten them
    target_indices = contacts[non_empty_indices].flatten()
    # For each row the corresponding query atom is the same
    # Hence in the flattened form the query atom index is simply repeated
    query_indices = np.repeat(non_empty_indices, contacts.shape[1])
    combined_indices = np.stack([query_indices, target_indices], axis=1)
    # Remove the padding values
    return combined_indices[target_indices != -1]


def _remove_duplicate_pairs(pairs):
    """
    Remove duplicate pairs of indices.
    """
    # Sort the pairs, so that the order of the indices does not matter
    pairs = np.sort(pairs, axis=1)
    # Remove duplicates
    return np.unique(pairs, axis=0)


def _find_rotatable_bonds(atoms, mask=None):
    """
    Identify rotatable bonds between two heavy atoms, where one atom
    has only one heavy bond partner and one or multiple hydrogen
    partners.
    These bonds are used to create new conformations during relaxation.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,)
        The structure to find rotatable bonds in.
    mask : ndarray, shape=(n,), dtype=bool
        Ignore bonds, where the index of the heavy atom in the mask is
        False.
        By default no bonds are ignored.

    Returns
    -------
    rotatable_bonds : list of tuple(int, int, bool, ndarray)
        The rotatable bonds.
        The tuple elements are

            #. Atom index of heavy atom with bonded hydrogen atoms.
            #. Atom index of bonded heavy atom.
            #. If false, the bond can only be rotated by 180°.
            #. Atom indices of bonded hydrogen atoms
    """
    if mask is None:
        atom_mask = np.ones(atoms.array_length(), dtype=bool)
    else:
        if len(mask) != atoms.array_length():
            raise IndexError(
                f"Mask has length {len(atom_mask)}, "
                f"but there are {atoms.array_length()} atoms"
            )

    if atoms.bonds is None:
        raise struc.BadStructureError(
            "The input structure must have an associated BondList"
        )

    all_bond_indices, all_bond_types = atoms.bonds.get_all_bonds()

    is_hydrogen = atoms.element == "H"
    is_nitrogen = atoms.element == "N"

    rotatable_bonds = []

    # Iterate over all heavy atoms
    for i in np.where(atom_mask & ~is_hydrogen)[0]:
        hydrogen_indices = np.zeros(4, np.int32)

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
                # values have been iterated through
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
            if bonded_heavy_btype == struc.BondType.SINGLE:
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
                        if (
                            rem_btype == struc.BondType.AROMATIC_DOUBLE
                            or rem_btype == struc.BondType.DOUBLE
                        ):
                            is_free = False
                            break
            elif bonded_heavy_btype == struc.BondType.DOUBLE:
                is_free = False
            elif bonded_heavy_btype == struc.BondType.ANY:
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
            rotatable_bonds.append(
                (i, bonded_heavy_index, is_free, np.asarray(hydrogen_indices)[:h_i])
            )

    return rotatable_bonds


def _compute_energy(
    dihedrals, coord, axes_support, axes_direction, is_free, groups, pairs
):
    """
    Calulcate the energy contributions of the rotatable hydrogen atoms to the
    nonbonded energy function.

    Parameters
    ----------
    dihedrals : jax.Array, shape=(n_bonds,), dtype=float
        The dihedral angles to rotate the hydrogen atoms by.
    coord : jax.Array, shape=(n_atoms, 3), dtype=float
        The coordinates of all atoms in the structure.
    axes_support, axes_direction : jax.Array, shape=(n_bonds, 3), dtype=float
        The support vector and normalized direction vector of the rotation axis for each
        rotatable bond.
    is_free : jax.Array, shape=(n_bonds,), dtype=bool
        Whether the bond can be rotated freely or is constrained to 180 degree
        increments.
    groups : jax.Array, shape=(n_atoms,), dtype=int
        Assigns a rotation group to each hydrogen atom.
        -1 for non-rotatable or heavy atoms.
        These groups basically map `n_bonds` to `n_atoms`.
    pairs : InteractionPairs
        The pairs of interacting atoms for the energy calculation.

    Returns
    -------
    energy : jax.Array, shape=(), dtype=float
        The calculated energy.
    """
    rotated_coord = _rotate(
        dihedrals, coord, axes_support, axes_direction, is_free, groups
    )
    distances = _distance(
        rotated_coord[pairs.atom_indices[:, 1]],
        rotated_coord[pairs.atom_indices[:, 0]],
    )
    return jnp.sum(
        # Electrostatic interaction
        pairs.electrostatic_factors / distances
        # nonbonded interaction
        + pairs.epsilon_2
        * (-2 * pairs.radii_6 / distances**6 + pairs.radii_12 / distances**12)
    )


def _distance(coord_1, coord_2):
    """
    Compute the Euclidean distance between two sets of coordinates.

    Parameters
    ----------
    coord_1, coord_2 : jax.Array, shape=(n, 3), dtype=float
        The coordinates to calculate the distance between.

    Returns
    -------
    distances : jax.Array, shape=(n,)
        The calculated distances.
    """
    return jnp.linalg.norm(coord_1 - coord_2, axis=-1)


def _rotate(dihedrals, coord, axes_support, axes_direction, is_free, groups):
    """
    Rotate the hydrogen atoms around the rotatable bonds.

    Parameters
    ----------
    dihedrals : jax.Array, shape=(n_bonds,), dtype=float
        The dihedral angles to rotate the hydrogen atoms by.
    coord : jax.Array, shape=(n_atoms, 3), dtype=float
        The coordinates of all atoms in the structure.
    axes_support, axes_direction : jax.Array, shape=(n_bonds, 3), dtype=float
        The support vector and normalized direction vector of the rotation axis for each
        rotatable bond.
    is_free : jax.Array, shape=(n_bonds,), dtype=bool
        Whether the bond can be rotated freely or is constrained to 180 degree
        increments.
    groups : jax.Array, shape=(n_atoms,), dtype=int
        Assigns a rotation group to each hydrogen atom.
        -1 for non-rotatable or heavy atoms.
        These groups basically map `n_bonds` to `n_atoms`.

    Returns
    -------
    rotated_coord : jax.Array, shape=(n_atoms, 3), dtype=float
        The updated coordinates.
    """
    # TODO Constrain dihedrals via logistic function

    # Generate next hydrogen conformation
    # Calculate rotation matrices for these angles
    x = axes_direction[:, 0]
    y = axes_direction[:, 1]
    z = axes_direction[:, 2]
    sin_a = jnp.sin(dihedrals)
    cos_a = jnp.cos(dihedrals)
    icos_a = 1 - cos_a
    # fmt: off
    group_rotation_matrix = jnp.stack(
        [
            cos_a + icos_a * x**2,      icos_a * x * y - z * sin_a, icos_a * x * z + y * sin_a,
            icos_a * x * y + z * sin_a, cos_a + icos_a * y**2,      icos_a * y * z - x * sin_a,
            icos_a * x * z - y * sin_a, icos_a * y * z + x * sin_a, cos_a + icos_a * z**2,
        ],
        axis=-1
    ).reshape(-1, 3, 3)
    # fmt: on

    rotatable_h_mask = groups != -1
    relevant_groups = groups[rotatable_h_mask]
    atom_rotation_matrix = group_rotation_matrix[relevant_groups]
    # Use one point on the rotation axis as support vector
    atom_support_vector = axes_support[:, 0][relevant_groups]
    # Apply rotation to relevant hydrogen atoms
    centered_coord = coord[rotatable_h_mask] - atom_support_vector
    rotated_coord = _apply_rotation(atom_rotation_matrix, centered_coord)
    # Readd support vector
    restored_coord = rotated_coord + atom_support_vector
    coord.at[rotatable_h_mask].set(restored_coord)
    return coord


def _apply_rotation(rot_matrices, coord):
    """
    Apply *n* rotation matrices to *n* coordinates.
    """
    return jnp.tensordot(rot_matrices, coord, axes=([2], [1])).transpose(0, 2, 1)
