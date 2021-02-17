# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "hydride.relax"
__all__ = ["relax_hydrogen", "_gather_rotational_hyd"]

cimport cython
cimport numpy as np

import random
import numpy as np
from biotite.structure import (
    AtomArray, partial_charges, rotate_about_axis, distance
)

# Van der Waals radii are given in angstrom
van_der_waals_radii = {
    "H": 1.2,
    "C": 1.70,
    "N": 1.55,
    "P": 1.80,
    "O": 1.52,
    "S": 1.80,
    "F": 1.47,
    "Cl": 1.76,
    "Br": 1.85,
    "I": 1.96,
    "Na": 2.27,
    "K": 2.75,
    "Mg": 1.73,
    "Cu": 1.40,
    "Li": 1.82
}

cpdef _gather_rotational_hyd(atom_array):
    """
    Gather hydrogen atoms that possess rotational freedom and all
    required information in order to perform rotation such as the
    rotation axis, the support vector, etc.

    Parameters
    ----------
    atom_array: :class:`AtomArray`
        The input AtomArray whose hydrogen atoms' positions are supposed
        to be relaxed.

    Returns
    -------
    rot_hyd_list: list
        List containing all required information about the hydrogen
        atoms belonging to a rotatable group, such as whether it is
        freely rotatable or not, the axis around which rotation must be
        performed, etc.
    """

    cdef int i

    bond_list = atom_array.bonds
    bonds, types = bond_list.get_all_bonds()

    rot_hyd_list = []
    for i in range(atom_array.shape[0]):
        bond_indices = bonds[i][bonds[i] != -1]
        bound_elements = atom_array.element[bond_indices]
        if "H" in bound_elements:
            amount_of_bound_hyd = np.count_nonzero(
                bound_elements == "H"
            )
            bound_elements_without_hyd = bound_elements[
                bound_elements != "H"
            ]
            # If the amount of binding partners unequal to hydrogen is
            # 1, a terminal group with rotational freedom is dealt with
            if bound_elements_without_hyd.shape[0] == 1:
                # Optimization of terminal double bonds is not performed
                # if it carries two hydrogen atoms as only a rotation by
                # 180° is possible, making no difference
                if 2 in types[i] and amount_of_bound_hyd == 2:
                    continue
                # Optimization of terminal double bonds is indeed
                # performed if it carries only one hydrogen atom as
                # rotation by 180° delivers two different conformations
                # potentially differing in their energy
                if 2 in types[i] and amount_of_bound_hyd == 1:
                    hyd_index = bond_indices[bound_elements == "H"][0]
                    hyd_carrier_coord = atom_array[i].coord
                    non_hydrogen_binding_partner = bond_indices[
                        bound_elements != "H"
                    ][0]
                    non_hyd_b_part_coord = atom_array[
                        non_hydrogen_binding_partner
                    ].coord
                    axis = hyd_carrier_coord - non_hyd_b_part_coord
                    support = hyd_carrier_coord
                    rot_hyd_list.append(
                        (2, hyd_index, axis, support, "sp2H1")
                    )
                    continue
                # A terminal group with single bond is dealt with
                # The index of the heavy atom's non-hydrogen binding
                # partner is determined in order to evaluate whether the
                # terminal group with single bond is involved in a
                # mesomeric system and therefore needs to be treated as
                # a terminal group with double bond
                # Moreover, it is determined which element the terminal
                # atom is as different elements are involved in
                # mesomeric systems under different circumstances
                non_hydrogen_binding_partner = bond_indices[
                    bound_elements != "H"
                ][0]
                _, neighbour_bond_types = bond_list.get_bonds(
                    non_hydrogen_binding_partner
                )
                terminal_element = atom_array[i].element
                # In case that a double bond or even an aromatic
                # system is located next to the terminal group, it
                # is involved in a mesomeric system and therefore the
                # stereochemical constraints of a double bond need to be
                # applied to it
                if (2 in neighbour_bond_types
                    or
                    5 in neighbour_bond_types):
                    # Covering the case of e. g. a primary amino group
                    # that is not protonated, i. e. only bound to two
                    # hydrogen atoms
                    # In this case, rotation by 180° does not deliver
                    # a conformation that is distinct from the initial
                    # one
                    if amount_of_bound_hyd == 2:
                        continue
                    # Covering case that a primary amino group is
                    # protonated
                    # In this case, no free electron pair is available
                    # for the involvement in a mesomeric system so that
                    # the primary amino group is freely rotatable
                    if terminal_element == "N":
                        hyd_indices = bond_indices[
                            bound_elements == "H"
                        ]
                        hyd_carrier_coord = atom_array[i].coord
                        non_hyd_b_part_coord = atom_array[
                            non_hydrogen_binding_partner
                        ].coord
                        axis = hyd_carrier_coord - non_hyd_b_part_coord
                        support = hyd_carrier_coord
                        rot_hyd_list.append(
                            (1, hyd_indices, axis, support, "sp3H3")
                        )
                    # Carbon as terminal element with single bond, i. e.
                    # in a methyl group, is not involved in a mesomeric
                    # system even it is located in the immediate
                    # vicinity of a double bond/aromatic system
                    # Therefore, the methyl group is freely rotatable
                    if terminal_element == "C":
                        hyd_indices = bond_indices[
                            bound_elements == "H"
                        ]
                        hyd_carrier_coord = atom_array[i].coord
                        non_hyd_b_part_coord = atom_array[
                            non_hydrogen_binding_partner
                        ].coord
                        axis = hyd_carrier_coord - non_hyd_b_part_coord
                        support = hyd_carrier_coord
                        rot_hyd_list.append(
                            (1, hyd_indices, axis, support, "sp3H3")
                        )
                    else:
                        # It is dealt with hydrogen atoms of the spH1
                        # category, i. e. hydrogen atoms of hydroxyl or
                        # thiol groups
                        # This groups are freely rotatable
                        hyd_index = bond_indices[
                            bound_elements == "H"
                        ]
                        hyd_carrier_coord = atom_array[i].coord
                        non_hyd_b_part_coord = atom_array[
                            non_hydrogen_binding_partner
                        ].coord
                        axis = hyd_carrier_coord - non_hyd_b_part_coord
                        support = hyd_carrier_coord
                        rot_hyd_list.append(
                            (1, hyd_index, axis, support, "spH1")
                        )
                # The terminal group is freely rotatable
                else:
                    if terminal_element == "C":
                        hybridisation = "sp3H3"
                    elif terminal_element == "N":
                        if amount_of_bound_hyd == 2:
                            hybridisation = "sp3H2"
                        else:
                            hybridisation = "sp3H3"
                    elif terminal_element == "O":
                        hybridisation = "spH1"
                    else:
                        # Sulfur is dealt with as no other elements
                        # forming bonds to hydrogen occur as terminal
                        # group
                        hybridisation = "spH1"
                    hyd_indices = bond_indices[bound_elements == "H"]
                    hyd_carrier_coord = atom_array[i].coord
                    non_hyd_b_part_coord = atom_array[
                        non_hydrogen_binding_partner
                    ].coord
                    axis = hyd_carrier_coord - non_hyd_b_part_coord
                    support = hyd_carrier_coord
                    rot_hyd_list.append(
                        (1, hyd_indices, axis, support, hybridisation)
                    )
    
    return rot_hyd_list


cpdef _random_permutation(array_coord, rot_hyd_list, sample_num):
    """
    Rotate pseudorandomly chosen terminal groups by pseudorandomly
    chosen angles.

    Parameters
    ----------
    array_coord: ndarray, shape=(n,3), dtype=float
        The array containing the coordinates of the input structure
        before pseudorandom rotation.
    rot_hyd_list: list
        List containing all required information about the hydrogen
        atoms belonging to a rotatable group, such as whether it is
        freely rotatable or not, the axis around which rotation must be
        performed, etc.
    sample_num: int
        Integer representing the amount of rotatable groups that are
        rotated within one simulation step.

    Returns
    -------
    array_coord: ndarray, shape=(n,3), dtype=float
        The array containing the coordinates after pseudorandom
        rotation.
    """

    # Application of `random.sample` ensures that no repetition of items
    # occurs in the random selection
    random_selection = random.sample(rot_hyd_list, k=int(sample_num))

    for selected_tuple in random_selection:
        axis = selected_tuple[2]
        support = selected_tuple[3]
        if selected_tuple[0] == 1:
            # Terminal groups with single bond not involved in mesomeric
            # systems are freely rotatable which is why a random angle
            # is chosen
            angle = random.uniform(0, 2 * np.pi)
            hyd_indices = selected_tuple[1].tolist()
            array_coord[hyd_indices] = rotate_about_axis(
                array_coord[hyd_indices], axis, angle, support
            )
        else:
            # Accounting for a double bond's stereochemical constraints
            # by either rotating the hydrogen atom by 180° or not at all
            # The same applies to terminal groups with single bond that
            # are involved in mesomeric systems
            angle = random.choice([0, np.pi])
            if angle == np.pi:
                hyd_index = selected_tuple[1][0]
                array_coord[hyd_index] = rotate_about_axis(
                    array_coord[hyd_index], axis, angle, support
                )

    return array_coord


cpdef _eval_energy(
    hyd_indices, array_coord, array_elements, partial_charges
):
    """
    Compute the energy of a certain configuration of a molecule inserted
    into the function `relax_hydrogen` as the parameter `atom_array`.

    To be more precise, the energy of a molecule's conformation is
    dependent on electrostatic interactions and steric repulsion in this
    approach. The individual energy contributions are computed according
    to the so-called HAAD algorithm developed by the Yang Zhang group.
    [1]_
    For the steric repulsion, van der Waals radii determined
    by Bondi are employed. [2]_

    Parameters
    ----------
    hyd_indices: ndarray, shape=(n,), dtype=int
        The array containing the indices of hydrogen atoms with
        rotational freedom comprised in the input structure.
    array_coord: ndarray, shape=(n,3), dtype=float
        The array containing the coordinates of the atoms comprised in
        the input structure.
    array_elements: ndarray, shape=(n,), dtype=str
        The array containing information about the elements of the
        individual atoms comprised in the input structure.
    partial_charges: ndarray, shape=(n,), dtype=float
        The array comprising the partial charges of the atoms comprised
        in the input structure.

    Returns
    -------
    energy: float
        The energy the respective conformation has.

    References
    ----------
    .. [1] Y Li, A Roy, Y Zhang,
        "HAAD: A Quick Algorithm for Accurate Prediction of
        Hydrogen Atoms in Protein Structures"
        PLoS One 4(8): e6701 (2009)
    .. [2] A Bondi,
        "Van der Waals Volumes and Radii"
        J. Phys. Chem., 68, 441 - 451 (1964)
    """

    cdef int i, hyd_index, atom_index

    energy = 0
    hyd_vdv_radius = van_der_waals_radii["H"]

    hyd_coords = array_coord[hyd_indices]

    # Enabling broadcasting of the array with hydrogen coordinates and
    # the array comprising all atoms' coordinates by introducing new
    # axes
    hyd_coords = hyd_coords[np.newaxis, :, :]
    atom_coords = array_coord[:, np.newaxis, :]
    distances = distance(hyd_coords, atom_coords)

    for i, hyd_index in enumerate(hyd_indices):
        for atom_index in range(array_coord.shape[0]):
            # Interaction of hydrogen atoms with themselves is not
            # considered
            if atom_index == hyd_index:
                continue
            else:
                atom_distance = distances[atom_index, i]
                element = array_elements[atom_index]
                atom_vdv_radius = van_der_waals_radii[element]
                vdv_sum = hyd_vdv_radius + atom_vdv_radius
                if atom_distance <= vdv_sum:
                    steric_repulsion = 10 * (
                        hyd_vdv_radius + atom_vdv_radius - atom_distance
                    )
                    if np.isnan(steric_repulsion):
                        continue
                    energy += steric_repulsion
                if atom_distance <= 4:
                    hyd_part_charge = partial_charges[hyd_index]
                    atom_part_charge = partial_charges[atom_index]
                    coulomb_interaction = (
                        hyd_part_charge * atom_part_charge
                    )
                    # Accounting for the fact that the partial charge is
                    # given as NaN if parameters are not available
                    if np.isnan(coulomb_interaction):
                        continue
                    energy += coulomb_interaction

    return energy


def relax_hydrogen(atom_array, iteration_step_num=1000000):
    """
    Find the position of hydrogen atoms belonging to rotatable terminal
    groups that corresponds to the minimum energy conformation and thus
    to the native conformation of the molecule.

    Parameters
    ----------
    atom_array: :class:`AtomArray`
        The AtomArray whose hydrogen atoms' positions are supposed to be
        relaxed.
    iteration_step_num: int
        An integer representing the amount of steps the simulation
        comprises.

    Returns
    -------
    return_atom_array: :class:`AtomArray`
        The input AtomArray, but with relaxed hydrogen positions.
    """

    cdef int i

    if not isinstance(atom_array, AtomArray):
        raise ValueError("Input must be AtomArray")

    if atom_array.bonds is None:
        raise AttributeError(
            f"The input AtomArray doesn't possess an associated "
            f"BondList."
        )

    part_charges = partial_charges(atom_array)
    array_coord = atom_array.coord.copy()
    array_elements = atom_array.element

    rot_hyd_list = _gather_rotational_hyd(atom_array)
    amount_rotatable = len(rot_hyd_list)
    hyd_indices = np.array([], dtype=int)
    for tuple_ in rot_hyd_list:
        hyd_indices = np.append(hyd_indices, tuple_[1])
    # Changing the position of too many rotatable groups at once impedes
    # a continuous decrease of the energy / a descent in the energy
    # landscape but rather leads to noise
    # For this reason, the amount of rotatable groups rotated at once is
    # kept low
    if amount_rotatable >= 50:
        sample_num = 10
    else:
        sample_num = amount_rotatable * 0.2
        sample_string = str(sample_num)
        if len(sample_string) != 1:
            sample_num = int(sample_string[0])

    # Creating lists to store accepted conformations of hydrogen
    # positions and the corresponding energies in
    minimum_energies = []
    minimum_conformations = []

    current_conformation = array_coord
    minimum_conformations.append(current_conformation)
    current_energy = _eval_energy(
        hyd_indices, current_conformation, array_elements, part_charges
    )
    minimum_energies.append(current_energy)

    # System is simulated for the given amount of iteration steps
    for i in range(iteration_step_num ):
        test_conformation = _random_permutation(
            current_conformation, rot_hyd_list, sample_num
        )
        test_energy = _eval_energy(
            hyd_indices, test_conformation, array_elements, part_charges
        )
        # Only accept conformations possessing corresponding energies
        # that obey the Boltzmann distribution
        # Effectively, this is achieved by pseudorandomly generating a
        # probability, i. e. a float between 0 and 1 and accepting a
        # conformation only if its energy's Boltzmann probability at the
        # given temperature is higher than the pseudorandom probability
        energy_difference = test_energy - current_energy
        # As simulated annealing is performed, the temperature decreases
        # exponentially
        temperature = 5 * np.exp(-0.0019569899977129293*i)
        if (
            # Boltzmann constant is omitted as energy does not possess a
            # proper unit
            random.random() < np.exp(-energy_difference / temperature)
        ):
            minimum_energies.append(test_energy)
            minimum_conformations.append(test_conformation)
            current_conformation = test_conformation
            current_energy = test_energy

    minimum_index = np.nonzero(
        minimum_energies == np.amin(minimum_energies)
    )[0][0]
    minimum_conformation = minimum_conformations[minimum_index]
    
    return_atom_array = atom_array.copy()
    return_atom_array.coord = minimum_conformation

    return return_atom_array