# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides one function for the addition of hydrogen atoms to
residues occurring in the Chemical Components Dictionary (CCD) and
notably to macromolecules made up of those residues, enabling hydrogen
addition to small molecules / ligands as well as biological
macromolecules like proteins and nucleic acids.
"""

__name__ = "hydride.add"
__author__ = "Jacob Marcel Anter"
__all__ = ["add_hydrogen"]

import warnings
import numpy as np
from biotite.structure import (
   Atom, array, AtomArray, BondList, filter_amino_acids,
   rotate_about_axis, get_residue_starts, BondType
)
from biotite.structure.info import residue, standardize_order
from biotite.structure.superimpose import _superimpose


# Constructing reference mobile fragment for hydrogen addition at amino
# group involved in peptide bond
# The reason is that the bond angle of the hydrogen of an amino group
# involved in a peptide bond differs from that of hydrogen of a free
# amino group
# The orientation of the hydrogen atom must be adjusted as well
# Initially, all atoms are placed along the x-axis with the correct
# distance from the nitrogen atom (cf. reference in DocString of the
# `add_hydrogen` function)
nitrogen = np.array([0,0,0])
calpha = np.array([1.46,0,0])
carbon = np.array([1.33,0,0])
hydrogen = np.array([1.02,0,0])
# Alpha carbon and hydrogen are rotated about the respective angle
# obtained from literature (cf. reference in DocString of the
# 'add_hydrogen' function)
# Rotation is performed in xy plane
z_axis = np.array([0,0,1])
# 122° (CA-N-C angle) correspond to 2.1293 radians
carbon = rotate_about_axis(carbon, z_axis, angle=2.1293)
# 241.5° (CA-N-C angle plus C-N-H angle) correspond to 4.215 radians
hydrogen = rotate_about_axis(hydrogen, z_axis, angle=4.215)
nitrogen_atom = Atom(nitrogen, element="N")
calpha_atom = Atom(calpha, element="C")
carbon_atom = Atom(carbon, element="C")
hydrogen_atom = Atom(hydrogen, element="H")
amino_fragment_with_hyd = array([
   carbon_atom,
   calpha_atom,
   hydrogen_atom,
   nitrogen_atom
])
amino_hydrogen = amino_fragment_with_hyd[2]
amino_fragment = np.array([
   carbon,
   calpha,
   nitrogen
])


def _gather_hyd_indices(
   counter, start, first_res_start, last_res_start, res_starts,
   chain_ids, ref_res_array, considered_chain_id, het_bool_value,
   aa_bool_value, res_name
):
   """
   Gather the indices of hydrogen atoms in the reference array from the
   Chemical Components Dictionary to add to the subject array.
   
   The decision which hydrogen atoms occurring in the reference array
   are to add to the subject array is based on the fact whether the
   subject array is incorporated in a chain or not, the subject array's
   position in the chain (beginning, middle or end) and the subject
   array's biochemical class (amino acid, nucleotide or small molecule/
   ligand). The subject array's position in a chain respectively whether
   it is incorporated into a chain at all is determined via the chain id
   of the considered residue and its neighbour residues whereas the
   biochemical class is determined by the considered residue's `hetero`
   annotation and the application of the function `filter_amino_acids`.
   Parameters
   ----------
   counter: int
      Variable that keeps track of the number the currently considered
      residue has in the sequence of all residues. It is required in
      order to determine the considered residue's chain id as well as
      those of its neighbours.
   start: int
      The start index of the considerd residue. It is required in order
      to determine the considered residue's position in a chain
      respectivey whether it is incorporated into a chain at all.
   ref_res_array: :class:`AtomArray`
      The AtomArray serving as reference for hydrogen addition for the
      subject array. It is retrieved from the Chemical Component
      Dictionary.
   considered_chain_id: str
      The chain id of the currently considered residue. Its retrieval is
      necessary in order to evaluate the residue's position in a chain,
      if it is incorporated in a chain at all.
   first_res_start: int
      The index at which the first residue of the `atom_array` inserted
      in the `add_hydrogen` function starts.
   last_res_start: int
      The index at which the last residue of the `atom_array` inserted
      in the `add_hydrogen` function starts.
   res_starts: ndarray, dtype=int
      The array containing the start indices of the residues comprised
      in the `atom_array` inserted in the `add_hydrogen` function.
   Returns
   -------
   hyd_indices: ndarray, dtype=int
      Array comprising the indices of the hydrogen atoms from the
      reference AtomArray that are supposed to be added to the subject
      array.
   amino_hyd_bool: bool
      Boolean value indicating whether a considered residue is an amino
      acid and additionally has an amino group involved in a peptide
      bond, i. e. whether the residue is located in the middle or the
      end of a peptide chain.
   monomer_class: int
      Integer denoting the biochemical class of the currently considered
      residue. 0 represents the biochemical class 'amino acid', 1
      represents 'nucleotide' and 2 represents 'ligans/small molecule'. 
   res_pos: str
      A string denoting the currently considered residue's position
      within a chain.
   """

   start_of_array = False
   end_of_array = False
   amino_hyd_bool = False
   single_res = False

   # Considering case that input `AtomArray` consists of one single
   # residue
   if start == first_res_start and start == last_res_start:
      single_res = True
   # Considering case that residue is located at the beginning of
   # inserted `AtomArray`
   elif start == first_res_start:
      start_of_array = True
      successor_chain_id = chain_ids[
         res_starts[counter + 1]
      ]
   # Considering case that residue is located at the end of inserted
   # `AtomArray`
   elif start == last_res_start:
      end_of_array = True
      precursor_chain_id = chain_ids[
         res_starts[counter - 1]
      ]
   # Considering case that residue is located in middle of inserted
   # `AtomArray`
   else:
      successor_chain_id = chain_ids[
         res_starts[counter + 1]
      ]
      precursor_chain_id = chain_ids[
         res_starts[counter - 1]
      ]
   
   if single_res:
      # In case of a single residue, all hydrogen atoms listed in the
      # reference array are added
      hyd_indices = np.array(
         [idx for idx, i in enumerate(ref_res_array.element)
            if i == "H"]
      )
      res_pos = "free"
      # In case of a single residue, be it an amino acid, a nucleotide
      # or a ligand/solvent molecule, all heavy atoms occurring in the
      # reference also occur in the subject array
      # Therefore, no subtraction needs to be performed and the ligand's
      # class is assigned to the residue
      monomer_class = 2
   # If a residue's `hetero` annotation is False, it is either an amino
   # acid or a nucleotide
   elif het_bool_value == False:
      # Checking whether aminoacid is dealt with
      if aa_bool_value:
         monomer_class = 0
         if start_of_array:
            if considered_chain_id == successor_chain_id:
               res_pos = "beginning"
               hyd_indices = np.array(
                  [idx for idx, i in
                     enumerate(ref_res_array.atom_name)
                     if i[0] == "H" and i != "HXT"]
               )
            # Considering case that free amino acid is dealt with
            else:
               res_pos = "free"
               hyd_indices = np.array(
                  [idx for idx, i in enumerate(ref_res_array.element)
                     if i == "H"]
               )
         elif end_of_array:
            if considered_chain_id == precursor_chain_id:
               res_pos = "end"
               amino_hyd_bool = True
               # The case of the amino acid proline must be paid
               # particular attention to as the hydrogen of the amino
               # group is not present if the amino group is involved in
               # a peptide bond
               if res_name == "PRO":
                  hyd_indices = np.array(
                     [idx for idx, i in
                        enumerate(ref_res_array.atom_name)
                        if i[0] == "H" and i != "H"]
                  )
               else:
                  hyd_indices = np.array(
                     [idx for idx, i in
                        enumerate(ref_res_array.atom_name)
                        if i[0] == "H" and i != "H2"]
                  )
            # Considering case that free amino acid is dealt with
            else:
               res_pos = "free"
               hyd_indices = np.array(
                  [idx for idx, i in enumerate(ref_res_array.element)
                     if i == "H"]
               )
         else:
            if (considered_chain_id == precursor_chain_id
               and
               considered_chain_id == successor_chain_id):
               res_pos = "middle"
               amino_hyd_bool = True
               if res_name == "PRO":
                  hyd_indices = np.array(
                     [idx for idx, i in
                        enumerate(ref_res_array.atom_name)
                        if i[0] == "H" and i != "HXT" and i != "H"]
                  )
               else:
                  hyd_indices = np.array(
                     [idx for idx, i in
                        enumerate(ref_res_array.atom_name)
                        if i[0] == "H" and i != "HXT" and i != "H2"]
                  )
            elif considered_chain_id == precursor_chain_id:
               res_pos = "end"
               amino_hyd_bool = True
               if res_name == "PRO":
                  hyd_indices = np.array(
                     [idx for idx, i in
                        enumerate(ref_res_array.atom_name)
                        if i[0] == "H" and i != "H"]
                  )
               else:
                  hyd_indices = np.array(
                     [idx for idx, i in
                        enumerate(ref_res_array.atom_name)
                        if i[0] == "H" and i != "H2"]
                  )
            elif considered_chain_id == successor_chain_id:
               res_pos = "beginning"
               hyd_indices = np.array(
                  [idx for idx, i in
                     enumerate(ref_res_array.atom_name)
                     if i[0] == "H" and i != "HXT"]
               )
            # Considering case that free amino acid is dealt with
            else:
               res_pos = "free"
               hyd_indices = np.array(
                  [idx for idx, i in enumerate(ref_res_array.element)
                     if i == "H"]
               )
      # A nucleotide is dealt with
      else:
         monomer_class = 1
         if start_of_array:
            if considered_chain_id == successor_chain_id:
               res_pos = "beginning"
               hyd_indices = np.array(
                  [idx for idx, i in
                     enumerate(ref_res_array.atom_name)
                     if i[0] == "H" and i != "HO3'"]
               )
            # Considering case that free amino acid is dealt with
            else:
               res_pos = "free"
               hyd_indices = np.array(
                  [idx for idx, i in enumerate(ref_res_array.element)
                     if i == "H"]
               )
         elif end_of_array:
            if considered_chain_id == precursor_chain_id:
               res_pos = "end"
               hyd_indices = np.array(
                  [idx for idx, i in
                     enumerate(ref_res_array.atom_name)
                     if i[0] == "H" and i != "HOP3"]
               )
            # Considering case that free amino acid is dealt with
            else:
               res_pos = "free"
               hyd_indices = np.array(
                  [idx for idx, i in enumerate(ref_res_array.element)
                     if i == "H"]
               )
         else:
            if (considered_chain_id == precursor_chain_id
               and
               considered_chain_id == successor_chain_id):
               res_pos = "middle"
               hyd_indices = np.array(
                  [idx for idx, i in
                     enumerate(ref_res_array.atom_name)
                     if i[0] == "H" and i != "HO3'" and i != "HOP3"]
               )
            elif considered_chain_id == precursor_chain_id:
               res_pos = "end"
               hyd_indices = np.array(
                  [idx for idx, i in
                     enumerate(ref_res_array.atom_name)
                     if i[0] == "H" and i != "HOP3"]
               )
            elif considered_chain_id == successor_chain_id:
               res_pos = "beginning"
               hyd_indices = np.array(
                  [idx for idx, i in
                     enumerate(ref_res_array.atom_name)
                     if i[0] == "H" and i != "HO3'"]
               )
            # Considering case that free amino acid is dealt with
            else:
               res_pos = "free"
               hyd_indices = np.array(
                  [idx for idx, i in enumerate(ref_res_array.element)
                     if i == "H"]
               )
   else:
      monomer_class = 2
      res_pos = "free"
      # It is assumed that the considered residue is just a small
      # molecule like fluoromethane that is not involved in a
      # macromolecular structure
      # Therefore, all hydrogen atoms listed in the reference are added
      # to the subject residue
      hyd_indices = np.array(
         [idx for idx, i in enumerate(ref_res_array.element)
            if i == "H"]
      )

   return (hyd_indices, amino_hyd_bool, monomer_class, res_pos)


def _create_atom_array_including_hyd(
   num_heavy_atoms, amount_of_hyd_to_add, atom_sum, subject_array,
   has_charge_annotation, has_occupancy_annotation,
   has_b_factor_annotation
):
   """
   Create a new :class:`AtomArray` for the subject residue including the
   hydrdogen atoms that are supposed to be added to the subject array.
   The length of the array's zero axis is determined by forming the sum
   of the atoms comprised in the `subject_array` and the amount of
   hydrogen atoms to add.
   Annotation entries for the heavy atoms as coordinates, atom names,
   etc. are copied from the `subject_array`. In case of the added
   hydrogen atoms, annotation categories that are identical throughout
   the residue (chain id, residue name, insertion code) are
   copied from the heavy atoms as well whereas annotation categories
   that are different among the individual hydrogen atoms as coordinates
   are added later on. The residue id, however, is added later on
   in order to be able to distinguish between zero rows and non-zero
   rows.
   Optional annotation categories such as charge or occupancy are copied
   if present in the input array.
   Parameters
   ----------
   num_heavy_atoms: int
      The amount of heavy, i. e. non-hydrogen atoms occurring in the
      reference array.
   amount_of_hyd_to_add: int
      The amount of hydrogen atoms that are supposed to be added to the
      subject array.
   Returns
   -------
   res_array_with_hyd: :class:`AtomArray`
      The newly created AtomArray for the subject array including the
      hydrogen atoms that are supposed to be added.
   """

   subject_coords = subject_array.coord
   subject_atom_names = subject_array.atom_name
   subject_elements = subject_array.element
   chain_id = subject_array.chain_id[0]
   res_id = subject_array.res_id[0]
   res_name = subject_array.res_name[0]
   ins_code = subject_array.ins_code[0]
   hetero_bool_value = subject_array.hetero[0]

   blank_hyd_atom = Atom([0,0,0], element="H")
   blank_heavy_atom = Atom([0,0,0])
   res_array_with_hyd = array(
      [blank_heavy_atom] * num_heavy_atoms
      +
      [blank_hyd_atom] * amount_of_hyd_to_add
   )

   res_array_with_hyd.coord[:num_heavy_atoms] = subject_coords
   res_array_with_hyd.chain_id = np.array([chain_id] * atom_sum)
   res_array_with_hyd.res_id = np.array([res_id] * atom_sum)
   res_array_with_hyd.ins_code = np.array([ins_code] * atom_sum)
   res_array_with_hyd.res_name = np.array([res_name] * atom_sum)
   res_array_with_hyd.hetero = np.array([hetero_bool_value] * atom_sum)
   res_array_with_hyd.atom_name[:num_heavy_atoms] = subject_atom_names
   res_array_with_hyd.element[:num_heavy_atoms] = subject_elements
   
   # Optional annotation categories are added if present in the input
   # array
   if has_charge_annotation:
      subject_charges = subject_array.charge
      res_array_with_hyd.set_annotation(
         "charge",
         np.concatenate(
            # Hydrogen never possesses a formal charge different from
            # zero
            (subject_charges, np.array([0] * amount_of_hyd_to_add))
         )
      )
   
   if has_occupancy_annotation:
      subject_occupancies = subject_array.occupancy
      res_array_with_hyd.set_annotation(
         "occupancy",
         np.concatenate(
            # For hydrogen, the default value of the occupancy 1 is used
            (subject_occupancies, np.array([1] * amount_of_hyd_to_add))
         )
      )

   if has_b_factor_annotation:
      subject_b_factors = subject_array.b_factor
      res_array_with_hyd.set_annotation(
         "b_factor",
         np.concatenate(
            # For hydrogen, the default value of the b factor 0 is used
            (subject_b_factors, np.array([0] * amount_of_hyd_to_add))
         ) 
      )

   return res_array_with_hyd


def _build_occurrence_array(hyd_indices, mobile_bond_list):
   """
   Build the so-called `occurrence_array` whose purpose is to store the
   amount of hydrogen atoms bound to a certain heavy atom in the
   `subject_array`.
   The `occurrence_array` stores the amount of hydrogen atoms bound to a
   considered heavy atom in the following way:
   Each row represents one central atom carrying hydrogen and each entry
   unequal to -1 (or each entry corresponding to the respective atom's
   index in the reference array) represents one hydrogen atom bound to
   this central atom.
   Thus, the sum of entries of one row unequal to -1 corresponds to the
   amount of hydrogen bound to the respective central atom.
   In order to achieve broadcasting of the array, gaps are filled
   with entries of -1.
   Parameters:
   -----------
   hyd_indices: ndarray, dtype=int
      Array comprising the indices of the hydrogen atoms from the
      reference AtomArray that are supposed to be added to the subject
      array.
   mobile_bond_list: :class:`BondList`, shape=(n,3)
      BondList comprising the indices of connected atoms of the
      so-called mobile structure, i. e. the reference array from the CCD
      as well as the respective BondType.
   Returns:
   --------
   occurrence_array: ndarray, shape=(n,m), dtype=int
      Array storing information about how many hydrogen atoms are bound
      to a certain heavy atom of the currently considered residue.
   atom_indices_carrying_hyd: list
      List storing the indices of heavy atoms in the reference that are
      bound to hydrogen.
   """

   atom_indices_carrying_hyd = []
   for hyd_index in hyd_indices:
      bond, _ = mobile_bond_list.get_bonds(hyd_index)
      atom_indices_carrying_hyd += list(bond)
   
   # The most frequent index and its corresponding number of
   # occurrences are determined in order to determine the length of
   # `occurrence_array` along axis 1
   most_frequent_index = max(
         set(atom_indices_carrying_hyd),
         key=atom_indices_carrying_hyd.count
      )
   amount_most_frequent = np.count_nonzero(
      atom_indices_carrying_hyd == most_frequent_index
   )
   unique_atom_ind_car_hyd, counts = np.unique(
      atom_indices_carrying_hyd,
      return_counts=True
   )
   amount_indices = unique_atom_ind_car_hyd.shape[0]

   # A dictionary is created in order to compare the amount of
   # hydrogen bound to one atom to the maximum amount of hydrogen
   # bound to an atom of the considered residue and thus determine
   # the amount of -1 entries to add if necessary
   count_dict = dict(zip(unique_atom_ind_car_hyd, counts))

   id_counter = 0
   occurrence_array = np.array(atom_indices_carrying_hyd, dtype=int)
   # The following loop builds the `occurrence_array`
   for hyd_occurrence in unique_atom_ind_car_hyd:
      hyd_count = count_dict[hyd_occurrence]
      if hyd_count != amount_most_frequent:
         num_of_fill_to_add = amount_most_frequent - hyd_count
         filling_array = np.array([-1] * num_of_fill_to_add)
         indices = np.where(
            atom_indices_carrying_hyd == hyd_occurrence
         )[0]
         # The computation of the `insertion_index` can be explained
         # as follows: The array containing the indices of the
         # `hyd_occurrences`, i. e. the indices of the considered
         # atom carrying hydrogen, is accessed
         # Then, the last index is accessed and 1 is added to this
         # index as the `filling_array` is supposed to be inserted
         # directly after the last occurrence of the index of the
         # considered atom carrying hydrogen in the
         # `occurrence_array`
         # In order to track the shift of indices due to
         # insertion, an `id_counter` is introduced
         insertion_index = indices[hyd_count - 1] + 1 + id_counter
         id_counter += num_of_fill_to_add
         occurrence_array = np.insert(
            occurrence_array,
            insertion_index,
            filling_array
         )
   occurrence_array = np.reshape(
      occurrence_array,
      newshape=(amount_indices, amount_most_frequent)
   )

   return (occurrence_array, unique_atom_ind_car_hyd)

def _construct_fragments(
   counter, amino_hyd_bool, res_starts, fixed_bond_list,
   subject_length_zero_axis, subject_atom_names, subject_coords,
   atom_names_carrying_hyd, ref_atom_names, ref_coords,
   atom_array_coords, subject_elements, monomer_class, res_pos
):
   """
   Construct fragments for the superimposition.
   A fragments pair consists of a fixed and a mobile fragment. The fixed
   fragment is built from the atom in the subject array which hydrogen
   is supposed to be added to as well as its neighbouring atoms whereas
   the mobile fragment is built from the corresponding atom in the
   reference array that carries hydrogen as well as its neighbouring
   atoms.
   Parameters
   ----------
   counter: int
      Variable that keeps track of the number the currently considered
      residue has in the sequence of all residues. In case of an amino
      acid whose amino group is involved in a peptide bond, it is
      required in order to add the carbon atom of the neighbouring amino
      acid's carboxyl group to the fixed fragment as this carbon atom
      can not bee accessed via the `fixed_bond_list`.
   amino_hyd_bool: bool
      Boolean value indicating whether a considered residue is an amino
      acid and additionally has an amino group involved in a peptide
      bond, i. e. whether the residue is located in the middle or the
      end of a peptide chain.
   res_starts: ndarray, dtype=int
      The array containing the start indices of the residues comprised
      in the `atom_array` inserted in the `add_hydrogen` function.
   fixed_bond_list: :class:`BondList`, shape=(n,3)
      BondList comprising the indices of connected atoms of the
      so-called fixed structure, i. e. the currently considered residue
      from the `atom_array` inserted into the `add_hydrogen` function as
      well as the respective BondType.
   subject_length_zero_axis: int
      Integer representing the subject array's length in the first
      dimension which is equivalent to the amount of heavy atoms
      comprised in the subject array.
   subject_atom_names: ndarray, dtype=str
      Array comprising the names of the atoms the subject array consists
      of.
   subject_coords: ndarray, shape=(n,3), dtype=float
      Array comprising the coordinates of the atoms comprised in the
      subject array.
   atom_names_carrying_hyd: list
      List comprising the names of atoms in the reference array from the
      CCD that are bound to hydrogen.
   ref_atom_names: ndarray, dtype=str
      Array comprising the names of the atoms the reference array from
      the CCD consists of.
   ref_coords: ndarray, shape=(n,3), dtype=float
      Array comprising the coordinates of the atoms comprised in the
      reference array.
   atom_array_coords: ndarray, shape=(n,3), dtype=float
      Array comprising the coordinates of the atoms comprised in the
      `atom_array` inserted into the `add_hydrogen` function.
   subject_elements: ndarray, dtype=str
      Array comprising the elements the subject array consists of.
   monomer_class: int
      Integer denoting the biochemical class of the currently considered
      residue. 0 represents the biochemical class 'amino acid', 1
      represents 'nucleotide' and 2 represents 'ligans/small molecule'. 
   res_pos: str
      A string denoting the currently considered residue's position
      within a chain.
   Returns
   -------
   list_of_fragment_pairs: list
      List comprising tuples which in turn comprise fragment pairs,
      i. e. a fixed fragment originating from the subject array and a
      corresponding mobile fragment originating from the reference array
      as well as the index of the respective atom of the fixed fragment
      that hydrogen is supposed to be added to.
   reduction: int
      Integer keeping track of the amount of heavy atoms occurring in
      the reference array from the CCD but not in the subject array.
   """

   list_of_fragment_pairs = []
   insertion_indices_for_absence_marker = []

   # The index of heavy atoms occurring in the reference but not in the
   # subject is determined in order to insert a respective 'marker'
   # string in the `list_of_fragment_pairs` at the respective position
   # This procedure requires that the heavy atoms occur in the subject
   # array as well as in the reference array in the same order
   # At first, `subject_atom_names` is transformed to
   # `subject_atom_names_carrying_hyd` by removing atom names of atoms
   # that do not carry hydrogen
   subject_atom_names_carrying_hyd = subject_atom_names.copy()
   print("subject_atom_names_carrying_hyd vor der Bereinigung: ", subject_atom_names_carrying_hyd)
   for atom_name in subject_atom_names:
      if atom_name not in atom_names_carrying_hyd:
         print("Immerhin wird erkannt, dass das Atom keinen Wasserstoff trägt.")
         subject_atom_names_carrying_hyd[
            subject_atom_names_carrying_hyd != atom_name
         ]
   print("nach der Bereinigung: ", subject_atom_names_carrying_hyd)
   
   print(atom_names_carrying_hyd)
   print(subject_atom_names_carrying_hyd)
   if (
      len(atom_names_carrying_hyd)
      !=
      subject_atom_names_carrying_hyd.shape[0]
   ):
      difference = (
         abs(
            len(atom_names_carrying_hyd)
            -
            subject_atom_names_carrying_hyd.shape[0]
         )
      )
      print("Die Differenz beträgt: ", difference)
      print("Die Monomerklasse: ", monomer_class)
      if monomer_class == 0:
         if res_pos == "beginning" or res_pos == "middle":
            if difference == 1:
               # The oxygen atom of the carboxyl group is lost due to
               # the condensation reaction
               # Therefore, the hydrogen atom bound to it is neglected
               # anyway
               print("Pass")
               pass
            else:
               print("Dieser scheiß zweig wird erreicht")
               for i in range(len(atom_names_carrying_hyd)):
                  #print("Scheiße")
                  if (atom_names_carrying_hyd[i] not in subject_atom_names_carrying_hyd) and (atom_names_carrying_hyd[i] != "OXT"):
                     insertion_indices_for_absence_marker.append(i)
         else:
            print("Oh, das Ende")
            for i in range(len(atom_names_carrying_hyd)):
               #print(i)
               if (
                  atom_names_carrying_hyd[i]
                  not in subject_atom_names_carrying_hyd
               ):
                  insertion_indices_for_absence_marker.append(i)
      elif monomer_class == 1:
         # Muss noch erledigt werden
         pass
      # A ligand/solvent molecule is dealt with
      else:
         for i in range(len(atom_names_carrying_hyd)):
            if (
               atom_names_carrying_hyd[i]
               not in subject_atom_names_carrying_hyd
            ):
               insertion_indices_for_absence_marker.append(i)


   # Iterating through the subject array in order to build fragments of
   # central atoms carrying hydrogen
   # Only coordinates are involved
   for i in range(subject_length_zero_axis):
      if subject_atom_names[i] in atom_names_carrying_hyd:
         fixed_fragment = np.array([])
         binding_partners, types = fixed_bond_list.get_bonds(i)
         if amino_hyd_bool and subject_atom_names[i] == "N":
            f_fragment_indices = np.append(
               binding_partners, np.array([i])
            )
            mobile_fragment = amino_fragment
            for i in f_fragment_indices:
               fixed_fragment = np.append(
                  fixed_fragment,
                  subject_coords[i]
               )
            # Carbon atom of carboxyl group of neighboured peptide
            # can't be accessed via `fixed_bond_list`
            # It occupies always the third position in an amino
            # acid's AtomArray
            previous_start = res_starts[counter - 1]
            fixed_fragment = np.insert(
               fixed_fragment,
               obj=0,
               values=atom_array_coords[previous_start + 2],
               axis=0
            )
         else:
            # The case of terminal double bonds must be paid particular
            # attention to as they otherwise form fragments consisting
            # of only two atoms which entail rotational freedom and
            # therefore do not account for the fact that sigma bonds of
            # atoms with sp2 hybridisation must be located within one
            # plane
            # Moreover, it must be evaluated whether whether terminal
            # single bonds are involved in mesomeric systems as if this
            # applies the single bond must be treated as double bond
            # One example is the guanidino group in the side chain of
            # arginine:
            # As the guanidino group is protonated in the the reference
            # array, resulting in delocalisation of the pi electrons
            # across the whole guanidino group, the nitrogen atom of the
            # guanidino group formally having a single bond ("NH1") must
            # be treated as having a double bond as well
            if binding_partners.shape[0] == 1:
               neighbour_binding_partners, neighbour_types = \
                  fixed_bond_list.get_bonds(binding_partners[0])
               if (
                  # Accounting for terminal double bond
                  types[0] == 2
                  or
                  # Accounting for aromatic and therefore actually
                  # cyclic systems of which one heavy atom has been
                  # removed
                  types[0] == 5
                  or
                  # Accounting for terminal single bonds involved in
                  # mesomeric systems
                  (subject_elements[i] == "N"
                     and (2 in neighbour_types or 5 in neighbour_types))
                  or
                  (subject_elements[i] == "O"
                     and (2 in neighbour_types or 5 in neighbour_types))
                  or
                  (subject_elements[i] == "S"
                     and (2 in neighbour_types or 5 in neighbour_types))
               ):
                  f_fragment_indices = np.append(
                     neighbour_binding_partners, binding_partners[0]
                  )
               else:
                  f_fragment_indices = np.append(
                     binding_partners, np.array([i])
                  )
            else:
               f_fragment_indices = np.append(
                  binding_partners, np.array([i])
               )
            mobile_fragment = np.array([])
            name_list = []
            for j in f_fragment_indices:
               fixed_fragment = np.append(
                  fixed_fragment, subject_coords[j]
               )
               name_list.append(subject_atom_names[j])
            # Introducing new index in order to prevent
            # interference with index 'j' from the iteration
            # through the subject array
            # It is iterated through the `name_list` and not
            # through the `ref_res_array` in order to ensure that
            # the coordinates belonging to a certain atom appear
            # in the same sequence
            for k in name_list:
               if k in ref_atom_names:
                  mobile_fragment = np.append(
                     mobile_fragment,
                     ref_coords[
                        k == ref_atom_names
                     ]
                  )
            mobile_fragment = np.reshape(
               mobile_fragment,
               newshape=(int(mobile_fragment.shape[0] / 3), 3)
            )
         fixed_fragment = np.reshape(
            fixed_fragment,
            newshape=(int(fixed_fragment.shape[0] / 3), 3)
         )

         list_of_fragment_pairs.append(
            (fixed_fragment, mobile_fragment, i)
         )

   #print("Die insertion indices: ", insertion_indices_for_absence_marker)
   if len(insertion_indices_for_absence_marker) != 0:
      #print("ah ja!")
      for i in insertion_indices_for_absence_marker:
         list_of_fragment_pairs.insert(i, "Missing in subject array")
   #print(list_of_fragment_pairs)

   return list_of_fragment_pairs


def _superimpose_hydrogen(
   fragment_list, subject_atom_names, subject_coords,
   ref_res_array, occurrence_array, hyd_indices, res_array_with_hyd,
   indices_of_added_hyd, amino_hydrogen, amino_hyd_bool,
   global_bond_array, id_counter, atom_count, monomer_class,
   res_pos
):
   """
   Perform superimposition between the fixed and mobile fragments with
   the final aim adding hydrogen atoms.
   The respective fixed and mobile fragment are first centered in the
   origin, i. e. the coordinate (0,0,0). Subsequently, the optimal
   rotation matrix is computed by applying the Kabsch algorithm. The
   obtained rotation matrix, together with translation, is applied in
   order to add hydrogen atoms to the fixed fragment.
   Parameters
   ----------
   fragment_list: list
      List comprising tuples which in turn comprise fragment pairs,
      i. e. a fixed fragment originating from the subject array and a
      corresponding mobile fragment originating from the reference array
      as well as the index of the respective atom of the fixed fragment
      that hydrogen is supposed to be added to.
   subject_atom_names: ndarray, dtype=str
      Array comprising the names of the atoms the subject array consists
      of.
   subject_coords: ndarray, shape=(n,3), dtype=float
      Array comprising the coordinates of the atoms comprised in the
      subject array.
   ref_res_array: :class:`AtomArray`
      The AtomArray serving as reference for hydrogen addition for the
      subject array. It is retrieved from the Chemical Component
      Dictionary.
   occurrence_array: ndarray, shape=(n,m), dtype=int
      Array storing information about how many hydrogen atoms are bound
      to a certain heavy atom of the currently considered residue.
   hyd_indices: ndarray, dtype=int
      Array comprising the indices of the hydrogen atoms from the
      reference AtomArray that are supposed to be added to the subject
      array.
   res_array_with_hyd: :class:`AtomArray`
      The newly created AtomArray for the subject array including the
      hydrogen atoms that are supposed to be added.
   indices_of_added_hyd: list
      The indices of the hydrogen atoms added to the so-called
      `res_array_with_hyd`.
   amino_hydrogen: :class:`Atom`
      The hydrogen atom of the artificially constructed fragment for
      peptide bonds.
   amino_hyd_bool: bool
      Boolean value indicating whether a considered residue is an amino
      acid and additionally has an amino group involved in a peptide
      bond, i. e. whether the residue is located in the middle or the
      end of a peptide chain.
   global_bond_array: ndarray, shape=(n,3), dtype=int
      The BondList of the `atom_array` inserted into the `add_hydrogen`
      function that is converted into an array. The `global_bond_array`
      is continuously extended by bonds of newly added hydrogen atoms as
      the algorithm iterates through the individual residues.
   id_counter: int
      Variable keeping track of of the index that is assigned to added
      hydrogen atoms when they are inserted into the
      `global_bond_array`.
   atom_count: int
      Variable keeping track of the amount of atoms comprised in the
      `atom_array_with_hyd`. It is required for assigning the final
      output array to its BondList.
   reduction: int
      Integer keeping track of the amount of heavy atoms occurring in
      the reference array from the CCD but not in the subject array.
   Returns
   -------
   res_array_with_hyd: :class:`AtomArray`
      The newly created AtomArray for the subject array including the
      hydrogen atoms that are supposed to be added.
   id_counter: int
      Variable keeping track of of the index that is assigned to added
      hydrogen atoms when they are inserted into the
      `global_bond_array`.
   global_bond_array: ndarray, shape=(n,3), dtype=int
      The BondList of the `atom_array` inserted into the `add_hydrogen`
      function that is converted into an array. The `global_bond_array`
      is continuously extended by bonds of newly added hydrogen atoms as
      the algorithm iterates through the individual residues.
   atom_count: int
       Variable keeping track of the amount of atoms comprised in the
      `atom_array_with_hyd`. It is required for assigning the final
      output array to its BondList.
   """

   central_counter = 0
   start_slice = 0
   end_slice = 0
   primary_index = 0

   heavy_atom_array = res_array_with_hyd[
      res_array_with_hyd.element != "H"
   ]
   insertion_index = heavy_atom_array.shape[0]
   #print("Insertion index: ", insertion_index)
   #print("ID counter: ", id_counter)

   for index in range(0, len(fragment_list)):
      # In case that a heavy atom carrying hydrogen is present in the
      # reference, but not in the subject array, correct slicing of
      # the array containing the hydrogen indices must be ensured by
      # continuing updating the respective counters/indices
      # Otherwise, hydrogen that is not supposed to occur in the
      # subject array anyway will be additionally placed at a wrong
      # position
      if fragment_list[index] == "Missing in subject array":
         #print("Oh, da fehlt was.")
         #print(central_counter)
         num_geminal_hyd = np.count_nonzero(
            occurrence_array[central_counter] != -1
         )
         central_counter += 1
         end_slice += num_geminal_hyd
         start_slice += num_geminal_hyd
      else:
         #print("Cool, es ist da.")
         #print(central_counter)
         fixed_fragment = fragment_list[index][0]
         mobile_fragment = fragment_list[index][1]
         heavy_atom_idx = fragment_list[index][2]

         # Move fragments in such a way that the atom carrying the
         # hydrogen is placed at the origin, i. e. (0,0,0)
         fix_central_coord = subject_coords[heavy_atom_idx]
         fix_centered = fixed_fragment - fix_central_coord
         if (
            amino_hyd_bool and subject_atom_names[heavy_atom_idx] == "N"
         ):
            mob_central_coord = amino_fragment[2]
         else:
            mob_central_coord = ref_res_array.coord[
               ref_res_array.atom_name == \
                  subject_atom_names[heavy_atom_idx]
            ][0]
         mob_centered = mobile_fragment - mob_central_coord
         
         # Performing superimposition
         rotation = _superimpose(fix_centered, mob_centered)
         num_geminal_hyd = np.count_nonzero(
            occurrence_array[central_counter] != -1
         )
         central_counter += 1
         end_slice += num_geminal_hyd
         for k in hyd_indices[start_slice:end_slice]:
            secondary_index = indices_of_added_hyd[primary_index]
            if (
               amino_hyd_bool
               and
               subject_atom_names[heavy_atom_idx] == "N"
            ):
               superimposed = amino_hydrogen.copy()
            else:
               superimposed = ref_res_array[k].copy()
            superimposed.coord -= mob_central_coord
            superimposed.coord = np.dot(
               rotation, superimposed.coord.T
            ).T
            superimposed.coord += fix_central_coord
            atom_name_to_add = ref_res_array[k].atom_name
            # Addition of atom name annotation is moved to this
            # place in order to enable removal zero rows via a
            # boolean mask
            res_array_with_hyd.atom_name[secondary_index] = \
               atom_name_to_add
            res_array_with_hyd.coord[secondary_index] = \
               superimposed.coord
            primary_index += 1
            # Adding hydrogen bonds to BondList
            global_hyd_idx = (
               id_counter + insertion_index
            )
            #print(global_hyd_idx)
            global_heavy_atom_idx = id_counter + heavy_atom_idx
            # Increase the indices of the bonds following the hydrogen
            # bond in the global bond list by 1 in order to preserve
            # consistent bond indexing
            # Residue-wise addition of hydrogen bonds and final
            # concatenation automatically corrects indices, but leads to
            # the loss of bonds between individual residues, which is
            # why indices are manually corrected
            # Third column, i. e. BondTypes, are separated from bond
            # indices and later added in order to prevent that BondTypes
            # are altered as well
            global_bond_indices = global_bond_array[:, :2]
            global_bond_types = np.atleast_2d(global_bond_array[:, 2]).T
            global_bond_indices = np.where(
               global_bond_indices >= global_hyd_idx,
               global_bond_indices + 1,
               global_bond_indices
            )
            global_bond_array = np.concatenate(
               (global_bond_indices, global_bond_types), axis=1
            )
            global_bond_array = np.concatenate(
               (global_bond_array, np.array([
                  [
                     global_heavy_atom_idx,
                     global_hyd_idx,
                     BondType.SINGLE
                  ]
               ])), axis=0
            )
            insertion_index += 1
            atom_count += 1
         start_slice += num_geminal_hyd
   # Adding number of added hydrogen atoms as well as number of heavy
   # atoms comprised in the 'res_array_with_hyd' to the `id_counter`
   id_counter += res_array_with_hyd.shape[0]
   
   return (
      res_array_with_hyd, id_counter, global_bond_array, atom_count
   )


def add_hydrogen(atom_array):
   """
   Add hydrogen atoms to a given AtomArray inserted into the function
   ('atom_array') based on comparison of the respective structure in the
   Chemical Component Dictionary.
   This function is useful for structures that have been determined
   experimentally, e. g. by x-ray crystallography and therefore lack
   the position of hydrogen atoms. This is oftentimes the case with
   protein structures from the Protein Data Bank, making the employment
   of these structures unfavourable, e. g. in molecular dynamics
   simulations, where energy contributions arising from interaction with
   hydrogen is crucial. This function, however, meets this deficit of
   experimentally determined structures by adding hydrogen to them based
   on comparison with the respective structure found in the Chemical
   Component Dictionary. This circumstance simultaneously poses a
   restriction, so that hydrogen addition to rather exotic compounds
   might fail. However, it splendidly works for biological compounds as
   proteins or nucleic acids since the 20 canonical amino acids
   respectively the nucleotides incorporated in DNA and RNA are core
   entries of the Chemical Component Dictionary.
   For peptide bonds, hydrogen addition is based on values for bond
   angles and bond lengths of the peptide bond taken from literature.
   [1]_
   Parameters
   ----------
   atom_array: :class:`AtomArray`
      A structure lacking hydrogen atoms and which hydrogen atoms are
      supposed to be added to. It must consist of entries of the
      Chemical Component Dictionary in order for the procedure to work.
   Returns
   -------
   atom_array_with_hyd: :class:`AtomArray`
      The structure inserted into the function with added hydrogen
      atoms. Note that the position of rotatable hydrogen atoms is
      optimized but requires application of the function
      `relax_hydrogen`.
   References
   ----------
   .. [1] H-D Jakubke and H Jeschkeit,
      "Aminosaeuren, Peptide, Proteine.", 96 - 97
      Publisher Chemistry, Weinheim 1982
   """

   if not isinstance(atom_array, AtomArray):
      raise ValueError("Input must be AtomArray")

   if atom_array.bonds is None:
        raise AttributeError(
            f"The input AtomArray doesn't possess an associated "
            f"BondList."
        )
   
   has_charge_annotation = False
   has_occupancy_annotation = False
   has_b_factor_annotation = False

   # The initial atom is just inserted in order to enable concatenation
   # and is later neglected by slicing
   atom_array_with_hyd = array([Atom([0,0,0])])

   # Optional annotation categories present in the input must be added
   # to the output as they otherwise get lost in the process of
   # concatenation
   try:
      atom_array.charge
   except AttributeError:
      pass
   else:
      atom_array_with_hyd.add_annotation("charge", "int")
      has_charge_annotation = True

   try:
      atom_array.occupancy
   except AttributeError:
      pass
   else:
      atom_array_with_hyd.add_annotation("occupancy", "float")
      has_occupancy_annotation = True
   
   try:
      atom_array.b_factor
   except AttributeError:
      pass
   else:
      atom_array_with_hyd.add_annotation("b_factor", "float")
      has_b_factor_annotation = True

   has_key_error = False
   unparametrized_residues = []
   res_starts = get_residue_starts(
      atom_array, add_exclusive_stop=True
   )
   first_res_start = res_starts[0]
   # Subtracting two from length because `res_starts` array includes
   # exclusive stop
   last_res_start = res_starts[len(res_starts) - 2]

   # Determine which residues are amino acids and which ones are
   # nucleotides in order to account for specialties regarding hydrogen
   # addition at bonds between different residues
   # Explicit determination of nucleotides is not necessary since the
   # exclusion principle is applied in the gathering of hydrogen indices
   # (if a residue is not an amino acid, but has the `hetero`
   # annotation False, it can be identified as nucleotide)
   aa_bool_array = filter_amino_acids(atom_array)

   atom_array_coords = atom_array.coord

   # Global bond list is adopted and hydrogen bonds are added to it as
   # retrieving bonds residue-wise would lead to the loss of bonds
   # between individual residues
   global_bond_array = atom_array.bonds.as_array()
   atom_count = atom_array.bonds.get_atom_count()
   id_counter = 0

   for counter, start in enumerate(res_starts):
      if counter == res_starts.shape[0] - 1:
         break
      subject_start = start
      subject_end = res_starts[counter + 1]
      subject_array = atom_array[subject_start:subject_end]

      # Obtaining reference AtomArray from the Chemical Components
      # Dictionary in order to know where to add hydrogen
      res_name = atom_array.res_name[start]
      print("Aktuelle Aminosäure: ", res_name)
      try:
         ref_res_array = residue(res_name)
      except KeyError:
         unparametrized_residues.append(res_name)
         has_key_error = True
         atom_array_with_hyd += subject_array
         continue
      res_elements = np.unique(ref_res_array.element)

      if "H" not in res_elements:
         atom_array_with_hyd += subject_array
         continue

      subject_array = subject_array[standardize_order(subject_array)]

      mobile_bond_list = ref_res_array.bonds
      fixed_bond_list = subject_array.bonds
      
      chain_ids = atom_array.chain_id
      considered_chain_id = subject_array.chain_id[0]
      aa_bool_value = aa_bool_array[start]
      het_bool_value = subject_array.hetero[0]
      
      hyd_func_return = _gather_hyd_indices(
         counter, start, first_res_start, last_res_start, res_starts,
         chain_ids, ref_res_array, considered_chain_id, het_bool_value,
         aa_bool_value, res_name
      )
      hyd_indices = hyd_func_return[0]
      amino_hyd_bool = hyd_func_return[1]
      monomer_class = hyd_func_return[2]
      res_pos = hyd_func_return[3]
      
      # Amount of hydrogen atoms to add as well as amount of atoms
      # comprised in the `subject_array` are determined in order to
      # create a new array for the residue with added hydrogen
      num_heavy_atoms = subject_array.shape[0]
      amount_of_hyd_to_add = hyd_indices.shape[0]
      atom_sum = num_heavy_atoms + amount_of_hyd_to_add

      res_array_with_hyd =  _create_atom_array_including_hyd(
         num_heavy_atoms, amount_of_hyd_to_add, atom_sum,
         subject_array, has_charge_annotation, has_occupancy_annotation,
         has_b_factor_annotation
      )
     
      # For adding the coordinates of the hydrogen atoms at the correct
      # indices, a list containing the indices of the added hydrogen
      # atoms is created
      indices_of_added_hyd = list(range(num_heavy_atoms, atom_sum))

      build_func_return = _build_occurrence_array(
         hyd_indices, mobile_bond_list
      )
      occurrence_array = build_func_return[0]
      unique_atom_ind_car_hyd = build_func_return[1]

      atom_names_carrying_hyd = []
      atom_names = ref_res_array.atom_name
      for i in unique_atom_ind_car_hyd:
         atom_names_carrying_hyd.append(atom_names[i])
      
      subject_atom_names = subject_array.atom_name
      subject_coords = subject_array.coord
      subject_elements = subject_array.element
      subject_length_zero_axis = subject_array.shape[0]
      ref_atom_names = ref_res_array.atom_name
      ref_coords = ref_res_array.coord
      ref_atom_names = ref_res_array.atom_name
      ref_coords = ref_res_array.coord

      fragment_list = _construct_fragments(
         counter, amino_hyd_bool, res_starts, fixed_bond_list,
         subject_length_zero_axis, subject_atom_names, subject_coords,
         atom_names_carrying_hyd, ref_atom_names, ref_coords,
         atom_array_coords, subject_elements, monomer_class, res_pos
      )

      superimpose_func_return = _superimpose_hydrogen(
         fragment_list, subject_atom_names, subject_coords,
         ref_res_array, occurrence_array, hyd_indices,
         res_array_with_hyd, indices_of_added_hyd, amino_hydrogen,
         amino_hyd_bool, global_bond_array, id_counter, atom_count,
         monomer_class, res_pos
      )
      res_array_with_hyd = superimpose_func_return[0]
      id_counter = superimpose_func_return[1]
      #id_counter += subject_array.shape[0]
      global_bond_array = superimpose_func_return[2]
      atom_count = superimpose_func_return[3]

      # Removing possible zero rows arising from heavy atoms carrying
      # hydrogen that are present in the reference array but not in the
      # subject array
      res_array_with_hyd = res_array_with_hyd[
         res_array_with_hyd.atom_name != ""
      ]
      # Performing concatenation
      atom_array_with_hyd += res_array_with_hyd

   if has_key_error:
      # Using NumPy's 'unique' function to ensure that each residue only
      # occurs once in the list
      unique_list = np.unique(unparametrized_residues)
      # Considering proper punctuation for the warning string
      warnings.warn(
         f"Entries in the Chemical Components Dictionary are not "
         f"available for the following residues: \n"
         f"{', '.join(unique_list)}. \n"
         f"Therefore, hydrogen addition for those residues is omitted.",
         UserWarning
      )
   atom_array_with_hyd = atom_array_with_hyd[1:]
   atom_array_with_hyd.bonds = BondList(
      atom_count, global_bond_array
   )
   atom_array_with_hyd.box = atom_array.box
   return atom_array_with_hyd