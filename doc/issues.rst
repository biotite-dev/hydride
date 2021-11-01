.. This source code is part of the Hydride package and is distributed
   under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
   information.

Common issues
=============

A group has an unexpected number of hydrogen atoms, e.g. a carboxy group is protonated
--------------------------------------------------------------------------------------

Probably, the formal charge for the concerning heavy atom is incorrect.
If the formal charge of the oxygen atom of a carboxy group is set to 0, a
protonated form is assumed.

The reason for incorrect charges is simply that the structure files contain
improper charge values.
For amino acids the charges can be corrected using the ``--charges``/``-c``
parameter from the *Hydride* command line interface, or via the
:func:`estimate_amino_acid_charges()` function from the Python API.
For all other molecules, this problem can be solved by fixing the input
structure file or by editing the ``charge`` attribute of the input
:class:`AtomArray` in the Python API.


A fragment is missing for a heavy atom and no hydrogen atom can be assigned to it
---------------------------------------------------------------------------------

Although, most molecules are covered by the fragment library, some uncommon
groups within a molecule may be not.
If this happens, *Hydride* will give you a warning
However, most of these exotic groups do not bear hydrogen atoms,
so you can ignore the warning in those cases.

If an expected hydrogen atom is missing, you can add a *template* of such a
molecule to the fragment library.
This template is a molecular model of this molecule containing hydrogen atoms.
Such a template structure file for a molecule can be e.g. downloaded from a
ligand database.
In the command line interface the template structure file is provided via
the ``--fragments``/``-f`` parameter.
Note that the *MMTF*, *MOL* and *SDF* formats are currently the only
reasonable formats, since the bond information can be read exclusively from
these formats.
In the Python API the template can be added to the standard fragment library.

.. code-block:: python

   library = copy.deepcopy(hydride.FragmentLibrary.standard_library())
   library.add_molecule(ligand)
   structure_with_h, _ = hydride.add_hydrogen(structure, fragment_library=library)


Terminal hydrogen coordinates remain unchanged after relaxation
---------------------------------------------------------------

Note that the relaxation only rotates about bonds of terminal heavy atoms
carrying hydrogen atoms.

If you are using the Python API a possible reason for this issue are
undefined bond types (:attr:`BondType.ANY`) in the :class:`BondList` of the
input :class:`AtomArray`.
These appear, if the :class:`BondList` was created using
:func:`connect_via_distances()`.