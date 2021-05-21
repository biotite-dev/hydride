.. This source code is part of the Hydride package and is distributed
   under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
   information.

Getting started
===============

*Hydride* adds hydrogen atoms to molecular models where these are
missing, for examples structures that were determined via low-resolution X-ray
diffraction.
Instead of using force field parameters to place hydrogen atoms at their
optimum bond angle and distance, it uses a large *fragment library* to perform
this task (see :ref:`theory`).
This allows *Hydride* to assign hydrogen atoms to a broad range of molecules.

For each heavy atom, i.e. an atom that is not hydrogen, a fragment is created.
This fragment contains the following information:

   - The element of the heavy atom
   - The charge of the heavy atom
   - The enantiomer, if applicable
   - The number of bonds and bond orders to connected heavy atoms

This fragment is searched in the fragment library, that contains for each
fragment the number of bonded hydrogen atoms and their positions relative to
the position of the heavy atom and its bond partners.

The fragment library is compiled from the
`Chemical Component Dictionary <https://www.wwpdb.org/data/ccd>`_,
containing all molecules that appear in the PDB.
This means that hydrogen atoms can be assigned to any molecule/residue from the
PDB and all molecules that share comparable groups.
Special cases, which are not covered by the standard fragment library, can be
resolved by adding a version of this molecule with hydrogen atoms to the
fragment library.

After the hydrogen atoms are added, the conformation is not optimal, yet.
In a second step *Hydride* relaxes the dihedral angles of terminal heavy atoms
carrying hydrogen atoms.
This method minimizes steric clashes and restores hydrogen bonds.

Installation
------------

In order to use *Hydride* you need to have Python (at least 3.7) installed.

You can install *Hydride* via

.. code-block:: console

   $ pip install hydride

Alternatively, you can check out the
`offical repository <https://github.com/biotite-dev/hydride>`_
and build and install the package via

.. code-block:: console

   $ pip install .

Note that this way the installation may take a few minutes, as 
the fragment library is built and the C-extensions are compiled.


Basic usage
-----------

The most simple invocation of the *Hydride* command line program is

.. code-block:: console

   $ hydride -i input_structure.pdb -o output_structure.pdb

*Hydride* reads a molecular model without hydrogen atoms from
``input_structure.pdb``, adds hydrogen atoms and writes the resulting model to
``output_structure.pdb``.
If hydrogen atoms remain in the input structure, these are automatically
removed.
*Hydride* supports the *PDB*, *PDBx/mmCIF*, *MMTF*, *MOL* and *SDF*
format.

If no input structure file path is given, the file is read from *STDIN*.
In this case the the format cannot be inferred from the file extension, so the
format must be explicitly given via the ``-I`` parameter.
Conversely, the structure is written to STDOUT and the ``-O`` parameter is
required, if no output file path is given.
This way *Hydride* can be used in a chain of commands.

.. code-block:: console

   $ some_tool | hydride | some_other_tool > output_structure.pdb

All command line parameters and their usage is listed in depth in
:ref:`cli`.

Often structure file miss proper formal charges values for each atom,
leading to false additions of hydrogen atoms, for example protonated carboy
groups.
This problem can be mitigated at least for amino acids by recalculating
formal charges at a given pH value.

.. code-block:: console

   $ hydride --charges 7.0 -i input_structure.pdb -o output_structure.pdb

Note that only formal charges in amino acids are updated this way.
For all other molecules the formal charge values from the input structure file
is taken.
Furthermore, the underlying method assigns charges based on the pK values of
the free amino acid.
The chemical environment of a residue is not taken into account.

Python API
----------

*Hydride* is not only command line program, but also a Python library
extending on :class:`AtomArray` objects from the
`Biotite package <https://www.biotite-python.org/>`_.

The :mod:`hydride` package provides two central functions:
:func:`add_hydrogen()` and :func:`relax_hydrogen()`.
While the former adds hydrogen atoms with appropriate bond angles and
lengths using the fragment library, the latter takes a structure
containing hydrogen atoms and optimizes the hydrogen positions by
rotating about dihedral angles of terminal groups.
Usually, both functions are called subsequently, for example:

.. code-block:: python

   hydrogenated_atoms, _ = hydride.add_hydrogen(heavy_atoms)
   hydrogenated_atoms.coord = hydride.relax_hydrogen(hydrogenated_atoms)

but these functions can also be used independently:
:func:`relax_hydrogen()` can be omitted, if a relaxation is not necessary
for the use case.
Conversely, :func:`add_hydrogen()` does not need to be called if the
:class:`AtomArray` already contains hydrogen atoms, but merely steric clashes
should be resolved.