.. This source code is part of the Hydride package and is distributed
   under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
   information.

.. _cli:

Command line interface
======================

A short description of all arguments is listed via

.. code-block:: console

   $ hydride -h

With the ``--verbose``/``-v`` parameter *Hydride* prints the complete stack
traceback of a potential error.


Structure input and output
--------------------------

``--infile``/``-i`` and ``-outfile``/``-o`` define the paths of the input or
output structure file, respectively.
*Hydride* supports the *PDB*, *PDBx/mmCIF*, *MMTF*, *MOL* and *SDF*
format.
By default, the format is determined from the file extension, but it can also
be explicitly provided via the ``--informat``/``-I`` or
``--outformat``/``-O`` argument, respectively.

If no input/output file path is given, *Hydride* reads the file content from
*STDIN* and writes the result to *STDOUT*.
In this case ``--informat``/``-I`` or ``--outformat``/``-O`` must be provided,
respectively.

If the input structure file contains multiple models, the model number to be
used is specified with ``--model``/``-m``. By default, the first model is used.

The addition of hydrogen atoms requires complete information about the
bonds between atoms.
Currently, this information can only be read from *PDB*, *MMTF*, *MOL* and
*SDF* files.
If bond information is absent, *Hydride* automatically connects
atoms based on the molecule/residue name and the atom names.
However, the automatic bond detection only works for molecules in the
`Chemical Component Dictionary <https://www.wwpdb.org/data/ccd>`_.
For all other molecules it is required to assign bonds manually and
perform the hydrogen addition via the :ref:`api`,

Frequently, structure files have no correct values for formal charges.
Instead the charge for all atoms is often given as ``0``, even if the atom
would be charged at physiological pH values.
As *Hydride* assigns hydrogen atoms based on the given formal charge of a
heavy atom, this can lead to e.g. protonated carboxy groups or deprotonated
amino groups.
*Hydride* recalculates the formal charges of atoms in amino acids, if the
``--charges``/``-c`` parameter together with the desired pH value is given.
Note that only formal charges in canonical amino acids are updated this way.
For all other molecules the formal charge values from the input structure file
is taken.
Furthermore, the underlying method assigns charges based on the pK values of
the free amino acid.
The chemical environment of a residue is not taken into account.


Hydrogen addition
-----------------

By default, hydrogen atoms are added to each applicable heavy atom.
Residues can be ignored by specifying each of them via the
``--ignore``/``-g`` argument.
To ignore the residues with residue ID 5 and 10 from chain ``A`` for example,
you can run

.. code-block:: console

   $ hydride -g A 5 -g A 10 -i input_structure.pdb -o output_structure.pdb

To add a hydrogen atom to a heavy atom, the fragment for the respective
heavy atom is searched in the *fragment library*.
The atom names for the added hydrogen atoms are taken from the *name library*
based on the name of the molecule and the heavy atom.
If the name library does not support a molecule, the hydrogen atom names
are generated based on a defined scheme.
If the fragment library misses the required fragment, no hydrogen is added
to the relevant heavy atom.

While the default fragment library comprise all molecules from the
`Chemical Component Dictionary <https://www.wwpdb.org/data/ccd>`_,
the default name library only contains names for the most common residues -
amino acids and nucleotides.
If the fragment library may miss fitting fragments for your molecule
(which rarely happens) or you want to ensure canonical hydrogen atom naming
for you molecule, can add this molecule (including hydrogen atoms) to both
libraries by providing a path to a corresponding structure file via
``--fragments``/``-f``.
Analogous to the input and output file parameters, the file format can be
specified with ``--fragformat``/``-f``.
Note that the file must contain proper bond information and correct formal
charges, so effectively a *MMTF*, *MOL* and *SDF* must be supplied.

By default, *Hydride* does not consider periodic boundary conditions,
as they appear e.g. in MD simulations.
This can be changed with the ``--pbc``/``-p`` option.
The required box vectors are read from the input structure file.


Relaxation
----------

After hydrogen atoms are placed using proper bond lengths and angles,
a short relaxation is performed.
This step reduces steric clashes and forms hydrogen bonds.
This step can be omitted with the ``--no-relax`` option.
By default, the relaxation runs until a local energy optimum is reached.
The number of relaxation steps can be limited with the
``--iterations``/``-n`` argument.
Setting this argument may decrease the runtime of the program but also
reduces the accuracy.
In each relaxation step, the dihedral angles of terminal heavy atoms
carrying hydrogen atoms are rotated by a defined increment.
By default, this increment is 10°, but this value can be reduced for more
accurate results or increased to shorten the runtime with the
``--angle-increment``/``-a`` argument.