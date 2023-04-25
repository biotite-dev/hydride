.. _api:

Python API
==========

.. currentmodule:: hydride

In summary, *Hydrides* provides two functionalities:

   - Adding hydrogen atoms to a structure with missing hydrogen atoms
   - Relaxing hydrogen positions in a structure with existing hydrogen atoms

Typically both functionalities are combined:
In a first step hydrogen atoms are added to a molecular model and then the
hydrogen atoms are relaxed in the resulting model.

Note that there are legit cases for using only one of these functionalities:
The initial placement of the hydrogen atoms might be sufficient for your use
case, so no relaxation of the rotatable hydrogen atoms is required.
Or you might have an existing structure containing hydrogen atoms and you want
to reduce atom clashes and reconstruct hydrogen bonds.

To illustrate the effects of each step, this tutorial will add hydrogen atoms
to a molecular model of *2-nitrophenol*, where hydrogen atoms are missing.
*Hydride* expands on data types from
`Biotite <https://www.biotite-python.org/>`_, so an :class:`AtomArray` is used
to represent a molecular model, including it atom descriptions, coordinates and
bonds.
If you are new to *Biotite*, have a look at the
`official documentation <https://www.biotite-python.org/>`_.

For this example the structure of *2-nitrophenol* is loaded from a *MOL* file.

.. code-block:: python
   
    import biotite.structure.io.mol as mol

    mol_file = mol.MOLFile.read("path/to/nitrophenol.mol")
    molecule = mol_file.get_structure()
    print(type(molecule))
    print()
    print(molecule)
    print(molecule.bonds.as_array())

.. code-block:: none

    <class 'biotite.structure.AtomArray'>

            0             O         2.244    0.777    0.012
            0             N         1.635   -0.278    0.006
            0             O         2.245   -1.333   -0.004
            0             C         0.156   -0.278    0.006
            0             C        -0.539    0.923    0.017
            0             O         0.138    2.100    0.029
            0             C        -0.535   -1.474   -0.012
            0             C        -1.919   -1.475   -0.013
            0             C        -2.613   -0.280    0.003
            0             C        -1.926    0.919    0.016

.. image:: /images/api_01.png
   :align: center


As you can see, this molecule does not contain any hydrogen atoms, yet.
This circumstance is changed by :func:`add_hydrogen()`.
Note that the input :class:`AtomArray` must not already contain hydrogen atoms.
If at some places hydrogen atoms already exist, these must be removed.
Furthermore, the :class:`AtomArray` must have an associated ``charge``
attribute, containing the formal charges, and an associated :class:`BondList`.

.. code-block:: python
   
    import hydride

    # Remove already present hydrogen atoms (only necessary in rare cases)
    molecule = molecule[molecule.element != "H"]
    # Add hydrogen atoms
    molecule, mask = hydride.add_hydrogen(molecule)
    print(molecule)
    print()
    print(mask)

.. code-block:: none

            0             O         2.244    0.777    0.012
            0             N         1.635   -0.278    0.006
            0             O         2.245   -1.333   -0.004
            0             C         0.156   -0.278    0.006
            0             C        -0.539    0.923    0.017
            0             O         0.138    2.100    0.029
            0             C        -0.535   -1.474   -0.012
            0             C        -1.919   -1.475   -0.013
            0             C        -2.613   -0.280    0.003
            0             C        -1.926    0.919    0.016
            0             H        -0.180    2.650    0.766
            0             H        -0.034   -2.435   -0.026
            0             H        -2.419   -2.437   -0.026
            0             H        -3.696   -0.236    0.006
            0             H        -2.511    1.832    0.025

        [ True  True  True  True  True  True  True  True  True  True False False
         False False False]

.. image:: /images/api_02.png
   :align: center


:func:`add_hydrogen()` returns two objects:
The :class:`AtomArray`, including the input atoms plus the added hydrogen
atoms, and a boolean mask (a :class:`numpy.ndarray`), that selects the original
input atoms.
This means, if the mask is applied to the returned :class:`AtomArray`, the
hydrogen atoms are removed again.

.. code-block:: python
   
    print(molecule[mask])

.. code-block:: none

            0             O         2.244    0.777    0.012
            0             N         1.635   -0.278    0.006
            0             O         2.245   -1.333   -0.004
            0             C         0.156   -0.278    0.006
            0             C        -0.539    0.923    0.017
            0             O         0.138    2.100    0.029
            0             C        -0.535   -1.474   -0.012
            0             C        -1.919   -1.475   -0.013
            0             C        -2.613   -0.280    0.003
            0             C        -1.926    0.919    0.016

Regarding the order of atoms in the returned :class:`AtomArray`, the hydrogen
atoms are placed behind the heavy atoms for each residue/molecule separately. 

Most hydrogen atoms are already placed correctly with respect to bond angles
and lengths.
However, the hydrogen atom in the hydroxy group is not in a energy-minimized
state, as an intramolecular hydrogen bond to the nitro group would be expected.
The energy minimization is performed with :func:`relax_hydrogen()`.

.. code-block:: python
   
    molecule.coord = hydride.relax_hydrogen(molecule)

.. image:: /images/api_03.png
   :align: center

:func:`relax_hydrogen()` is able to optimize the position of hydrogen atoms at
terminal groups.
In this case it was able to orient the hydrogen atom at the hydroxy group
towards the nitro group to form a hydrogen bond.


Custom fragment and atom name libraries
---------------------------------------

:func:`add_hydrogen()` uses a :class:`FragmentLibrary` to find the correct
number and positions of hydrogen atoms for each heavy atom in the input
:class:`AtomArray` and a :class:`AtomNameLibrary` to find
the correct atom name (e.g. ``'HA'``).
The default fragment library (:meth:`FragmentLibrary.standard_library()`)
contains all fragments from the entire
`Chemical Component Dictionary <https://www.wwpdb.org/data/ccd>`_ and the
default atom name library (:meth:`AtomNameLibrary.standard_library()`)
contains atom names for all amino acids and standard nucleotides.
Furthermore, the atom name library provides a reasonable naming scheme for
unknown residues/molecules.
Therefore, it is seldom required to provide a custom library.
However, in rare cases the fragment library might not contain a required
fragment or the automatic atom naming is not sufficient.

A custom :class:`FragmentLibrary` can be created either via its constructor
or by copying the default library.

.. code-block:: python

    library = copy.deepcopy(hydride.FragmentLibrary.standard_library())

Then molecular models containing hydrogen atoms are added to the library.
Internally, a model is split into fragments and these fragments are stored in
the library's internal dictionary.
Hence, such a molecular model act as *template* for the later hydrogen
addition.

.. code-block:: python

    library.add_molecule(template_molecule)

Finally the fragment library can be provided to :func:`add_hydrogen()`.

.. code-block:: python

    hydride.add_hydrogen(molecule, fragment_library=library)

|

In a similar way, a custom :class:`AtomNameLibrary` can be created, filled with
template molecules and used in :func:`add_hydrogen()`.

.. code-block:: python

    library = copy.deepcopy(hydride.AtomNameLibrary.standard_library())
    library.add_molecule(template_molecule)
    hydride.add_hydrogen(molecule, name_library=library)


Handling periodic boundary conditions
-------------------------------------

By default, :func:`add_hydrogen()` and :func:`relax_hydrogen()` do not take
periodic boundary conditions into account, as they appear e.g. in MD
simulations.
Consequently, interactions over the periodic boundary are not taken into
account and, more importantly, hydrogen atoms are not placed correctly, if the
molecule is divided by the boundary.
To tell *Hydride* to consider periodic boundary conditions the `box` parameter
needs to be provided.
The value can be either a an array of the three box vectors or ``True``, in
which case the box is taken from the input structure.

.. code-block:: python

    molecule, _ = hydride.add_hydrogen(molecule, box=True)
    molecule.coord = hydride.relax_hydrogen(molecule, box=True)

Note that this slows down the addition and relaxation procedure.


Tweaking relaxation speed and accuracy
--------------------------------------

Usually, the relaxation is fast and accurate enough for most applications.
However, the user is able to adjust some parameters to shift the
speed-accuracy-balance to either side.

By default, :func:`relax_hydrogen()` runs until the energy of the conformation
does not improve anymore.
However, the maximum number of iterations can also be given with the
``iterations`` parameter.
If the number of relaxation steps reaches this value, the relaxation terminates
regardless of whether an energy minimum is attained. 

The ``angle_increment`` parameter controls the angle by which each rotatable
bond is changed in each relaxation iteration. Lowering this value leads to a
higher *resolution* of the returned conformation, but more iterations are
required to find the optimum.


Observing the relaxation
------------------------

In order to get insights into the course of the relaxation, the user can
optionally obtain the coordinates and energy values for each iteration via the
``return_trajectory`` and ``return_energies`` parameters, respectively.

.. code-block:: python

    import matplotlib.pyplot as plt

    molecule_with_h, mask = hydride.add_hydrogen(molecule)
    coord, energies = hydride.relax_hydrogen(
        molecule_with_h, return_trajectory=True, return_energies=True
    )
    print(coord.shape)
    print(energies.shape)

    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    ax.plot(energies)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    fig.tight_layout()

.. image:: /images/api_04.png
   :align: center


Handling missing formal charges
-------------------------------

Both :func:`add_hydrogen()` and :func:`relax_hydrogen()` require associated
formal charges for correct fragment identification or the electrostatic
potential, respectively.
However, for many entries in the *RCSB PDB* they are not properly set.
At least for canonical amino acids this issue can be remedied with
:func:`estimate_amino_acid_charges()`.
This function calculates the formal charges of atoms in amino acids based on a
given pH value.

.. code-block:: python

    charges = hydride.estimate_amino_acid_charges(molecule, ph=7.0)
    molecule.set_annotation("charge", charges)


Handling missing bonds
----------------------

The input :class:`AtomArray` is also required to have an associated
:class:`BondList`.
Unfortunately, this information cannot be retrieved from *PDB* or *PDBX/mmCIF*
files.
If the residues in the input structure are part of the
*Chemical Component Dictionary*, the :class:`BondList` can be created with
:func:`biotite.structure.connect_via_residue_names()`.

.. code-block:: python

    import biotite.structure as struc

    molecule.bonds = struc.connect_via_residue_names(molecule)

|

Classes and functions
---------------------

.. autoclass:: FragmentLibrary
    :members:
    :undoc-members:
    :inherited-members:

|

.. autoclass:: AtomNameLibrary
    :members:
    :undoc-members:
    :inherited-members:

|

.. autofunction:: add_hydrogen

|

.. autofunction:: relax_hydrogen

|

.. autofunction:: estimate_amino_acid_charges
