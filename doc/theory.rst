.. This source code is part of the Hydride package and is distributed
   under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
   information.

.. _theory:

Theoretical background
======================

*Hydride* employs two steps for accurate calculation of hydrogen positions for
a target molecule.

For each heavy atom, the number of bonded hydrogen atoms and their positions
are calculated based on molecular geometry of reference molecules in the first
step, as these reference molecules have known hydrogen positions.

After initial placement of hydrogen atoms, most of their positions should be
accurate, as they are constrained by the position of the respective bonded
heavy atom, since the bond lengths and angles are (approximately) constant.
However, there are exceptions:
Terminal heavy atoms connected with a single bond to the remaining molecule
(e.g. a hydroxy or methyl group) have no unambiguous hydrogen positions,
as they are able to rotate about this single bond (see below).

.. image:: /images/rotation_freedom.png
   :width: 800

Hence, the positions of hydrogen atoms bonded to these terminal heavy atoms
are relaxed in a second step.
This step is completely optional:
If it is omitted, the returned molecular model is not energy minimized, so
some (non-covalent) hydrogen bonds or *Van-der-Waals* interactions might be
missing.
Nevertheless, this molecular model is still reasonable with respect to
bond lengths and angles, but some atom clashes might appear.


Hydrogen addition
-----------------

The addition of hydrogen atoms is based on molecular geometries of groups from
reference molecules.
For this purpose the reference molecules are compiled into a
*fragment library*:

Each molecule is split into *fragments*, one for each heavy atom in the
molecule.
Each fragment consists of

   - the element, charge and coordinates of the central heavy atom *(blue)*,
   - the coordinates of the bonded hydrogen atoms *(white)*,
   - the coordinates of the bonded heavy atoms
     and the order of the bonds to them (*gray*) and
   - the chirality of the fragment, if applicable.

These fragments are stored in the aforementioned fragment library,
a data structure that maps the combination of a fragment's

   - central atom element,
   - central atom charge,
   - chirality and
   - order of bonds to connected heavy atoms

(called *library key* from now on) to

   - the coordinates of heavy atoms connected to the central atom and
   - the coordinates of hydrogen atoms connected to the central atom.

The coordinates of the fragment's central atom are not saved, as the
coordinates of the fragment are translated, so that the central atom lies
always in the coordinate origin.
Duplicate library keys are ignored *(slightly transparent)* and hence
will not be part of the fragment library.

.. image:: /images/library.png
   :width: 800

In the figure shown above the library contains only fragments from benzene and
isobutylene.
However, *Hydride*'s default fragment library contains fragments from
all compounds from the
`Chemical Component Dictionary <https://www.wwpdb.org/data/ccd>`_.

The target molecule, in the shown example we use toluene (*red*), is split into
fragments in a similar fashion.
But in contrast to the molecules for the fragment library the created
fragments of course miss hydrogen atoms.

.. image:: /images/target_fragments.png
   :width: 800

Now for each target molecule fragment *(red)*, or *target fragment* in short,
the matching fragment from the fragment library *(blue)* is selected.
Although the target fragment has no hydrogen atoms, this works, because the
hydrogen atoms are not part of the library key.

In the next step, the target fragment coordinates are translated so that the
central atom lies in the coordinate origin, as does the library fragment.
Then the library fragment is superimposed onto the target fragment by rotation
about the coordinate origin [1]_ [2]_.
Probably, the two fragments will not overlap perfectly, but the superimposition
will minimize the *root-mean-square deviation* between the fragments.
In the final step the library fragment is moved back to the original position
of the target fragment simply by applying the reversed translation vector.
The hydrogen coordinates of the transformed library fragment *(encircled)* are
the desired coordinates for the target fragment.

.. image:: /images/superimposition.png
   :width: 800

If the library does not contain a match for a target molecule fragment, the
algorithm is unable to assign hydrogen atoms to this central atom.
Hence, it is desirable to have a large fragment library to cover a broad range
of different fragments.

After this procedure is finished for each target fragment, the obtained
hydrogen positions are adopted by the target molecule.
*(The hydrogen position of the previous figure is encircled again.)*

.. image:: /images/position_adoption.png
   :width: 800


Hydrogen relaxation
-------------------

Lorem ipsum


References
----------

.. [1] W Kabsch,
   "A solution for the best rotation to relate two sets of vectors."
   Acta Cryst, 32, 922-923 (1976).
   
.. [2] W Kabsch,
   "A discussion of the solution for the best rotation to relate
   two sets of vectors."
   Acta Cryst, 34, 827-828 (1978).