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

Most hydrogen atoms are accurately placed after this step, since the position
of the bonded heavy atom is fixed and the angles and lengths of the bonds
connecting the hydrogen and heavy atoms can be considered constant.
However, heavy atoms that are connected to the rest of the molecule with
a single bond (e.g. a hydroxy or methyl group) have ambiguous hydrogen
positions, due to its rotational freedom.

.. image:: /images/rotation_freedom.png
   :width: 800

Hence, the positions of hydrogen atoms bonded to these terminal heavy atoms
are relaxed in a second step.
This step is completely optional:
If it is omitted, the returned molecular model is not energy minimized, so
some (non-covalent) hydrogen bonds or *Van-der-Waals* interactions might be
missing.
Nevertheless, this molecular model is still reasonable with respect to
bond lengths and angles, but some steric clashes might appear.


Hydrogen addition
-----------------

The information about the number and positions of hydrogen atoms for a given
heavy atom is leveraged from known molecular geometries of reference
molecules containing hydrogen.
For example, to be able to add hydrogen atoms to the carbon atom of a methyl
group, *Hydride* needs a reference molecule containing a methyl group with
hydrogen.

These reference molecules are compiled into a so called *fragment library*:
Each reference molecule is segmented into its *fragments*, one fragment
for each heavy atom in the molecule:
Each fragment contains the central heavy atom *(blue)* and all heavy atoms
*(gray)* and hydrogen atoms *(white)* directly bonded to it.
A fragment is characterized by its *library key*, a combination of

   - the element of the central heavy atom,
   - its charge,
   - its chirality,
   - and the order of bonds to connected heavy atoms

The coordinates for each fragment in the reference molecules are added to the
fragment library and can thereupon be accessed via its *library key*.
The coordinates of the fragment's central atom are not saved,
since the coordinates are translated to place the central heavy atom into
the coordinate origin. 
Note, that duplicate library keys are ignored *(slightly transparent)* and
hence will not be part of the fragment library.

.. figure:: /images/library.png
   :width: 800

   *Library creation with fragments from benzene and isobutylene.*

In the figure shown above the library contains only fragments from benzene and
isobutylene.
However, *Hydride*'s default fragment library contains fragments from
all compounds from the
`Chemical Component Dictionary <https://www.wwpdb.org/data/ccd>`_.

The target molecule, in the shown example we use toluene (*red*), is also split
into fragments.
But in contrast to the molecules, that were used to create the fragment
library, these fragments of course miss hydrogen atoms.

.. figure:: /images/target_fragments.png
   :width: 800

   *Fragmentation of the target molecule toluene.*

Nevertheless, a library key can also be formed for each of these
*target fragments*.
Hence, the fragment library is accessed with this key to obtain a corresponding
*library fragment* with hydrogen atoms.

In the next step, the target fragment coordinates are translated to place
its central atom in the coordinate origin, too.
Now the library fragment is rotated about the coordinate origin to
minimize the atom atom distances to the target fragment [1]_ [2]_.
In the final step the library fragment is moved back to the original position
of the target fragment using the reversed translation vector.
The hydrogen coordinates in this transformed library fragment *(encircled)*
are the desired coordinates for the target fragment.


central atom lies in the coordinate origin, as does the library fragment.
Then the library fragment is superimposed onto the target fragment by rotation
about the coordinate origin [1]_ [2]_.
Probably, the two fragments will not overlap perfectly, but the superimposition
will minimize the *root-mean-square deviation* between the fragments.
In the final step the library fragment is moved back to the original position
of the target fragment simply by applying the reversed translation vector.
The hydrogen coordinates of the transformed library fragment *(encircled)* are
the wanted hydrogen coordinates for this target fragment.

.. figure:: /images/superimposition.png
   :width: 800

   *Superimposition of a library fragment onto a target fragment.*

If the library does not contain an entry for the library key of target
fragment, the algorithm is unable to predict hydrogen atoms for this heavy
atom.
Therefore, the default fragment library of *Hydride* contains a large set
of reference molecules, covering all molecules appearing in PDB entries.

This process is performed for each target fragment.
The obtained hydrogen coordinates are assumed for the target molecule.
*(The hydrogen position of the previous figure is encircled again.)*

.. figure:: /images/position_adoption.png
   :width: 800

   *Adoption of obtained hydrogen coordinates.*


Hydrogen relaxation
-------------------

Energy function
^^^^^^^^^^^^^^^

Now the position of hydrogen atoms connected to terminal heavy atoms can
be further optimized, to reduce steric clashes and form hydrogen bonds for
example.
More precisely, the relaxation aims to minimize an energy function :math:`V`
based on non-bonded interaction terms from the *Universal Force Field* (UFF) [3]_.



, i.e. the energy minimized,
in order to reduce steric clashes and form hydrogen bonds for example.

*Hydride* uses an energy function :math:`V` based on the non-bonded
interaction terms of the *Universal Force Field* (UFF) [3]_.
The interaction terms comprise an electrostatic :math:`V_\text{el}` and a
*Lennart-Jones* :math:`V_\text{LJ}` term.
For the position vectors :math:`\vec{r}` of two atoms :math:`i` and :math:`j`
the contribution to the energy function is

.. math::

   V_(\vec{r}_i, \vec{r}_j) &= 
      V_\text{el}(\vec{r}_i, \vec{r}_j) + V_\text{LJ}(\vec{r}_i, \vec{r}_j) \\
      V_\text{el}(\vec{r}_i, \vec{r}_j) &= 332.067 \frac{q_i q_j}{D_{ij}} \\
      V_\text{LJ}(\vec{r}_i, \vec{r}_j) &= \epsilon_{ij} \left(
       \frac{\delta_{ij}^{12}}{D_{ij}^{12}} - 2\frac{\delta_{ij}^6}{D_{ij}^6}
   \right)

:math:`D_{ij}` is the euclidean distance between the atoms :math:`i`
and :math:`j`.

.. math::

   D_{ij} = | \vec{r}_j - \vec{r}_i |

:math:`\epsilon_{ij}` and :math:`\delta_{ij}` are the well depth and optimal
distance between these atoms, respectively, and are calculated as

.. math::

   \epsilon_{ij} &= \sqrt{ \epsilon_i  \epsilon_j}, \\
   \delta_{ij}   &= \frac{r_i + r_j}{2}.

:math:`\epsilon` and :math:`\delta` are taken from the UFF.
To obtain more accurate distances for hydrogen bonds, :math:`\delta` is
multiplied with :math:`0.79` for potential hydrogen bond acceptor-donor
pairs [4]_.
By default, the charges :math:`q` are calculated via the PEOE method [5]_
implemented in :func:`biotite.structure.partial_charges()`.

Interactions are calculated between all pairs of rotatable hydrogen atoms
and all other atoms within a defined cutoff distance of 10 Å.
All other interaction pairs do not need to be considered, as their distances
to each other are not altered during the course of relaxation.

**Units:**

   - Energies: *(kcal/mol)*
   - Lengths: *(Å)*
   - Charges: *(1)*

Relaxation algorithm
^^^^^^^^^^^^^^^^^^^^

Based on this energy function, the applicable hydrogen atoms are iteratively
rotated about the bond of the terminal heavy atom.
However, if the terminal heavy atom is bonded via a (partial) double bond to
the rest of the molecule, free rotation is prohibited.
For `imine <https://en.wikipedia.org/wiki/Imine>`_ groups, as they appear e.g.
in arginine, two hydrogen conformations are still possible though.
Due to these discrete values a continuous optimizer cannot be employed.
Hence, *Hydride* uses a variant of the *hill climbing* algorithm, that aims
to reach local minimum of the energy function :math:`V`.

Let :math:`\phi_1 ... \phi_n` be the dihedral angles of the rotatable terminal
bonds :math:`1 ... n`.
Each :math:`\phi_k` affects the positions :math:`\vec{r}_p ... \vec{r}_q` of
the hydrogen atoms bonded to the corresponding heavy atom.

In each iteration the dihedral angles of all rotatable bonds are altered by a
an angle increment :math:`\Delta \phi` in either direction.
:math:`\Delta \phi` is small (by default 10°) or 180° for freely rotatable
bonds and imine groups, respectively.
Let :math:`\phi_1^* ... \phi_n^*` be these updated angles.
Let :math:`\vec{r}_p^* ... \vec{r}_q^*` be the new positions resulting from
the new angle :math:`\phi_k^*`.

For each rotatable bond :math:`k`, the energy difference with respect to the
change in :math:`\phi_k` (:math:`\Delta V^*`) is calculated as

.. math::

   \Delta V^*(k) =
      \sum_{i=p}^q \sum_j^\text{all} V_(\vec{r}_i^*, \vec{r}_j) - V_(\vec{r}_i, \vec{r}_j)

Put into words, this means that all interaction terms are evaluated that
involve the atoms :math:`p ... q` affected by the rotatable bond :math:`k`.
For each interaction term, the energy difference between the positions
before and after the isolated update of :math:`\phi_k` is calculated.
:math:`\Delta V^*` is the sum of these energy differences.

If :math:`\Delta V^*(k)` is negative, the new dihedral angle for bond :math:`k`
is preferable, as it leads to a lower energy. 
Hence, :math:`\phi_k^*` is accepted and used as the new :math:`\phi_k` in the
next iteration.
Otherwise, it is rejected and the next iteration uses the :math:`\phi_k` from
the previous iteration.

When within an iteration no :math:`\phi_k^*` is accepted anymore for any
:math:`k`, the energy has reached the local minimum and the algorithm has
finished.

References
----------

.. [1] W Kabsch,
   "A solution for the best rotation to relate two sets of vectors."
   Acta Cryst, 32, 922-923 (1976).
   
.. [2] W Kabsch,
   "A discussion of the solution for the best rotation to relate
   two sets of vectors."
   Acta Cryst, 34, 827-828 (1978).

.. [3] AK Rappé, CJ Casewit, KS Colwell, WA Goddard III and WM Skiff,
   "UFF, a full periodic table force field for molecular mechanics
   and molecular dynamics simulations."
   J Am Chem Soc, 114, 10024-10035 (1992).

.. [4] T Ogawa and T Nakano,
   "The Extended Universal Force Field (XUFF): Theory and applications."
   CBIJ, 10, 111-133 (2010)

.. [5] J Gasteiger and M Marsili,
   "Iterative partial equalization of orbital electronegativity - a
   rapid access to atomic charges"
   Tetrahedron, 36, 3219 - 3288 (1980).