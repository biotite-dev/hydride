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
bond lengths and angles, but some steric clashes might appear.


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

.. figure:: /images/library.png
   :width: 800

   *Library creation with fragments from benzene and isobutylene.*

In the figure shown above the library contains only fragments from benzene and
isobutylene.
However, *Hydride*'s default fragment library contains fragments from
all compounds from the
`Chemical Component Dictionary <https://www.wwpdb.org/data/ccd>`_.

The target molecule, in the shown example we use toluene (*red*), is split into
fragments in a similar fashion.
But in contrast to the molecules for the fragment library the created
fragments of course miss hydrogen atoms.

.. figure:: /images/target_fragments.png
   :width: 800

   *Fragmentation of the target molecule toluene.*

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

.. figure:: /images/superimposition.png
   :width: 800

   *Superimposition of a library fragment onto a target fragment.*

If the library does not contain a match for a target molecule fragment, the
algorithm is unable to assign hydrogen atoms to this central atom.
Hence, it is desirable to have a large fragment library to cover a broad range
of different fragments.

After this procedure is finished for each target fragment, the obtained
hydrogen positions are adopted by the target molecule.
*(The hydrogen position of the previous figure is encircled again.)*

.. figure:: /images/position_adoption.png
   :width: 800

   *Adoption of obtained hydrogen coordinates.*


Hydrogen relaxation
-------------------

Energy function
^^^^^^^^^^^^^^^

After initial hydrogen atom placement the position of hydrogen connected to
terminal heavy atoms can be further optimized, i.e. the energy minimized,
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