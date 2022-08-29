.. This source code is part of the Hydride package and is distributed
   under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
   information.

Hydride - Adding hydrogen atoms to molecular models
===================================================

Many tasks in structural biology ranging from simulations and hydrogen
bond detection to mere visual analysis, require complete molecular
models.
However, most experimentally determined structures do not include
the position of hydrogen atoms, due to their small size and electron
density.

*Hydride* is an easy-to-use program and library written in Python that
adds missing hydrogen atoms to molecular models based on known bond
lengths and angles.
Since it does not require force-field parameters for the specific
molecule(s), it can be used for adding hydrogen atoms to almost any
organic molecule - from small ligands to large protein complexes.


If you use *Hydride* in a scientific publication, please cite:

| Kunzmann, P. Anter, J. M. & Hamacher, K. *Algorithms Mol. Biol.* (2022) 17:7.
| `<https://doi.org/10.1186/s13015-022-00215-x>`_


.. image:: /images/cover_structure.png
   :width: 400
   :align: center

.. toctree::
   :maxdepth: 1
   :hidden:
   
   intro
   cli
   api
   issues
