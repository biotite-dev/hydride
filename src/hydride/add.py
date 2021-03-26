# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "hydride"
__author__ = "Patrick Kunzmann, Jacob Marcel Anter"
__all__ = ["add_hydrogen"]

import numpy as np
import biotite.structure as struc
from .library import FragmentLibrary


def add_hydrogen(atoms, fragment_library):
   if (atoms.element == "H").any():
      raise struc.BadStructureError(
         "Input structure already contains hydrogen atoms"
      )
   
   hydrogen_coord = fragment_library.calculate_hydrogen_coord(atoms)
   print("\n"*5)
   for e in hydrogen_coord:
      print(e)
      print()
   flattened_hydrogen_coord = []
   for coord in hydrogen_coord:
      flattened_hydrogen_coord += list(coord)
   hydrogen_atoms = struc.AtomArray(len(flattened_hydrogen_coord))
   hydrogen_atoms.coord = np.array(flattened_hydrogen_coord)

   # Set annotation arrays
   hydrogen_atoms.element[:] = "H"
   hydrogen_atoms.atom_name[:] = "H"
   
   hydrogenated_atoms = atoms + hydrogen_atoms

   # Actual covalent bonds of heavy atom to hydrogen
   hydrogen_bonds = []
   j = atoms.array_length()
   for i in range(len(hydrogen_coord)):
      for _ in range(len(hydrogen_coord[i])):
         hydrogen_bonds.append((i, j, struc.BondType.SINGLE))
         j += 1
   hydrogen_bonds = struc.BondList(
      hydrogenated_atoms.array_length(),
      np.array(hydrogen_bonds, dtype=np.uint32)
   )
   hydrogenated_atoms.bonds = hydrogenated_atoms.bonds.merge(hydrogen_bonds)

   return hydrogenated_atoms