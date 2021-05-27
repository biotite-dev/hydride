"""
The code from the snippets in ``api.py``.
Used to test the correctness of these snippets and generate output text
and images.
"""

import tempfile
import biotite.structure as struc
import biotite.structure.io.mol as mol
import biotite.structure.info as info
import ammolite
from util import COLORS, init_pymol_parameters


ZOOM = 1.5
PNG_SIZE = (400, 400)


def color_atoms(pymol_object, atom_array):
    for element in ("H", "C", "N", "O", "P"):
        pymol_object.color(COLORS[element], atom_array.element == element)


# 2-nitrophenol
molecule = info.residue("OPO")
# Swap positions of oxygen atoms in nitro group
# for a better visible hydrogen bond
molecule_copy = molecule.copy()
molecule_copy.coord[0] = molecule.coord[2]
molecule_copy.coord[2] = molecule.coord[0]
molecule = molecule_copy
# Remove hydrogen atoms
molecule = molecule[molecule.element != "H"]
# Align the *molecule main axis* to x-axis
molecule = struc.align_vectors(
    molecule, molecule.coord[3] - molecule.coord[1], [-1, 0, 0]
)
# Write to temporary file
mol_file = mol.MOLFile()
mol_file.set_structure(molecule)
temp = tempfile.NamedTemporaryFile("w", suffix=".mol")
mol_file.write(temp)
temp.flush()

init_pymol_parameters()


########################################################################
#
#

import biotite.structure.io.mol as mol

mol_file = mol.MOLFile.read(temp.name)
molecule = mol_file.get_structure()
print(type(molecule))
print()
print(molecule)

#
#
########################################################################

print("\nEND OF SNIPPET\n")
pymol_heavy = ammolite.PyMOLObject.from_structure(molecule, "heavy")
#pymol_heavy.orient()
pymol_heavy.zoom(buffer=ZOOM)
color_atoms(pymol_heavy, molecule)
ammolite.cmd.png("api_01.png", *PNG_SIZE)
ammolite.cmd.disable("heavy")


########################################################################
#
#

import hydride

# Remove already present hydrogen atoms (only necessary in rare cases)
molecule = molecule[molecule.element != "H"]
# Add hydrogen atoms
molecule_with_h, mask = hydride.add_hydrogen(molecule)
print(molecule_with_h)
print()
print(mask)

#
#
########################################################################


print("\nEND OF SNIPPET\n", end="")
pymol_hydrogen = ammolite.PyMOLObject.from_structure(molecule_with_h, "hydrogen")
pymol_heavy.zoom(buffer=ZOOM)
color_atoms(pymol_hydrogen, molecule_with_h)
ammolite.cmd.png("api_02.png", *PNG_SIZE)
ammolite.cmd.disable("hydrogen")


########################################################################
#
#

print(molecule_with_h[mask])

#
#
########################################################################


print("\nEND OF SNIPPET\n", end="")


########################################################################
#
#

molecule_with_h.coord = hydride.relax_hydrogen(molecule_with_h)

#
#
########################################################################


print("\nEND OF SNIPPET\n", end="")
pymol_relaxed = ammolite.PyMOLObject.from_structure(molecule_with_h, "relaxed")
pymol_heavy.zoom(buffer=ZOOM)
color_atoms(pymol_relaxed, molecule_with_h)
_, atom_i, atom_j = struc.hbond(molecule_with_h, cutoff_angle=105)[0]
pymol_relaxed.distance("hbond", atom_i, atom_j, show_label=False)
ammolite.cmd.set_color("hbond_color", list(COLORS["O"]))
ammolite.cmd.color("hbond_color", "hbond")
ammolite.cmd.png("api_03.png", *PNG_SIZE)
ammolite.cmd.disable("relaxed")

template_molecule = molecule_with_h


########################################################################
#
#

import copy

library = copy.deepcopy(hydride.FragmentLibrary.standard_library())
library.add_molecule(template_molecule)
hydride.add_hydrogen(molecule, fragment_library=library)

#
#
########################################################################


print("\nEND OF SNIPPET\n", end="")


########################################################################
#
#

library = copy.deepcopy(hydride.AtomNameLibrary.standard_library())
library.add_molecule(template_molecule)
hydride.add_hydrogen(molecule, name_library=library)

#
#
########################################################################


print("\nEND OF SNIPPET\n", end="")
import matplotlib


########################################################################
#
#

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

#
#
########################################################################


print("\nEND OF SNIPPET\n", end="")
fig.savefig("api_04.png")

import biotite.structure.info as info
molecule = info.residue("ASP")


########################################################################
#
#

charges = hydride.estimate_amino_acid_charges(molecule, ph=7.0)
molecule.set_annotation("charge", charges)

#
#
########################################################################


print("\nEND OF SNIPPET\n", end="")


########################################################################
#
#

import biotite.structure as struc

molecule.bonds = struc.connect_via_residue_names(molecule)

#
#
########################################################################