import time
from os.path import join, abspath, dirname
import numpy as np
from sklearn.decomposition import PCA
import biotite.structure as struc
import biotite.structure.io.mol as mol
import ammolite
from util import COLORS, init_pymol_parameters


PNG_SIZE = (300, 300)
ZOOM = 1.0
MOL_DIR = dirname(abspath(__file__))


def load_and_orient(mol_name):
    molecule = mol.MOLFile.read(
        join(MOL_DIR, "molecules", f"{mol_name}.sdf")
    ).get_structure()
    molecule.coord -= struc.centroid(molecule)
    return molecule

propane  = load_and_orient("propane")


ammolite.launch_interactive_pymol()

init_pymol_parameters()

center = struc.array([struc.Atom([0,0,0], atom_name="C", element="C")])
center.bonds = struc.BondList(1)
CENTER = ammolite.PyMOLObject.from_structure(center, "center_")
ammolite.cmd.disable("center_")

all_bonds, _ = propane.bonds.get_all_bonds()
for carbon_i, name in zip((0, 1), ("group_constrained", "group_free")):
    bonded_i = all_bonds[carbon_i]
    bonded_i = bonded_i[bonded_i != -1]
    fragment = propane[np.append(bonded_i, [carbon_i])]
    
    pymol_fragment = ammolite.PyMOLObject.from_structure(
        fragment, name, delete=False
    )
    CENTER.zoom(buffer=ZOOM)
    pymol_fragment.color(COLORS["C"], fragment.element != "H")
    pymol_fragment.color(COLORS["N"], -1)
    #ammolite.cmd.png(f"{name}.png", *PNG_SIZE)
    #ammolite.cmd.disable(name)