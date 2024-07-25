from os.path import abspath, dirname, join
import ammolite
import biotite.structure as struc
import biotite.structure.io.mol as mol
import numpy as np
from util import COLORS, init_pymol_parameters

PNG_SIZE = (300, 300)
ZOOM = 3.5
MOL_DIR = dirname(abspath(__file__))


def load_and_orient(mol_name):
    molecule = mol.MOLFile.read(
        join(MOL_DIR, "molecules", f"{mol_name}.sdf")
    ).get_structure()
    molecule.coord -= struc.centroid(molecule)
    return molecule


benzene = load_and_orient("benzene")
butylene = load_and_orient("isobutylene")
toluene = load_and_orient("toluene")
benzene_heavy, butylene_heavy, toluene_heavy = [
    atoms[atoms.element != "H"] for atoms in (benzene, butylene, toluene)
]


init_pymol_parameters()

center = struc.array([struc.Atom([0, 0, 0], atom_name="C", element="C")])
center.bonds = struc.BondList(1)
CENTER = ammolite.PyMOLObject.from_structure(center, "center_")
ammolite.cmd.disable("center_")

pymol_toluene = ammolite.PyMOLObject.from_structure(toluene, "toluene")
CENTER.zoom(buffer=ZOOM)
pymol_toluene.color(COLORS["O"], toluene.element != "H")
ammolite.cmd.png("toluene.png", *PNG_SIZE)
ammolite.cmd.disable("toluene")

pymol_toluene_heavy = ammolite.PyMOLObject.from_structure(
    toluene_heavy, "toluene_heavy"
)
CENTER.zoom(buffer=ZOOM)
pymol_toluene_heavy.color(COLORS["O"])
ammolite.cmd.png("toluene_heavy.png", *PNG_SIZE)
ammolite.cmd.disable("toluene_heavy")

pymol_benzene = ammolite.PyMOLObject.from_structure(benzene, "benzene")
CENTER.zoom(buffer=ZOOM)
pymol_benzene.color(COLORS["N"], benzene.element != "H")
ammolite.cmd.png("benzene.png", *PNG_SIZE)
ammolite.cmd.disable("benzene")

pymol_butylene = ammolite.PyMOLObject.from_structure(butylene, "butylene")
CENTER.zoom(buffer=ZOOM)
pymol_butylene.color(COLORS["N"], butylene.element != "H")
ammolite.cmd.png("butylene.png", *PNG_SIZE)
ammolite.cmd.disable("butylene")


def visualize_fragments(molecule, mol_name, color):
    all_bonds, _ = molecule.bonds.get_all_bonds()
    frag_num = 0
    for i in range(molecule.array_length()):
        if molecule.element[i] != "H":
            bonded_i = all_bonds[i]
            bonded_i = bonded_i[bonded_i != -1]
            fragment = molecule[np.append(bonded_i, [i])]
            fragment_name = f"{mol_name}_frag_{frag_num:02d}"
            frag_num += 1

            pymol_fragment = ammolite.PyMOLObject.from_structure(
                fragment, fragment_name
            )
            CENTER.zoom(buffer=ZOOM)
            pymol_fragment.color(COLORS["C"], fragment.element != "H")
            pymol_fragment.color(color, -1)
            ammolite.cmd.png(f"{fragment_name}.png", *PNG_SIZE)
            ammolite.cmd.disable(fragment_name)


visualize_fragments(toluene_heavy, "toluene", COLORS["O"])
visualize_fragments(benzene, "benzene", COLORS["N"])
visualize_fragments(butylene, "butylene", COLORS["N"])
