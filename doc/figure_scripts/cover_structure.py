import numpy as np
import pyximport
pyximport.install(
    build_in_temp=False,
    setup_args={"include_dirs":np.get_include()},
    language_level=3
)

from os.path import join
import numpy as np
import biotite.structure as struc
import biotite.structure.io.mmtf as mmtf
import biotite.database.rcsb as rcsb
import hydride
import ammolite
from util import COLORS, init_pymol_parameters


mmtf_file = mmtf.MMTFFile.read(rcsb.fetch("1bna", "mmtf"))
heavy_atoms = mmtf.get_structure(
    mmtf_file, model=1, include_bonds=True, extra_fields=["charge"]
)
heavy_atoms = heavy_atoms[heavy_atoms.res_name != "HOH"]

all_atoms, _ = hydride.add_hydrogen(heavy_atoms)
all_atoms.coord = hydride.relax_hydrogen(all_atoms)

# Select a single CG base pair
heavy_atoms = heavy_atoms[np.isin(heavy_atoms.res_id, (3, 22))]
all_atoms = all_atoms[np.isin(all_atoms.res_id, (3, 22))]

bonds = struc.hbond(all_atoms)


init_pymol_parameters()
ammolite.cmd.set("valence", 0)

pymol_heavy = ammolite.PyMOLObject.from_structure(heavy_atoms)
pymol_heavy.show_as("sticks")
pymol_all = ammolite.PyMOLObject.from_structure(all_atoms)
pymol_all.show_as("sticks")
pymol_all.orient()
pymol_all.zoom(buffer=1.0)
ammolite.cmd.rotate("z", 90)

for pymol_object, atoms in zip(
    (pymol_heavy, pymol_all), (heavy_atoms, all_atoms)
):
    for element in ("H", "C", "N", "O", "P"):
        pymol_object.color(COLORS[element], atoms.element == element)

pymol_all.disable()
pymol_heavy.enable()
ammolite.cmd.png("cover_heavy.png", 400, 800)

pymol_heavy.disable()
pymol_all.enable()
for i, (_, atom_i, atom_j) in enumerate(bonds):
    color = COLORS["N"] if all_atoms.element[atom_j] == "N"\
            else COLORS["O"]
    pymol_all.distance(str(i), atom_i, atom_j, show_label=False)
    ammolite.cmd.set_color(f"bond_color_{i}", list(color))
    ammolite.cmd.color(f"bond_color_{i}", str(i))
ammolite.cmd.png("cover_hydrogenated.png", 400, 800)