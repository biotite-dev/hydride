# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join, dirname, realpath
import numpy as np
import biotite.structure as struc


def data_dir():
    return join(dirname(realpath(__file__)), "data")


def place_over_periodic_boundary(atoms, dimension, box_size):
    """
    Place molecule at the border of a box with the given `box_size`,
    to 'cut' it in half due to PBC in the given dimension.
    In all other dimensions the molecule is placed in the center.
    """
    box = np.identity(3) * box_size
    min_coord = np.min(atoms.coord[:, dimension])
    max_coord = np.max(atoms.coord[:, dimension])
    atoms.coord[:, dimension] += \
        box_size - min_coord - (max_coord - min_coord) / 2
    # Keep the molecule in the center of the box in all other dimensions
    for d in [0, 1, 2]:
        if d == dimension:
            continue
        atoms.coord[:, d] -= np.min(atoms.coord[:, d])
        atoms.coord[:, d] += box_size / 2
    # Perform the 'cut'
    atoms.coord = struc.move_inside_box(atoms.coord, box)
    return atoms