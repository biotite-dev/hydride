# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import biotite.structure.io.pdbx as pdbx
import pytest
from tests.util import data_dir


@pytest.fixture
def atoms():
    """
    AtomArray for first model of ``1L2Y``.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir(), "1l2y.bcif"))
    return pdbx.get_structure(
        pdbx_file, model=1, include_bonds=True, extra_fields=["charge"]
    )
