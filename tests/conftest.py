# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest
from tests.util import data_dir


def pytest_sessionstart(session):
    """
    Compile Cython source files, if Cython is installed and files are
    not compiled, yet.
    """
    try:
        import pyximport

        pyximport.install(
            build_in_temp=False,
            setup_args={"include_dirs": np.get_include()},
            language_level=3,
        )
    except ImportError:
        pass


@pytest.fixture
def atoms():
    """
    AtomArray for first model of ``1L2Y``.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir(), "1l2y.bcif"))
    return pdbx.get_structure(
        pdbx_file, model=1, include_bonds=True, extra_fields=["charge"]
    )
