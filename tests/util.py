# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join, dirname, realpath


def data_dir():
    return join(dirname(realpath(__file__)), "data")