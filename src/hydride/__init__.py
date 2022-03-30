# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__version__ = "1.1.1"
__name__ = "hydride"
__author__ = "Jacob Anter, Patrick Kunzmann"

from .add import *
from .charge import *
from .fragments import *
from .names import *

# Module can only be imported if the C-extension has already been built
try:
    from .relax import *
except ImportError:
    pass