# Copyright 2020 Gunnar Voet
#
# This file is part of gvpy.
#
# gvpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# gvpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with gvpy.  If not, see <http://www.gnu.org/licenses/>.


"""gvpy is a Python package with various little bits and pieces of code that are mostly useful for oceanographic data analysis.

## License

Copyright 2020 Gunnar Voet

gvpy is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

gvpy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with gvpy.  If not, see <http://www.gnu.org/licenses/>.

## Installation

Clone the package from https://github.com/gunnarvoet/gvpy. Then install `gvpy`
by changing into the root directory and running

>>> python setup.py install

or using [pip](https://pypi.org/project/pip/)

>>> pip install .

To install in developer mode run either

>>> python setup.py develop

or

>>> pip install -e .

Now in python you should be able to run

```python
import gvpy
```

which will provide several sub-modules as documented here.
"""

__all__ = ["plot", "ocean", "misc", "signal", "io", "maps", "time"]

__author__ = "Gunnar Voet"
__email__ = "gvoet@ucsd.edu"
__version__ = "0.2.0"

# workaround for when whatever is defined as the default backend is not around:
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib as mpl

    mpl.use("Agg")

from . import io, misc, ocean, plot, signal, maps, time
