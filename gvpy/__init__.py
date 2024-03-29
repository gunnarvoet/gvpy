"""Python package with various little bits and pieces of code that are mostly
useful for oceanographic data analysis. Please note that the code changes often
and without any warning or notice.

## Installation

Clone the package from
[https://github.com/gunnarvoet/gvpy](https://github.com/gunnarvoet/gvpy). For a regular
installation, change into the package root directory and run

```sh
python setup.py install
```

or using [pip](https://pypi.org/project/pip/)

```sh
pip install .
```

To install the package in developer mode (needed to make code changes available on the fly) run either

```sh
python setup.py develop
```

or

```sh
pip install -e .
```

Now, in Python you should be able to run

```python
import gvpy
```

which will provide several sub-modules, for example `gvpy.ocean`, as documented here.

## License

Copyright 2023 Gunnar Voet

gvpy is free software: you can redistribute it and/or modify it under the terms
of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3 of the License, or any later version.

gvpy is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License along
with gvpy.  If not, see <http://www.gnu.org/licenses/>.
"""

__all__ = ["io", "ocean", "plot", "time", "signal", "maps", "mp", "trilaterate", "xr", "misc", "gm81"]

__author__ = "Gunnar Voet"
__email__ = "gvoet@ucsd.edu"
__version__ = "0.2.0"

# workaround for when whatever is defined as the default backend is not around:
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib as mpl

    mpl.use("Agg")

from . import io, misc, ocean, plot, signal, maps, time, mp, trilaterate, xr, gm81
