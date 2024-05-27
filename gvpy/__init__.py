"""Python package with various little bits and pieces of code that are mostly
useful for oceanographic data analysis.  Use with caution and at your own risk.
Please note that the code changes often and may do so without any warning or
prior notice.  Consider copying snippets you find helpful to your own library.

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

Copyright 2024 Gunnar Voet

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

import importlib.metadata
import importlib.util

__all__ = [
    "io",
    "ocean",
    "plot",
    "time",
    "signal",
    "maps",
    "mp",
    "trilaterate",
    "xr",
    "misc",
    "gm81",
]

__author__ = "Gunnar Voet"
__email__ = "gvoet@ucsd.edu"
# version is defined in pyproject.toml
__version__ = importlib.metadata.version("gvpy")

# workaround for when whatever is defined as the default backend is not around:
if importlib.util.find_spec("matplotlib.pyplot") is None:
    import matplotlib as mpl

    mpl.use("Agg")
else:
    pass

from . import io, misc, ocean, plot, signal, maps, time, mp, trilaterate, xr, gm81
