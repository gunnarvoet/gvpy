# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

gvpy is a personal utility library for oceanographic data analysis, imported everywhere as
`import gvpy as gv`. It is installed editable into analysis projects, so a change here takes effect
immediately in every project and notebook that uses it. Function signatures and behavior are a
public interface for those projects even though the README promises no stability.

## Commands

All commands go through uv.

```sh
make test                 # uv run pytest
make check                # uv run ruff check src/gvpy/ tests/
make format               # uv run ruff format src/gvpy/ tests/
make format-check         # ruff format --diff, no changes written
make docs                 # build pdoc site into docs/ and open it
make servedocs            # pdoc live server

uv run pytest tests/test_signal.py                      # one file
uv run pytest tests/test_signal.py::test_name           # one test
uv run pytest -k "quickfig"                             # by keyword
```

- `make check` and `make format-check` both pass clean. Keep them that way.
- pytest runs with `filterwarnings = ["error"]` and `xfail_strict = true`. Any warning raised during
  a test, including a deprecation warning from numpy, xarray, or matplotlib, fails the test.
- Ruff lint rules are pinned to `E`, `F`, `I` in `pyproject.toml`. `dict(key=value)` is the house
  style (C408 is ignored on purpose). Do not convert these calls to literals.

## Architecture

Src-layout package in `src/gvpy/`, one flat module per topic, built with `uv_build`.

- `__init__.py` imports every public module eagerly and lists it in `__all__`. A new top-level
  module must be added to both the `__all__` list and the `from . import ...` line.
- Modules reach each other through the top-level package: `gm81`, `xr`, `mp`, `mod`, `trilaterate`,
  and `io` do `import gvpy as gv` and call `gv.ocean.*`, `gv.plot.*`, and so on. This is a circular
  import that works only because `gv.<module>` is resolved inside function bodies at call time.
  Using `gv.<module>.<name>` at module level (a default argument, a decorator, a module constant)
  breaks `import gvpy`.
- Importing gvpy has side effects. `xr.py` registers the xarray accessors `.gv` (DataArray and
  Dataset) and `.gadcp` (Dataset). `plot.py` imports `cm`, which registers the custom colormaps with
  matplotlib. `cm` is intentionally absent from `__all__`. `__init__.py` falls back to the Agg
  backend when `matplotlib.pyplot` cannot be found.
- `xr.py` is the convenience layer over the rest: accessor methods such as `da.gv.plot()`,
  `da.gv.tplot()`, and `da.gv.plot_spectrum()` dispatch on coordinates and delegate to `plot`,
  `signal`, and `gm81`.
- `mixsea` is installed from the `modscripps/mixsea` git `main` branch through `[tool.uv.sources]`.
- `mod` (epsi, FCTD) is superseded by the separate `modfish` package. Old notebooks reference
  `gv.figure` (now `plot`) and `gv.adcp` (moved out of the package). Both are gone.
- Several `ocean` functions need external data or network access: Smith & Sandwell bathymetry from
  a local file, WOCE and WAGHC climatologies over OPeNDAP, and the marineregions.org gazetteer.
  Tests must not depend on them.

## Conventions

- numpydoc docstrings. Docs are built with `pdoc -d numpy --math`, so LaTeX in docstrings renders.
- Add a plot variant as a new function. Do not change an existing plotting function's behavior or
  hide a variant behind a new parameter.
- `gv.misc.log()` returns a loguru logger. Use f-strings. Printf-style positional args print
  literally.
- Test files set `matplotlib.use("Agg")` before importing pyplot and close figures after each test
  (see the autouse fixture in `tests/test_plot.py`). No `__init__.py` in `tests/`.

## Branches, releases, docs

- Work lands on `dev`. `dev` is occasionally merged into `main` and tagged `vYYYY.MM`.
- The version is `YYYY.MM`, bumped by hand in `pyproject.toml`. `__version__` reads it from the
  installed package metadata.
- A push to `main` triggers `.github/workflows/docs.yml`, which installs from `uv.lock`
  (`uv sync --locked`), runs `make ghdocs`, and deploys to GitHub Pages. A dependency change
  without an updated `uv.lock` fails that build.
- License is GPL-3.0-or-later. `gm81.py` is third-party GPL code (Joern Callies' GM81) and keeps
  its own header. The pdoc theme in `.pdoc-theme-gv/` is a git submodule.
- `docs/` and `notebooks/` are gitignored. `docs/` is build output and `make docs` deletes it first.
