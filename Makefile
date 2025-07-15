.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

lint: ## check style
	uvx ruff check gvpy

check: ## check style
	uvx ruff check gvpy

format: ## format code using ruff
	uvx ruff format gvpy

docs: ## generate documentation using pdoc
	rm -rf docs
	uvx --with . pdoc -d numpy -o docs -t .pdoc-theme-gv --math ./gvpy
	$(BROWSER) docs/index.html

ghdocs: ## generate documentation using pdoc
	rm -rf docs
	PDOC_ALLOW_EXEC=1 pdoc -d numpy -o docs -t .pdoc-theme-gv --math ./gvpy

# if there are any issues with importing certain modules, set environment
# variable PDOC_ALLOW_EXEC
# docs: ## generate documentation using pdoc
# 	rm -rf docs
# 	PDOC_ALLOW_EXEC=1 uvx --with . pdoc -d numpy -o docs -t .pdoc-theme-gv --math ./gvpy
# 	$(BROWSER) docs/index.html

servedocs: ## compile the docs & watch for changes
	uvx --with . pdoc -d numpy -t .pdoc-theme-gv --math ./gvpy
	# $(BROWSER) http://localhost:8080

test: ## run tests quickly with the default Python
	uvx --with . pytest
