PYTEST_ARGS?=
SHELL = /bin/sh

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
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

BROWSER := python3 -c "$$BROWSER_PYSCRIPT"

PYTHON_FILES = $(shell git ls-files | grep -e 'py$$' | grep -v -e '__init__.py')
PYTHON_INIT_FILES = $(shell git ls-files | grep '__init__.py')

help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

coverage: ## check code coverage quickly with the default Python
	pytest --cov=fv3gfs --cov-report=html
	$(BROWSER) htmlcov/index.html

test: ## run tests quickly with the default Python
	pytest $(PYTEST_ARGS) tests

test_mpi:
	mpirun -n 6 --allow-run-as-root --mca btl_vader_single_copy_mechanism none --oversubscribe pytest $(PYTEST_ARGS) tests/mpi
	$(MAKE) -C examples/mpi

lint:
	black --diff --check $(PYTHON_FILES) $(PYTHON_INIT_FILES)
	flake8 $(PYTHON_FILES)
	# ignore unused import error in __init__.py files
	flake8 --ignore=F401 $(PYTHON_INIT_FILES)
	mypy --follow-imports silent --show-error-codes fv3gfs
	@echo "LINTING SUCCESSFUL"

# We do not use --no-annotate to keep the dependencies annotations
# but we beautify the files with `sed`
#  - line 1: remove direct git dependencies
#  - line 2-5: beautify the comments (inline comments generated)
#  - line 5: get rid of an illegaly generated [toml]
#  - line 6: replace the pip update command with the make call in docs
constraints.txt: requirements.txt requirements_gpu.txt setup.py
	pip-compile $^ --output-file constraints.txt
	sed -i '/^git+https/d' constraints.txt
	sed -i 's/ *# \(.*\)/# \1/g' constraints.txt
	sed -i -z 's/\n#/  #/6g' constraints.txt
	sed -i 's/\n,/,/g' constraints.txt
	sed -i 's/via,/via/g' constraints.txt
	sed -i '/^\[toml\]/d' constraints.txt
	sed -i '5s/.*/#    make constraints.txt/' constraints.txt

clean:
	$(MAKE) -c examples/mpi clean

reformat:
	black $(PYTHON_FILES) $(PYTHON_INIT_FILES)

docs: ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

.PHONY: docs clean test lint reformat test_mpi coverage help
