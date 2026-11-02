PY ?= python3

.PHONY: smoke verify experiments analysis extensions figures all

smoke:
	$(PY) code/exp3.py CHECK
	$(PY) code/verify.py

verify:
	$(PY) code/verify.py

experiments:
	$(PY) code/experiments.py X1 X1b X2 X3 X4 X5 X6 X7 X9 X10

analysis:
	$(PY) code/analysis.py

extensions:
	$(PY) code/exp3.py CHECK X11 X12 X13 X14
	$(PY) code/analysis_ext.py

figures:
	bash diagrams/render.sh

all: verify experiments analysis extensions
