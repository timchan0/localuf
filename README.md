# localuf (Local Union-Find)

[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://timchan0.github.io/localuf/)

A Python package to simulate and visualise
Union-Find-related decoders for the surface and repetition codes.
Used in the papers on [Snowflake](https://arxiv.org/abs/2406.01701)
and [Macar/Actis](https://quantum-journal.org/papers/q-2023-11-14-1183/).

## Installation Instructions
localuf is available as a [PyPI package](https://pypi.org/project/localuf/)
so it can be installed by running `pip install localuf`.

## Tutorial
See the [`demo_notebooks/intro.ipynb`](https://github.com/timchan0/localuf/blob/main/demo_notebooks/intro.ipynb)
notebook.
The first section is a demo of the Macar and Actis decoders.
The second section is a demo of the Snowflake decoder.

## Documentation
If you just want to read the documentation,
click the blue **Documentation** badge at the top of this README.

Optionally, you can build the HTML docs locally.
First ensure you are in a Python environment (conda env, venv, etc.) where Sphinx is installed:

- One-liner: `sphinx-build -b html docs docs/_build/html`
- Or via Makefile: `make -C docs html`

## Code Structure
All UML class diagrams can be found in [`uml_class_diagrams.md`](https://github.com/timchan0/localuf/blob/main/uml_class_diagrams.md).