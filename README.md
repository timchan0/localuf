# localuf (Local Union-Find)

A Python package to simulate and visualise
Union-Find-related decoders for CSS codes.
All UML class diagrams can be found in `uml_class_diagrams.md`.

## Local Installation Instructions

### Using Conda

Create environment called `localuf`:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate localuf
```

Install the `localuf` package in editable mode (and its dependencies from pip):

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

### Using Pip (Untested)

Install the package in editable mode and its dependencies using pip:

```bash
pip install -e .
```

## Usage

See `demo_notebooks/intro.ipynb`.
The first section is a demo of the Macar and Actis decoders.
The second section is a demo of the Snowflake decoder.