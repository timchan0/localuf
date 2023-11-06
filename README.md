# localuf (Local Union-Find)

A Python package to simulate and visualise Union-Find decoders for CSS codes.

## Package Requirements

From conda:
- python
- matplotlib
- networkx
- pytest
- ipykernel
- scipy
- pandas
- snakeviz
- statsmodels

From pip:
- pymatching

## Local Installation Instructions
Create environment called `localuf`:

`conda create -n localuf python matplotlib networkx pytest scipy pandas snakeviz statsmodels ipykernel ipywidgets`

Activate the environment:

`conda activate localuf`

Install `pymatching` (see https://pymatching.readthedocs.io/en/latest/#installation):

`pip install pymatching --upgrade`

Install `localuf` Python package locally (using `setup.cfg` and `setup.py`):

`python -m pip install --no-build-isolation --no-deps -e .`

## Visualising Macar

First, sample an error using physical error probability $p =5 \times 10^{-2}$
and visualise it on the circuit-level decoding graph:

```python
import localuf

sf = localuf.Surface(3, 'circuit-level')
error = sf.make_error(0.05)
sf.draw(error, with_labels=False)
```

Drawing key:
* bitflipped edges are thick red; else, thin black
* boundary nodes are blue; defects red; else, green

Next, compute the syndrome, feed it into Macar, and visualise the decoding:

```python
luf = localuf.decoders.LUF(sf)
syndrome = sf.get_syndrome(error)
luf.decode(syndrome, draw=True)
```

This should make an interactive widget in which one can click through each timestep.
If this does not work, add `interactive=False` keyword argument to output a static image.
Drawing key:
* ungrown edges are invisible
* half-grown edges are dotted
* fully grown edges are solid
* active nodes are square-shaped
* inactive nodes are circular
* CID is shown as a label
* nodes with anyons are outlined in black
* pointers are shown by arrows on edges
* edges so far added to the correction are in red
* the top-left node also shows the controller stage (PS stands for presyncing etc.)

## Making Threshold Data

Simulate $10^4$ decoding cycles of UF on the surface code under code capacity,
for $(d, p) \in \{3, 5, 7\} \times \{0.06, 0.08, 0.1, 0.12\}$:

```python
import numpy as np

fUF = localuf.sim.make_threshold_data(
    ds=range(3, 9, 2),
    ps=np.linspace(0.06, 0.12, 4),
    ns=int(1e4),
    code_class=localuf.Surface,
    decoder_class=localuf.decoders.UF,
    error_model='code capacity',
)
```

Plot the logical success/failure data:

```python
localuf.plot.threshold_data(fUF)
```

One should see a threshold around 0.09.

## Making Runtime Data

Simulate $10^2$ syndrome validations of Macar on the surface code under circuit-level noise,
for $(d, p) \in \{3, 5, 7, 9\} \times \{10^{-3}, 5\times 10^{-3}, 10^{-2}\}$,
then plot the data:

```python
tMacar = localuf.sim.make_runtime_data(
    ds=range(3, 11, 2),
    ps=[1e-3, 5e-3, 1e-2],
    n=int(1e2),
    error_model='circuit-level',
    validate_only=True,
)
localuf.plot.mean_runtime(tMacar)
```

To do the same for Actis, set `visible=False`.