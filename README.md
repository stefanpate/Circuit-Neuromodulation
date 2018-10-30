# Circuit-Neuromodulation
Neuromodulation of Neuromorphic Circuits (Ribar and Sepulchre, 2018): https://arxiv.org/abs/1805.05696

## Overview
The repository provides a graphical interface to control the behavior of the neural circuit by shaping its I-V curves, as detailed in the paper. It is organized as follows:
- `gui.py`: Main file that generates the interface for changing the parameters of the circuit and a live plot of the its behavior.
- `model.py`: Definition of the neuron model, as described in the paper.
- `requirements.txt`: Python dependencies.

## Binder version
The interface can be used online without a Python distribution installed through Binder:

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/lukaribar/Circuit-Neuromodulation/master?filepath=gui_old_notebook.ipynb)

**NOTE**: This uses an old version of the interface, without live plotting and additional utilities. For most up to date version, use **`gui.py`**.

## Reference
If you use the model, please cite the paper:

```
@ARTICLE{ribar2018neuromodulation,
   author = {{Ribar}, L. and {Sepulchre}, R.},
   title = "{Neuromodulation of Neuromorphic Circuits}",
   journal = {ArXiv e-prints},
   archivePrefix = "arXiv",
   eprint = {1805.05696},
   primaryClass = "q-bio.NC",
   keywords = {Quantitative Biology - Neurons and Cognition},
   year = 2018,
   month = may,
   adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180505696R},
   adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
## License
MIT
