# Circuit-Neuromodulation
Neuromodulation of Neuromorphic Circuits (Ribar and Sepulchre, 2019): https://ieeexplore.ieee.org/document/8698454
(arXiv version: https://arxiv.org/abs/1805.05696)

## Overview
The repository provides a graphical interface to control the behavior of the neural circuit by shaping its I-V curves, as detailed in the paper. It is organized as follows:
- `gui.py`: Main file that generates the interface for changing the parameters of the circuit and a live plot of the its behavior.
- `model.py`: Definition of the neuron model.
- `requirements.txt`: Python dependencies.

## Binder version
The interface can be used online without a Python distribution installed through Binder:

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/lukaribar/circuit-neuromodulation/master?filepath=gui_old_notebook.ipynb)

**NOTE**: This uses an old version of the interface, without live plotting and additional utilities. For most up to date version, use **`gui.py`**.

## Reference
If you use the model, please cite the paper:

```
@ARTICLE{ribar2019neuromodulation,
author={L. {Ribar} and R. {Sepulchre}},
journal={IEEE Transactions on Circuits and Systems I: Regular Papers},
title={Neuromodulation of Neuromorphic Circuits},
year={2019},
volume={66},
number={8},
pages={3028-3040},
doi={10.1109/TCSI.2019.2907113},
ISSN={1549-8328},
month={Aug},}
```

## License
MIT
