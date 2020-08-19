# Circuit-Neuromodulation
Neuromodulation of Neuromorphic Circuits (Ribar and Sepulchre, 2019): https://ieeexplore.ieee.org/document/8698454
(arXiv version: https://arxiv.org/abs/1805.05696)

## Overview
The repository provides a graphical interface to control the behavior of the neural circuit by shaping its I-V curves, as detailed in the paper. It is organized as follows:
- `gui.py`: Main file that generates the interface for changing the parameters of the circuit and a live plot of the its behavior.
- `neuron_model.py`: Definition of the neuron model.
- `requirements.txt`: Python dependencies.

The repository also provides definitions for simulating network behavior:
- `network_model.py`: Definition synaptic and resistive interconnections, uses `neuron_model.py` for the model of the nodes.
- `hco_gui.py`: Graphical interface for simulating a 2-neuron network with excitatory or inhibitory interconnections.

Models `neuron_model.py` and `network_model.py` provide definitions for conductance-based elements as well. An example of a conductance-based model is shown in:
- `gui_conductance.py`: Graphical interface for controlling a conductance-based equivalent of the current-based model in `gui.py`.

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
