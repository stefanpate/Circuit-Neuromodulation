# Circuit-Neuromodulation
Contact: Luka Ribar (lukaribar.mg@gmail.com)

Python implementation of the neuromorphic neuronal model described in:

- [Neuromodulation of Neuromorphic Circuits](https://arxiv.org/abs/1805.05696) (Ribar and Sepulchre, IEEE Transactions on Circuits and Systems, 2019)
- [Neuromorphic Control](https://arxiv.org/abs/2011.04441) (Ribar and Sepulchre, 2020)

## Overview
The repository provides a graphical interface to control the behavior of the neural circuit by shaping its I-V curves, as detailed in the paper. It is organized as follows:
- `gui.py`: Main file that generates the interface for changing the parameters of the circuit and a live plot of the its behavior.
- `neuron_model.py`: Definition of the neuron model.
- `requirements.txt`: Python dependencies.

The repository also provides definitions for simulating network behavior:
- `network_model.py`: Definitions of synaptic and resistive interconnections, uses `neuron_model.py` for the nodal model.
- `hco_gui.py`: Graphical interface for simulating a 2-neuron network with excitatory or inhibitory interconnections.

Models `neuron_model.py` and `network_model.py` provide definitions for conductance-based elements as well. An example of a conductance-based model is shown in:
- `gui_conductance.py`: Graphical interface for controlling a conductance-based equivalent of the current-based model in `gui.py`.

## Reference
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
```
@ARTICLE{ribar2020neuromorphic,
title={Neuromorphic Control}, 
author={L. {Ribar} and R. {Sepulchre}},
year={2020},
eprint={2011.04441},
archivePrefix={arXiv},
primaryClass={eess.SY}}
```
## License
MIT
