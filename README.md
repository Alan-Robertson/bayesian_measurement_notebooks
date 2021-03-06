## Bayesian Circuit Design for Measurement Error Mitigation ##

These notebooks should be invoked using the `jupyter notebook` command.

### Requirements ###

*Software*
* python           : 3.8.3
* jupyter core     : 4.7.1
* qtconsole        : 4.7.7
* ipython          : 7.20.0
* ipykernel        : 5.3.4
* jupyter client   : 6.1.7
* nbconvert        : 6.0.7
* ipywidgets       : 7.6.3
* nbformat         : 5.1.2
* traitlets        : 5.0.5

*Python Libraries:*
* qiskit           : 0.17.4
* numpy            : 1.20.3
* matplotlib       : 3.3.3
* seaborn          : 0.11.0
* qinfer           : 1.0

IBMQ Credentials are required for experiments that access those devices.

### Notebooks ###

All notebooks are standalone and collect all data required before performing the associated experiments.

Runtimes will vary from a few seconds to several weeks for the larger experiments.

* Demonstration of Biased Measurement - Applies sequential X circuits to the IBMQ Armonk, London and Quito devices and prints results

* Armonk Sequential X Results - Runs SMC for the barrierred sequential X circuit on the IBMQ Armonk Device

* Biased Noise - Notebook for constructing toy biased and correlated error channels
* Correlated BV - Notebook for applying the toy biased and correlated error channels to BV circuits

* BV - Notebook for the construction the Bernstein Vazarani Algorithm, largely taken from the IBMQ Qiskit Documentation

* Melbourne Hamming Weight - Experiments on the IBMQ Melbourne using states of varying Hamming weight.

* RB Linear - SMC over Randomised Benchmarking Circuits

* More RB - Some more experiments involving randomised benchmarking with large circuit depths.

* Multi Qubit Bias - Hamming weight experiments for the IBMQ London and IBMQ Virgo

* QAOA - Applies the QAOA Algorithm and experiments on SMC with QAOA, makes extensive use of code from the qiskit documentation 

* Simple Measurement Bias - Some toy models for testing the SMC against measurement bias

* SMC Unitary Testing - Testing SMC methods against arbitrary SU(2) rotations

* Unitary Test - More testing SMC methods against SU(2) rotations

