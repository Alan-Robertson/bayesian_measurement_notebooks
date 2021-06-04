## Bayesian Circuit Design for Measurement Error Mitigation ##

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


### Notebooks ###

* Demonstration of Biased Measurement - Applies sequential X circuits to the IBMQ Armonk, London and Quito devices and prints results

* Armonk Sequential X Results - Runs SMC for the barrierred sequential X circuit on the IBMQ Armonk Device

* Biased Noise - Notebook for constructing toy biased and correlated error channels
* Correlated BV - Notebook for applying the toy biased and correlated error channels to BV circuits

### Old Notebooks ### 
These notebooks were used as scratchpads and are in an unclean state. They are provided for completeness. They may be found in the `retired' directory.

* hamming_weight_comparison - Experimenting with comparing the hamming weight with the error rate
