# Bayesian Methods for Machine Learning
Repository containing my implementations and solutions of Bayesian algorithms for machine learning 
as per the coding assignments in the "Bayesian Methods for Machine Learning" course by the 
HSE University, available here: https://www.coursera.org/learn/bayesian-methods-in-machine-learning  

The repository is structured in the following way; folders are:
- `data`: contains some of the input data required to run some parts of the code, as well as the images generated by the code

- `notebooks`: contains the notebooks with my solutions to the assignments, exactly as provided in the 
course. Apart from adding my solutions, the notebooks have not been modified / edited in any way.

- `src`: this folder contains the work of the various assignments, written in a modular and object-oriented fashion.
The modules within this folders are designed to be used as an out-of-the-box package. While their content and features have
been inspired by the course assignments, they have also been heavily modified and streamlined by
myself. If you want to see the original assignments, check out the `notebooks` folder instead. Please note
that the content of the assignments of weeks 2 (Expectation-Maximization Algorithm), 4 (Markov-Chain Monte Carlo model) 
and 6 (Gaussian Processes and Bayesian Optimization) are featured here, whereas the content of week 5 and
 of the final assignment (Variational Auto-Encoders) is only available in the notebook format.
 
- `scripts`: the scripts to run the modules in the `src` folder. Provided you have all your environment correctly set
up with the required installed libraries (see `requirements.txt` for the libraries needed), these scripts should run 
out-of-the-box without the need for modification.


NOTES: 
 - This code is written for Python 3.9 and may not work in earlier versions.
 - `tensorflow` version `1.x` is required to run some of the notebooks, but given it's not required to run modules in the
`src` folder, I have nto included it in the `requirements.txt` file.
