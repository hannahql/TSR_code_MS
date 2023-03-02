# TSR_code_MS
Code supporting the simulations in "Experimental Design in Two-Sided Platforms: An Analysis of Bias" by Ramesh Johari, Hannah Li, Gabriel Weintraub, and Inessa Liskovich. Management Science '22.

Customer-side, listing-side, and two-sided experiments are implemented in the mean field model as well as in the finite system Markov chain model. 

In the mean field model, we characterize the steady state behavior by solving the optimization problem defined in Theorem 1. We solve these solutions numerically and analyze the bias of the estimators in these numerics.

In the finite system model, we implement the continuous time Markov chain defined in the paper, with a fixed number of listings and with customers arriving over time. We study the bias, standard error, and root mean squared error (RMSE) of the estimators in the finite setting. 

# Usage

See attached Jupyter notebook for an implementation of both numerics and simulations. 

The user defines the model parameters of customer types, listing types, listings' rate of replenishment, customer arrival rates, consideration probability, and treatment and control utilities. The code studies the performance of the different experiment types under these parameters.


# Package and Language requirements. 

This code is written in Python 3. 

The code requires the following packages: numpy, pandas, matplotlib.pyplot, seaborn, copy, csv, time
itertools, operator, importlib, json, random, pdb, os, multiprocessing, functools, and scipy.
