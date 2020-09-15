# JHD-paper-zenodo

Python source code to generate data in https://arxiv.org/abs/2009.03291

Python version used: 3.7.3

The code is separated in two parts: python libraries containing generic functions, and jupyter notebook files to generate and display data shown in the article.

Python libraries:
* topo_generic.py - generic functions to compute eigenenergies, eigenstates and Berry curvature.
* cpp.py - functions to return the hamiltonian of the cpp, and to minimize the excitation energy in the parameter space.
* JHD.py - function to compute the hamiltonian of the JHD, its gradient, and to minimize the excitation energy in a subset of the parameter space.

Jupyter Notebook files: one .ipynb file is associated to each figure. It contains:
* the code to generate the data
* pre-generated data stored in fig*_data, used for the article figures
* display of the plot as in the article

Packages version used:
* numpy 1.19.1
* scipy 1.5.1
* jupyter 1.0.0
* jupyterlab 2.1.5