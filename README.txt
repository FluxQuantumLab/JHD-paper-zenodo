# Transconductance quantization in a topological Josephson tunnel junction circuit
By Léo Peyruchat, Joël Griesmar, Jean-Damien Pillet, Çağlar Girit

Python source code to generate data from https://arxiv.org/abs/2009.03291

The code is separated in two parts: python libraries containing generic functions, and jupyter notebook files to generate and display data shown in the article.

Python libraries:
* topo_generic.py - generic functions to compute eigenenergies, eigenstates and Berry curvature.
* cpp.py - functions to return the hamiltonian of the cpp, and to minimize the excitation energy in the parameter space.
* JHD.py - function to compute the hamiltonian of the JHD, its gradient, and to minimize the excitation energy in a subset of the parameter space.

Jupyter Notebook files: one .ipynb file is associated to each figure. It contains:
* the code to generate the data
* pre-generated data stored in fig*_data, used for the article figures
* display of the plot as in the article

The conda configuration, with all the packages used for this work, can be installed following these instructions:

* install [miniconda](http://conda.pydata.org/miniconda.html)
* install all dependencies with
```
conda env create -f environment.yml
```
* activate new conda env
```
source activate JHD
```
* open jupyter lab
```
jupyter-lab
```