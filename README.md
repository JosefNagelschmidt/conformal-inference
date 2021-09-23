# Introduction

This is the electronic appendix of my master thesis ``Prediction bands for the individual treatment effect`` at the
University of Bonn, summer term 2021.

The emphasis is on three aspects:

- Develop a working and stable python implementation for several split conformal prediction methods commonly 
found in regression settings across the literature
- Provide examples to demonstrate the usage of the implemented main class
- Make simulations reproducible via the specification of a python (and R) environment, and
the usage of a build system (``pytask``) which automatically runs the simulations and writes results into a new ``./bld`` directory

## The main class

While I originally used a functional implementation for my calculations (found in  ``./src/legacy/functions.py``) and launched the simulations from jupyter notebooks (``./src/legacy/``), for replication purposes and usability I decided to implement all 
conformal methods under one class, which can be found in ``./src/conformal_methods/split_conformal_inference.py``. This class 
implements the fitting and prediction of the following four conformal methods:

- split conformal prediction intervals, as in ``Lei, G’Sell, et al. (2018)``
- locally-weighted split conformal prediction intervals, as in ``Lei, G’Sell, et al. (2018)``
- conformalized quantile regression, as in ``Romano, Patterson, and Candes (2019)``
- distributional conformal prediction, as in ``Chernozhukov, Wüthrich, and Zhu (2019)``

The class has been checked with unit tests (``./src/conformal_methods/test_split_conformal_class.py``) and the ``pytest`` module and seems to work as expected (feel free to run those tests on your machine).

Examples of how to use the main class in practise can be found in ``./examples``.

## Replication

Some information for building this project (if you would like to fully replicate the main results from the thesis):

- The project has been tested on Ubuntu 20.04 LTS
- The project requires an installed ``R`` interpreter on your machine (tested under version 4.1.1), as well as the conda package management system. For the python part, the project makes use of a conda environment, as specified in the ``requirements.yml``. To create an equivalent new conda environment on your local machine, run ``conda env create --name my_local_env_name --file=requirements.yml`` in the root of the cloned repository.
- You have to activate the conda environment via ``conda activate my_local_env_name`` and then run ``conda develop .`` before proceeding.
- The `pytask` build system ``https://github.com/pytask-dev/pytask`` will come along with the conda environment and enables you to run `pytask` in a terminal opened in the root of the project. This will run all simulations and would probably take some days to complete; therefore I have set the default repetitions parameter for each simulation (``n_sims``) to a smaller number - feel free to change this value in the files under ``./src/simulations/specs/``.
