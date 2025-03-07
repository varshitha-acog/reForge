Getting Started
===============

Installation
------------

The "installation" of reForge consists of including the reforge project directory 
in your Python path, and making sure that you have all the needed dependencies.
All of the packages that reforge depends on can be installed with conda and/or pip.

1. **Clone the repository:**

.. code-block:: bash

    git clone https://github.com/DanYev/reforge.git


2. **Install the virtual environment:**

.. code-block:: bash

    module load mamba
    mamba env create -n reforge --file environment.yml
    source activate reforge


3. **Include reforge in your Python path OR install it via pip:**

.. code-block:: bash

    export PYTHONPATH=$PYTHONPATH:path/to/reforge/repository
    # or
    pip install -e . # from the reforge repository directory     


Testing the setup
-----------------

From the reforge repository directory, you can run the tests with the following command:

.. code-block:: bash

    cd path/to/reforge/repository
    bash run_tests.sh --all


For users of SOL and PHX clusters, you first need to initiate the interactive session with GPU support or
better - submit a job to the GPU queue:

.. code-block:: bash

    cd path/to/reforge/repository
    sbatch run_tests.sh --all


Running the Examples
--------------------

At the moment, the coarse-grained examples can only be run with GROMACS. OpenMM support is in active development. 
Thus, to run the tutorials, you need to have GROMACS installed on your system.

.. warning::

    For users of SOL and PHX clusters!!! Last time I checked, some versions of GROMACS 
    were not working properly on the SOL and PHX clusters. The current setup is tested with 
    the following versions:

.. code-block:: bash

    module load gromacs-2023.3-openmpi-cuda-qx # for PHX
    module load gromacs/2023.4-gpu-mpi # for SOL

Some basic examples can be found here `examples <https://github.com/DanYev/cgtools/tree/main/docs/examples>`_, 
and will be updated as the project progresses.

