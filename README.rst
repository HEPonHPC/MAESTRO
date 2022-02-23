===========================
Introduction to MF-STRO-DF
===========================

Overview of MF-STRO-DF
~~~~~~~~~~~~~~~~~~~~~~~~

The problem being considered here involves fitting Monte Carlo simulations that
describe complex phenomena to experiments. This is done by finding parameters
of the resource intensive and noisy simulation that yield the least squares
objective function value to the noisy experimental data. This problem is solved
using a stochastic trust-region optimization algorithm where in each iteration,
a local approximation of the simulation signal and of the simulation noise is
constructed over data, which is obtained by running the simulation at strategically
placed design points within the trust-region around the current iterate. Then
the simulation components of the objective are replaced by their approximations
and this analytical and closed-form optimization problem is solved to find the
next iterate within the trust-region. Then the trust region is moved and the
iterations continue until a satisfactory convergence criteria is met.

.. _mfstrodf_dependencies:

Dependencies
~~~~~~~~~~~~

Required dependencies:

* Python_ 3.7
* NumPy_ 1.15.0 or above
* SciPy_ 1.5.4 or above
* pandas_ 1.1.5 or above
* pyDOE_ 0.3.8 or above
* pyDOE2_ 1.3.0 or above
* sklearn_
* numba_ 0.40.0 or above
* h5py_ 2.8.0 or above
* apprentice_ (DFO branch) 1.0.6 or above

Optional dependencies:

* matplotlib_ 3.0.0 or above
* GPy_ 1.9.9 or above

For running with the mpi4py parallelism:

* A functional MPI 1.x/2.x/3.x implementation, such as MPICH_, built with shared/dynamic libraries
* mpi4py_ v3.0.0 or above

.. _mfstrodf_initial_install:

Installation
~~~~~~~~~~~~

Before installing MF-STRO-DF, the DFO branch of apprentice_ needs to be installed first::

    git clone -b DFO --single-branch git@github.com:HEPonHPC/apprentice.git
    cd  apprentice/
    pip install .

    cd ..

Then proceed to installing MF-STRO-DF::

    git clone git@bitbucket.org:mkrishnamoorthy/workflow.git
    cd workflow
    pip install .

If you want to run with mpi4py parallelism you need to have a functional MPI
1.x/2.x/3.x implementation, such as MPICH_, built with shared/dynamic libraries
and install mpi4py using the command below. This step is optional and
if mpi4py is not installed, all code will automatically run on a single rank::

    pip install mpi4py


.. _mfstrodf_test_the_install:

Testing the installation
~~~~~~~~~~~~~~~~~~~~~~~~

Now we will test whether the installation was successful. First, create a log
directory in the root folder - where apprentice and workflow projects are located
(instead of running these commands at the root folder location, you can instead
run these commands at the ``/tmp`` on unix systems)::

    cd ..
    mkdir -p log/workflow/simpleapp/sumOfDiffPowers

Then from the current directory, go to the ``workflow/mfstrodf`` directory location::

    cd <location of workflow project>/workflow/mfstrodf/

Then run the MF-STRO-DF algorithm on a simple application with the noisy `sum of different powers`_
function as the Monte Carlo simulator to check whether the installation was successful and everything
is set up properly. If everything is running properly, then you should see the optimization output as
described in :ref:`MF-STRO-DF output<mfstrodf_output>`::

    python optimizationtask.py
      -a ../parameter_config_backup/simpleapp_sumOfDiffPowers/algoparams.json
      -c ../parameter_config_backup/simpleapp_sumOfDiffPowers/config.json
      -d ../../log/workflow/simpleapp/sumOfDiffPowers/WD

.. _`sum of different powers`: https://www.sfu.ca/~ssurjano/sumpow.html
.. _apprentice: https://github.com/HEPonHPC/apprentice
.. _h5py: https://www.h5py.org
.. _numba: https://numba.pydata.org
.. _sklearn: https://scikit-learn.org/stable/
.. _matplotlib: https://matplotlib.org
.. _pyDOE: https://pythonhosted.org/pyDOE/
.. _pyDOE2: https://pypi.org/project/pyDOE2/
.. _pandas: https://pandas.pydata.org
.. _Conda: https://docs.conda.io/en/latest/
.. _mpi4py: https://bitbucket.org/mpi4py/mpi4py
.. _MPICH: http://www.mpich.org/
.. _NumPy: http://www.numpy.org
.. _PyPI: https://pypi.org
.. _SciPy: http://www.scipy.org
.. _Python: http://www.python.org
.. _GPy: https://gpy.readthedocs.io/en/deploy/
