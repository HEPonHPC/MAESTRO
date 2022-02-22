===========================
MF-STRO-DF with simpleapp
===========================

.. _mfstrodf_tutorial_simpleapp:

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this tutorial, we describe how to setup a simpleapp problem, run
the MF-STRO-DF algorithm over it to generate optimal parameters. For this, we
will go through the follwoing steps:

* Test the install
* Set the algorithm parameters
* Set the configuration inputs

  * Select a simple MC function
  * Select a model function
  * Select a subprobmel function

* Run MF-STRO-DF on simpleapp
* Understand the output

Getting started
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before installing MF-STRO-DF, the DFO branch of apprentice_ needs to be installed first::

    git clone -b DFO --single-branch git@github.com:HEPonHPC/apprentice.git
    cd  apprentice/
    pip install .

    cd ..

Then proceed to installing MF-STRO-DF::

    git clone git@bitbucket.org:mkrishnamoorthy/workflow.git
    cd workflow
    pip install .

Then, test the installation as described in the
:ref:`test installation documentation<mfstrodf_test_the_install>`.

Setting the algorithm parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we need to select the algorithm parameters. More details about the
parameters expected, their data types, and examples can be found in the
:ref:`algorithm parameters documentation<mfstrodf_input_algo_parameters>`.
Here is the example algorithm parameters JSON file for the sum of different powers`_
simpleapp present in ``parameter_config_backup/simpleapp_sumOfDiffPowers/algoparams.json``.

  .. code-block:: json
    :force:
    :caption: parameter_config_backup/simpleapp_sumOfDiffPowers/algoparams.json

    {
      "tr": {
          "radius": 5,
          "max_radius": 20,
          "min_radius": 1e-5,
          "center": [
              2.13681979279795745,
              -2.7717580840968665,
              3.2082901878570345
          ],
          "mu": 0.01,
          "eta": 0.01
      },
      "param_names": [
        "x",
        "y",
        "z"
      ],
      "param_bounds": [
        [-5,5],
        [-5,5],
        [-5,5]
      ],
      "kappa":100,
      "max_fidelity":1000000,
      "usefixedfidelity":false,
      "N_p": 10,
      "dim": 3,
      "theta": 0.01,
      "thetaprime": 0.0001,
      "fidelity": 1000,
      "max_iteration":50,
      "max_fidelity_iteration":5,
      "min_gradient_norm": 0.00001,
      "max_simulation_budget":10000000,
      "output_level":10
    }

Selecting a simple MC function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next step is to select a simple function to which noise will be added to
emulate a Monte Carlo simulator. The function needs to be written in Python_ 3.7.
This function should be a static method called ``mapping`` within a class
that inherits ``SimpleApp``. This class should be added to ``mfstrodf/mc/simpleapp.py``
An example `sum of different powers`_ function is written in Python_ and present
in ``mfstrodf/mc/simpleapp.py``. This code is shown below.

.. code-block:: python
    :linenos:
    :caption: mfstrodf/mc/simpleapp.py

    # This class can inherits SimpleApp if you want to reuse the utility
    # functions in SimpleApp. Otherwise, it has to inherit MCTask, and implement
    # your own versions of the abstract functions in MCTask
    class SumOfDiffPowers(SimpleApp):
      # This need to be a static method that is called mapping
      @staticmethod
      def mapping(x):
        sum = 0
        for ii in range(len(x)):
          xi = x[ii]
          new = (abs(xi)) ** (ii + 2)
          sum = sum + new
        # return a single floating point number
        return sum

``SimpleApp`` itself inherits ``MCTask``. Both SimpleApp and ``MCTask`` contain
useful utility functions that will allow you to interface with the MF-STRO-DF
algorithm with ease. More information about the interface of these methods can be
found in their :ref:`function documentation<mfstrodf_code_doc>`.

For this tutorial, we will select the SumOfDiffPowers simpleapp with the following
mc object configuration:

  .. code-block:: json
    :force:

      "mc":{
        "caller_type":"function call",
        "class_str":"SumOfDiffPowers",
        "parameters":{}
      }

Selecting a model function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to select a predefined function or to create your own function in
``mfstrodf/model.py`` to construct models.
Detailed instructions for selecting the appropriate function can be found in:

* reuse a :ref:`predefined function functiom<mfstrodf_model_avail_func>`
* :ref:`create your own function<mfstrodf_model_create>` model

For this tutorial, we will construct the model using
:ref:`appr_pa_m_construct<mfstrodf_model_avail_func_appr_pa_m>` function with the
following model object configuration:

  .. code-block:: json
    :force:

    "model":{
      "function_str":{
        "MC":"appr_pa_m_construct",
        "DMC":"appr_pa_m_construct"
      },
      "parameters":{
        "MC":{"m":2},
        "DMC":{"m":1}
      }
    }

Selecting a subprobmel function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to select a predefined function or to create your own function in
``mfstrodf/subproblem.py`` to get a subproblem object.
Detailed instructions for selecting the appropriate function can be found in:

* reuse a :ref:`predefined function functiom<mfstrodf_subproblem_avail_func>`
* :ref:`create your own function<mfstrodf_subproblem_create>` model

For this tutorial, we will get the subproblem object using
:ref:`appr_tuning_objective<mfstrodf_model_avail_func_appr_tuning_objective>`
function with the following subproblem object configuration:

  .. code-block:: json
    :force:

    "subproblem":{
      "parameters":{
        "optimization":{
          "nstart":5,
          "nrestart":10,
          "saddle_point_check":false,
          "minimize":true,
          "use_mpi":true
        }
      },
      "function_str":"appr_tuning_objective"
    }

Setting the configuration inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration input consists of the objects from the last three steps
So the configuration output for this tutorial is:

  .. code-block:: json
    :force:

    {
      "mc":{
        "caller_type":"function call",
        "class_str":"SumOfDiffPowers"
        "parameters":{}
      },
      "model":{
        "function_str":{
          "MC":"appr_pa_m_construct",
          "DMC":"appr_pa_m_construct"
        },
        "parameters":{
          "MC":{"m":2},
          "DMC":{"m":1},
        }
      },
      "subproblem":{
        "parameters":{
          "optimization":{
            "nstart":5,
            "nrestart":10,
            "saddle_point_check":false,
            "minimize":true,
            "use_mpi":true
          }
        },
        "function_str":"appr_tuning_objective"
      }
    }

More information about the key expected, their definition, their data types,
and examples can be found in the
:ref:`configuration input documentation<mfstrodf_input_config>`.

Running MF-STRO-DF on your problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, we will assume that the :ref:`dependencies<mfstrodf_dependencies>`
and apprentice_ are installed correctly as described in the
:ref:`initial installation test<mfstrodf_initial_install>`.
Then, we install the workflow code by typing the following commands::

  cd workflow
  pip install .

Then, create a log directory in the root folder - where apprentice and workflow projects are located
(instead of running these commands at the root folder location, you can instead
run these commands at the ``/tmp`` on unix systems)::

  cd ..
  mkdir -p log/workflow/simpleapp/sumOfDiffPowers

Then from the current directory, go to the ``workflow/mfstrodf`` directory location::

    cd <location of workflow project>/workflow/mfstrodf/

Then try the MF-STRO-DF algorithm on the `sum of different powers`_ simpleapp using the command::

  python optimizationtask.py
    -a <algorithm_parameters_JSON_location>
    -c <configuration_input_JSON_location>
    -d ../../log/workflow/simpleapp/sumOfDiffPowers/<working_dir_name>

Here, replace ``<algorithm_parameters_JSON_location>`` and ``<configuration_input_JSON_location>``
with the correct location and assign an appropriate name in ``<working_dir_name>``.

Understanding the output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If every thing runs as expected, since :math:`output\_level\ge10` in the algorithm parameter input,
the output should contain a one line summary of each iteration of the MF-STRO-DF
algorithm run as described in the
:ref:`one line output documentation<mfstrodf_output_single_line>`.

.. _Python: http://www.python.org
.. _`sum of different powers`: https://www.sfu.ca/~ssurjano/sumpow.html
.. _apprentice: https://github.com/HEPonHPC/apprentice
