===========================
MF-STRO-DF with simpleapp
===========================

.. _mfstrodf_tutorial_simpleapp:

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Simpleapp is a an application comprised of simple functions to which
noise is added to emulate the Monte Carlo simulator.

In this tutorial, we describe how to setup a simpleapp problem, run the
MF-STRO-DF algorithm over it to generate optimal parameters. For this, we
will go through the follwoing steps:

* Test the install
* Set the algorithm parameters
* Set the configuration inputs

  * Select a simple MC function and setup its parameters
  * Select a surrogate model function and setup its parameters
  * Select a function structure and setup its parameters

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
Here is the example algorithm parameters JSON file for the simpleapp
present in ``parameter_config_backup/simpleapp/algoparams.json``.

  .. code-block:: json
    :force:
    :caption: parameter_config_backup/simpleapp/algoparams.json

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
        "max_fidelity":1e9,
        "usefixedfidelity":false,
        "N_p": 25,
        "dim": 3,
        "theta": 0.01,
        "thetaprime": 0.0001,
        "fidelity": 1000,
        "max_iteration":50,
        "max_fidelity_iteration":5,
        "min_gradient_norm": 1e-5,
        "max_simulation_budget":1e16,
        "output_level":10
    }

Selecting a simple MC function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next step is to select a functions to which noise will be added to
emulate a Monte Carlo simulator. The function needs to be written in Python_ 3.7.
This function should be written in a class inside ``mfstrodf/mc/simpleapp.py`` and
this function should be a static method called ``mapping``.
Currently, the following four functions are available for use with simpleapp (see
``mfstrodf/mc/simpleapp.py``):

* `sum of different powers`_
* `rotated hyper-ellipsoid`_
* `sphere`_
* `sum of squares`_

As an example, the `sum of different powers`_ function within
``mfstrodf/mc/simpleapp.py`` is shown below.

.. code-block:: python
    :linenos:
    :caption: mfstrodf/mc/simpleapp.py

    class SumOfDiffPowers():
    @staticmethod
    def mapping(x):
        s = 0
        for i in range(len(x)):
            n = (abs(x[i])) ** (i + 2)
            s = s + n
        return s

``SimpleApp`` inherits ``MCTask`` that contains
useful utility functions that will allow you to interface with the MF-STRO-DF
algorithm with ease. More information can be found in the
:ref:`MC Task description<mfstrodf_mctask>`.

For this tutorial, we will select all four functions mentioned above with simpleapp.
This is done using the following mc object configuration:

  .. code-block:: json
    :force:

      "mc":{
      "caller_type":"function call",
      "class_str":"SimpleApp",
      "parameters":{
        "functions":["SumSquares", "Sphere", "RotatedHyperEllipsoid", "SumOfDiffPowers"]
      }
    }

Selecting a surrogate model function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to select a predefined function or to create your own function in
``mfstrodf/model.py`` to construct surrogate models.
Detailed instructions for selecting the appropriate function can be found in:

* reuse a :ref:`predefined model function<mfstrodf_model_avail_func>` function
* :ref:`create your own model<mfstrodf_model_create>` function

For this tutorial, we will construct the surrogate model using
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

Selecting the function structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to select a predefined function or to create your own function in
``mfstrodf/fstructure.py`` to get a f_structure object.
Detailed instructions for selecting the appropriate function can be found in:

* reuse a :ref:`predefined f_structure object<mfstrodf_f_structure_avail_func>` function
* :ref:`create your own f_structure object<mfstrodf_f_structure_create>` function

For this tutorial, we will get the f_structure object using
:ref:`appr_tuning_objective<mfstrodf_f_structure_avail_func_appr_tuning_objective>`
function with the following f_structure object configuration:

  .. code-block:: json
    :force:

    "f_structure":{
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

Note that if the data and weights keys are not specified in the parameter object
of the ``f_structure`` configuration, then a data value of ``[1,0]`` and a weight of ``1`` is
assumed for each term of ``appr_tuning_objective``.
If you want to specify your own data and weights, then assign complete path of the
data and weights files to the ``data`` and ``weights`` keys, respectively in
the ``parameter`` object of the ``f_structure`` configuration.
Exampe data and weights files for this tutorial can be found in
``parameter_config_backup/simpleapp/data.json`` and ``parameter_config_backup/simpleapp/weights``,
respectively.

Setting the configuration inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration input consists of the objects from the last three steps.
So the configuration output for this tutorial is:

  .. code-block:: json
    :force:

      {
        "mc":{
        "caller_type":"function call",
        "class_str":"SimpleApp",
        "parameters":{
          "functions":["SumSquares", "Sphere", "RotatedHyperEllipsoid", "SumOfDiffPowers"]
        }
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
      "f_structure":{
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

Then try the MF-STRO-DF algorithm on the simpleapp using the command::

  optimization-task
    -a <algorithm_parameters_JSON_location>
    -c <configuration_input_JSON_location>
    -d ../log/workflow/simpleapp/<working_dir_name>

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
.. _`rotated hyper-ellipsoid`: https://www.sfu.ca/~ssurjano/rothyp.html
.. _`sphere`: https://www.sfu.ca/~ssurjano/spheref.html
.. _`sum of squares`: https://www.sfu.ca/~ssurjano/sumsqu.html
.. _apprentice: https://github.com/HEPonHPC/apprentice
