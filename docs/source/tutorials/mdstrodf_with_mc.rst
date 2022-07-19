=============================================
MÆSTRO with a Monte Carlo simulator
=============================================

.. _maestro_tutorial_mc:

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this tutorial, we describe how to run the MÆSTRO algorithm over a Monte
Carlo simulator to generate optimal parameter tunes. We do this using the
follwoing steps:

* Test the install
* Set the algorithm parameters
* Set the configuration inputs

  * Setup the Monte Carlo simulator

    * As a function call
    * As a script run
    * As a decaf_ - henson_ workflow run

  * Select a surrogate model function and setup its parameters
  * Select a function structure and and setup its parameters

* Run MÆSTRO
* Understand the output

Getting started
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install mæstro, execute the following commands::

    git clone git@github.com:HEPonHPC/maestro.git
    cd maestro
    pip install .

Then, test the installation as described in the
:ref:`test installation documentation<maestro_test_the_install>`.

Setting the algorithm parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we need to select the algorithm parameters. More details about the
parameters expected, their data types, and examples can be found in the
:ref:`algorithm parameters documentation<maestro_input_algo_parameters>`.
Here is the example JSON file with algorithm parameters for a minified problem
with a small subset of observables generated using `Pythia 8 Monte Carlo event generator`_
present in ``parameter_config_backup/miniapp/algoparams.json``.

  .. code-block:: json
    :force:
    :caption: parameter_config_backup/miniapp/algoparams.json

    {
      "tr": {
          "radius": 2.5,
          "max_radius": 3,
          "min_radius": 1e-5,
          "center": [
            1.3006076218684384,
            1.820541,
            0.28972391035308026
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
          [0,2],
          [0.2,2],
          [0,1]
      ],
      "kappa":100,
      "max_fidelity":100000,
      "usefixedfidelity":false,
      "N_p": 6,
      "dim": 3,
      "theta": 0.01,
      "thetaprime": 0.0001,
      "fidelity": 1000,
      "max_iteration":50,
      "max_fidelity_iteration":5,
      "min_gradient_norm": 0.00001,
      "max_simulation_budget":100000000000000000,
      "output_level":10
    }


Setting up the Monte Carlo simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next step is to setting up the Monte Carlo simulator. The simulator can be
run using a function call, executing a script, or in a decaf_ - henson_ workflow.

.. _maestro_tutorial_mc_function_call:

Setting up the Monte Carlo simulator using a function call
************************************************************************

To run the Monte Carlo simulator using a function call, write a class that is
inherited from the MC task base class ``MCTask``. In this class, you first define
the MC call function as ``run_mc(self):``. Then, define the other inherited but abstract
functions of ``MCTask`` in your own class and override any functions defined in ``MCTask``.
More information about ``MCTask`` is provided in the
:ref:`MC Task description<maestro_mctask>`.  Finally, you set your class along with the
relevant parameters in the mc object configuration.

As an example, the MC call function for
miniapp within ``maestro/mc/miniapp.py`` is shown below.

.. code-block:: python
    :linenos:
    :caption: maestro/mc/miniapp.py

    # MiniApp should inherit MCTask
    class MiniApp(MCTask):
      def run_mc(self):
        # In this tutorial, we demonstrate how to run miniapp MC in serial. If you
        # want to run miniapp MC in parallel, see the run_mc function
        # in maestro/mc/miniapp.py

        # Get a list of parameter directory (defined in superclass MCTask)
        dirlist = self.get_param_directory_array(self.mc_run_folder)
        for dno,d in enumerate(dirlist):
            # Get parameter from the directory (defined in superclass MCTask)
            param = self.get_param_from_directory(d) # from super class
            # Get fidelity from the directory (defined in superclass MCTask)
            run_fidelity = self.get_fidelity_from_directory(d) # from super class

            if run_fidelity !=0:
                # Set the output file path
                outfile = os.path.join(d,"out_curr{}.yoda".format(rank))
                # Execute the miniapp MC command.
                # mc_location is defined in the mc object configuration
                # (see line 5 in the mc object configuration JSON below)
                p = Popen(
                  [
                    self.mc_parmeters['mc_location'],
                    str(param[0]), str(param[1]), str(param[2]),
                    str(run_fidelity), str(np.random.randint(1,9999999)),
                    "0", "1", output_loc
                  ],
                  stdin=PIPE, stdout=PIPE, stderr=PIPE)
                p.communicate(b"input data that is passed to subprocess' stdin")
        comm.barrier()

For selecting this MC call function as the one to run within the MC
task, define the mc object configuration as shown below:

.. code-block:: json
  :linenos:
  :force:

  "mc":{
    "caller_type":"function call",
    "class_str":"MiniApp",
    "parameters":{
      "mc_location":"<location of miniapp MC executable>",
    }
  }

In this mc object configuration, set the ``caller_type`` as ``function call`` and the
``class_str`` as the class name defined above ``Miniapp``. Also, add all the parameters
that need to be sent to the MC task within ``parameters``.

.. _maestro_tutorial_MC_script:

Setting up the Monte Carlo simulator by executing a script
************************************************************************
To run the Monte Carlo simulator using a script call, a helper script is provided that will interleave
the calls to the optimization task and the MC task until the end of the
MÆSTRO algorithm. The MC task can be a script that calls the ``run_mc``
described in the subsection above or the MC task can directly call a MC executable.
These two approaches are describe in detail below.

Calling the MC task with a script that calls the ``run_mc`` function
=========================================================================

First, create a enclosing script that calls ``run_mc`` function. An example script
for miniapp that calls the ``run_mc`` function described above (see ``maestro/mc/bin/miniapp.py``)
is show below.

.. code-block:: python
    :linenos:
    :caption: maestro/mc/bin//miniapp.py

    if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run miniapp')
    parser.add_argument("-d", dest="MCDIR", type=str, default="log/MC_RUN",
                        help="MC directory")
    parser.add_argument("-c", dest="CONFIG", type=str, default=None,
                        help="Config file location")

    args = parser.parse_args()
    import json
    with open(args.CONFIG,'r') as f:
        ds = json.load(f)
    mc_parameters = ds['mc']['parameters']

    from maestro.mc import MiniApp
    mctask = MiniApp(args.MCDIR,mc_parameters)
    mctask.run_mc()

Next, set the appropriate mc configuration object for the script run

.. code-block:: json
  :linenos:
  :force:

  "mc":{
      "caller_type":"script run",
      "class_str":"MiniApp",
      "commands":[
          "<location of enclosing script> <location of MC directory> <location of config file>"
      ],
      "parameters":{

      }
    }

In the mc configuration object, set the ``caller_type`` as ``script run`` and the
``class_str`` as the name of your MC Task class e.g., ``Miniapp``. Also, add all the parameters
that need to be sent to the MC task within ``parameters``. Finally, add the
enclosing script call command within the ``commands`` array. This command will be used by
the interleaving helper script to call the MC task.

Calling the MC task by running the MC executable command
=========================================================================

To call the MC task by running the MC executable command directly, set the mc
configuration object for script run as shown below.

.. code-block:: json
  :linenos:
  :force:

  "mc":{
      "caller_type":"script run",
      "class_str":"MiniApp",
      "commands":[
          "<location of MC executable> <arguments to the MC executable>"
      ],
      "parameters":{

      }
    }

An example mc configuration object for this kind of MC task can be found in
``parameter_config_backup/a14app/config.json``.

.. _maestro_tutorial_mc_setting_decafhenson:

Setting up the Monte Carlo simulator in a decaf_ - henson_ workflow
************************************************************************

To run the Monte Carlo simulator within the decaf_ - henson_ workflow, a JSON object
with the task commands needs to be defined. As an example, such a JSON object for
miniapp within ``workflow/miniapp/decaf-henson.json`` is shown below.

.. code-block:: json
  :linenos:
  :force:

  {
    "workflow": {
        "filter_level": "NONE",
        "nodes": [
            {
             	"start_proc": 0,
                "nprocs": "<number of ranks>",
                "cmdline": "<project location>/maestro/optimization-task.py
                  -a <project location>/parameter_config_backup/miniapp/algoparams.json
                  -c <project location>/parameter_config_backup/miniapp/config.json
                  -d <working directory location>",
                "func": "opt_task_py",
                "inports": [],
                "outports": []
            },
            {
             	"start_proc": 0,
                "nprocs": "<number of ranks>",
                "cmdline": "<MC task command>",
                "func": "mc_task_py",
                "inports": [],
                "outports": []
            }
        ],
        "edges": [
        ]
    }
  }

In the JSON object above, ``<MC task command>``  is either the script that calls
the ``run_mc`` function or the MC executable command as shown in the ``commands``
array in :ref:`setting MC simulator by executing a script<maestro_tutorial_MC_script>`.
Also, the ``<number of ranks>`` is an integer number of ranks to use to run the
optimization task and MC task, ``<project location>`` is the location of the MÆSTRO project,
and ``<working directory location>`` is the lcoation of the working directory for this run

To call the MC task as a task of the workflow, set the mc
configuration object for miniapp as shown below.

.. code-block:: json
  :linenos:
  :force:

  "mc":{
      "caller_type":"workflow",
      "class_str":"MiniApp",
      "parameters":{

      }
    }

Selecting a surrogate model function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to select a predefined function or to create your own function in
``maestro/model.py`` to construct surrogate models.
Detailed instructions for selecting the appropriate function can be found in:

* reuse a :ref:`predefined model function<maestro_model_avail_func>` function
* :ref:`create your own model<maestro_model_create>` function

For this tutorial, we will construct the surrogate model using
:ref:`appr_pa_m_construct<maestro_model_avail_func_appr_pa_m>` function with the
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
``maestro/fstructure.py`` to get a f_structure object.
Detailed instructions for selecting the appropriate function can be found in:

* reuse a :ref:`predefined f_structure object<maestro_f_structure_avail_func>` function
* :ref:`create your own f_structure object<maestro_f_structure_create>` function

For this tutorial, we will get the f_structure object using
:ref:`appr_tuning_objective<maestro_f_structure_avail_func_appr_tuning_objective>`
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
``parameter_config_backup/miniapp/data.json`` and ``parameter_config_backup/miniapp/weights``,
respectively.

Setting the configuration inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration input consists of the objects from the last three steps.
So the configuration output for this tutorial is:

  .. code-block:: json
    :force:

    {
      "mc":"appropriate mc configuration object depending on whether the caller_type"
            "is function call, script run, or workflow",
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
:ref:`configuration input documentation<maestro_input_config>`.

Running MÆSTRO on your problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, we will assume that the :ref:`dependencies<maestro_dependencies>`
and apprentice_ are installed correctly as described in the
:ref:`initial installation test<maestro_initial_install>`.
Then, we install the MÆSTRO code by typing the following commands::

  cd maestro
  pip install .

Then, depending on the ``caller_type`` used, try the MÆSTRO algorithm on miniapp
using the commands below.

When ``caller_type`` is ``function call``
************************************************************************

.. code-block::
  :force:

  optimization-task
    -a <algorithm_parameters_JSON_location>
    -c <configuration_input_JSON_location>
    -d ../log/workflow/miniapp/<working_dir_name>

Here, replace ``<algorithm_parameters_JSON_location>`` and ``<configuration_input_JSON_location>``
with the correct location and assign an appropriate name in ``<working_dir_name>``.

When ``caller_type`` is ``script run``
************************************************************************

.. code-block::
  :force:

  maestro-run
    -a <algorithm_parameters_JSON_location>
    -c <configuration_input_JSON_location>
    -f <parameter_config_backup_location with data, weights, and other settings
            e.g., parameter_config_backup/miniapp>
    -d ../log/workflow/miniapp/<working_dir_name>
    -h <optional hostfile location>
    -n <total number of ranks to use (integer)>

Here, replace ``<algorithm_parameters_JSON_location>`` and ``<configuration_input_JSON_location>``
with the correct location and assign an appropriate name in ``<working_dir_name>``.
The optional hostfile contains list of nodes and number of ranks to use on these nodes.
The total number of ranks is the number of ranks to use as ``numProcs`` in ``mpirun`` calls of the
interleaving optimization and MC tasks.
If hostfile is specified, the total number of ranks to use should be the sum of
all the ranks used across all nodes.

When ``caller_type`` is ``workflow``
************************************************************************

.. code-block::
  :force:

  cd <location of decaf-henson JSON file>

  mpirun -np <number of ranks to use (integer)>
      <location of decaf-henson_python executable>/decaf-henson_python

The number of ranks to use should be the equal to or greater than the value set in the ``nprocs``
key  in the decaf-henson JSON file as shown in the the section on
:ref:`setting MC simulator in decaf-henson workflow<maestro_tutorial_mc_setting_decafhenson>`.

To run this command with a hostfile::

  cd <location of decaf-henson JSON file>

  mpirun -hostfile <hostfile location> -np <number of ranks to use (integer)>
      <location of decaf-henson_python executable>/decaf-henson_python

The hostfile contains list of nodes and number of ranks to use on these nodes.
Also, the number of ranks to use should be the sum of all the ranks used across all nodes.

Understanding the output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If every thing runs as expected, since :math:`output\_level\ge10` in the algorithm parameter input,
the output should contain a one line summary of each iteration of the MÆSTRO
algorithm run as described in the
:ref:`one line output documentation<maestro_output_single_line>`.

.. _decaf: https://link.springer.com/chapter/10.1007/978-3-030-81627-8_7
.. _henson: https://dl.acm.org/doi/10.1145/2907294.2907301
.. _apprentice: https://github.com/HEPonHPC/apprentice
.. _`Pythia 8 Monte Carlo event generator`: https://pythia.org
