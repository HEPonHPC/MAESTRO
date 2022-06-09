=============================================
MC Task
=============================================

.. _maestro_mctask:

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``MCTask`` at ``maestro/mc/mctask.py`` contains
useful utility functions that will allow you to interface with the MF-STRO-DF
algorithm with ease. Here, we describe the functions that need to be implemented
in your MC task class that inherits ``MCTask``
and the code documentation from ``MCTask`` class that describe the functions
already defined in this class.

Inherited functions not defined in MC Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following functions need to be implemented in your MC task class that inherits
``MCTask``. If these functions are not defined, when they are called, it may result in
an exception.

* ``run_mc(self)``: This function is used to execute the MC task as a function call. See
  :ref:`calling mc task as a function call tutorial<maestro_tutorial_mc_function_call>` for an example.
* ``merge_statistics_and_get_max_sigma(self)``: This function is used to merge
  statistics (for multi-fidelity runs) and get the maximum standard deviation
  across all parameters from the MC run.
* ``convert_mc_output_to_df(self, all_param_directory)``: Convert the output from the MC
  simulator for all parameters in the ``all_param_directory`` to a pandas_ data frame output. An example dataframe output is shown below.
  In the example below, there are three parameters and two terms of the objective function.
  Terms that end with ``.P`` are the parameters and those ending with ``.V`` are the values
  associated with either the ``MC``, i.e., MC sample values  or the ``DMC``, i.e., MC standard deviation
  values. You can add more rows, i.e, more sets of parameter and values  for additional terms in the objective function
  or more columns, i.e., more components of the each term of the objective that
  come from the MC simulator.::

        >df
                        MC                      DMC
        Term1.P        [[1,2],[3,4],[6,3]]     [[1,2],[3,4],[6,3]]
        Term1.V        [19, 18, 17]            [99, 98, 97]
        Term2.P        [[1,2],[3,4],[6,3]]     [[1,2],[3,4],[6,3]]
        Term2.V        [29, 28, 27]            [89, 88, 87]


MC Task code documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :ref:`of MC Task API description<maestro_codedoc_mctaskapi>` in the code documentation section for a complete
documentation of all implemented and abstract (not implemented) functions within ``MCTask``.


.. _pandas: https://pandas.pydata.org
