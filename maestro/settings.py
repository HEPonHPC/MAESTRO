import copy
import pprint

import os,sys
import json
import time
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.16f')
import numpy as np


class Settings(object):
    """
    Settings and state object
    """

    def __init__(self,working_dir=None, algorithm_parameters_file=None, config_file=None,param_matadata_file=None):
        """

        Initialize the algorithms state and settings. Three main settings/state
        file include the algorithm file, configuration and parameter metadata file

        :param working_dir: working directory path
        :type working_dir: str
        :param algorithm_parameters_file: algorithm parameter file path
        :type algorithm_parameters_file: str
        :param config_file: configuration file path
        :type config_file: str
        :param param_matadata_file: parameter metadata file path
        :type param_matadata_file: str

        """
        self.algorithm_parameters_dict = {}
        self.config_dict = {}
        self.param_meta_data = {}
        if algorithm_parameters_file is None and config_file is None:
            WD = WorkingDirectory(working_dir)
            algorithm_parameters_file = WD.get_log_path('algorithm_parameters_dump.json')
            config_file = WD.get_log_path('config_dump.json')

        self.initialize_algorithm_parameters(algorithm_parameters_file)
        self.initialize_config(config_file, working_dir)

        if param_matadata_file is None:
            param_metadata_file = self.working_directory.get_log_path("parameter_metadata_dump.json")
            if os.path.exists(param_metadata_file):
                self.param_meta_data = self.read_setting_file(param_metadata_file)

    def read_setting_file(self, file):
        """

        Read settings and state JSON file

        :param file: file path
        :type file: str
        :return: JSON structure from the file
        :rtype: dict

        """
        try:
            with open(file, 'r') as f:
                ds = json.load(f)
        except:
            raise Exception("setting file type not supported")
        return ds

    def write_setting_file(self, ds, file):
        """

        Write the settings/state file

        :param ds: Dictionary of state/settings
        :type ds: dict
        :param file: output file path
        :type file: str

        """
        from maestro import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            with open(file, 'w') as f:
                json.dump(ds, f, indent=4)

    def initialize_algorithm_parameters(self, file:str):
        """

        Check whether all required algorithm parameter keys are defined and
        initialize algorithm parameters

        :param file: algorithm file path
        :type file: str

        """
        ds = self.read_setting_file(file)
        self.algorithm_parameters_dict = ds
        """
        kappa: adaptive sampling constant. Used in the bound for MC standard dev to check MC accuracy
        N_p: minimum number of parameters in TR to run MC over to build models
        theta (not in WF diagram): Used as a factor in the calculation of minimum distance b/w parameters
        thetaprime (not in WF diagram): Used as a factor in the calculation of equivalence distance b/w parameters in 
                    the new pool and those in the previous pool to find close matches from p_pool for points in p_init
        
        max_fidelity_iteration: (k_\lambda in WF diagram): numbe of iterations for which the fidelity can be at max_fidelity
        min_gradient_norm (alpha in WF diagram): projected gradient norm
        """
        l1keys = ['tr','param_names','param_bounds','kappa','max_fidelity',
                  'N_p','dim','theta','thetaprime','fidelity','max_iteration','max_fidelity_iteration',
                  'min_gradient_norm','max_simulation_budget']
        """
        mu: model precision constant (used as a factor to check model precision if current iterate is close to minimum)
        eta: model fitness threshold (used as the bound for significant reduction condition rho)
        """
        tr_l2keys = ['radius','max_radius','min_radius','center','mu','eta']
        if 'simulation_budget_used' not in self.algorithm_parameters_dict:
            self.algorithm_parameters_dict['simulation_budget_used'] = 0
        # Do consisency check
        for key in l1keys:
            if key not in self.algorithm_parameters_dict.keys():
                raise Exception("algorithm parameter key not defined: "+key)
        for key in tr_l2keys:
            if key not in self.algorithm_parameters_dict['tr'].keys():
                raise Exception("algorithm parameter key not defined: tr."+key)
        if 'current_iteration' not in self.algorithm_parameters_dict:
            self.algorithm_parameters_dict['current_iteration'] = 0

    def initialize_config(self, file:str,working_dir:str=None):
        """

        Check whether all required configuration keys are defined and
        initialize configuration parameters

        :param file: configuration file path
        :type file: str
        :param working_dir: working directory path
        :type working_dir: str

        """
        ds = self.read_setting_file(file)
        self.config_dict = ds
        if working_dir is None and 'working_directory_str' not in self.config_dict:
            raise Exception("Working directory is required to continue")
        if working_dir is not None:
            self.config_dict['working_directory_str'] = working_dir
        self.config_dict['working_directory'] = WorkingDirectory(self.config_dict['working_directory_str'])
        # can be "function call","script run","workflow"
        # self.config_dict['mc'] = {}
        # self.config_dict['mc']['caller_type'] = ds['mc']['caller_type']
        # self.config_dict['mc']['ranks'] = ds['mc']['ranks']
        # self.config_dict['mc']['parameters'] = ds['mc']['parameters']
        try:
            import maestro.mc
            mc_class = getattr(maestro.mc,ds['mc']['class_str'])
            self.config_dict['mc']['object'] = mc_class(self.mc_run_folder_path,self.mc_parameters)
        except:
            raise Exception("MC class \""+ds['mc']['class_str']+"\" not found in maestro.mc")
        if self.mc_call_using_script_run and 'commands' not in ds['mc']:
            raise Exception("MC commands needs to be set in config_dict.mc.location_str for script call")

        # self.config_dict['model']['parameters'] = ds['model']['parameters']

        from maestro import ModelConstruction
        self.config_dict['model']['function'] = {}
        for data_name in ds['model']['function_str'].keys():
            try:
                method_to_call = getattr(ModelConstruction,ds['model']['function_str'][data_name])
                self.config_dict['model']['function'][data_name] = method_to_call
            except:
                raise Exception("Model function \""+ds['model']['function_str'][data_name]+"\" not found in maestro.ModelConstruction")

        # self.config_dict['f_structure']['parameters'] = ds['f_structure']['parameters']
        if 'model' not in self.config_dict['f_structure']['parameters']:
            self.config_dict['f_structure']['parameters']['model'] = {}
        if 'model_scaled' not in self.config_dict['f_structure']['parameters']:
            self.config_dict['f_structure']['parameters']['model_scaled'] = {}
        # self.config_dict['f_structure']['parameters']['optimization'] = ds['f_structure']['parameters']['optimization']

        try:
            from maestro import Fstructure
            method_to_call = getattr(Fstructure,ds['f_structure']['function_str'])
            self.config_dict['f_structure']['function'] = method_to_call
        except:
            raise Exception("Fstructure function \""+ds['f_structure']['function_str']+"\" not found in maestro.Fstructure")

        if 'algorithm_status_dict' in self.config_dict:
            self.algorithm_status.from_dict(self.config_dict['algorithm_status_dict'])
        else:
            self.algorithm_status.update_status(0)

    def set_start_time(self):
        """

        Set start time

        """
        self.config_dict['start_time'] = time.time()

    def set_end_time(self):
        """

        Set end time

        """
        self.config_dict['end_time'] = time.time()

    def save(self, next_step=None, to_log = False):
        """

        Save all setting/state parameters and variables

        :param next_step: next step in the optimization algorithm
        :type next_step: str
        :param to_log: true if the save is to log iteration state at the
            end of the iteration and if false the save is for dumping the
            state within an iteration
        :type to_log: bool

        """
        if to_log:
            algorithm_parameters_file = self.working_directory.get_log_path('algorithm_parameters_dump_k{}.json'.format(self.k))
            config_file = self.working_directory.get_log_path('config_dump_k{}.json'.format(self.k))
            param_metadata_file = self.working_directory.get_log_path('parameter_metadata_dump_k{}.json'.format(self.k))
        else:
            algorithm_parameters_file = self.working_directory.get_log_path('algorithm_parameters_dump.json')
            config_file = self.working_directory.get_log_path('config_dump.json')
            param_metadata_file = self.working_directory.get_log_path("parameter_metadata_dump.json")

        if next_step is not None:
            self.config_dict['next_step'] = next_step

        self.set_end_time()

        self.config_dict['algorithm_status_dict'] = self.algorithm_status.as_dict()
        if 'comm' in self.config_dict['f_structure']['parameters']['optimization']:
            self.config_dict['f_structure']['parameters']['optimization'].pop('comm')
        config_dict_to_save = copy.deepcopy(self.config_dict)
        if 'object' in config_dict_to_save['mc']: config_dict_to_save['mc'].pop('object')
        if 'function' in config_dict_to_save['model']: config_dict_to_save['model'].pop('function')
        if 'function' in config_dict_to_save['f_structure']: config_dict_to_save['f_structure'].pop('function')
        if 'working_directory' in config_dict_to_save: config_dict_to_save.pop('working_directory')
        if 'algorithm_status' in config_dict_to_save: config_dict_to_save.pop('algorithm_status')
        self.write_setting_file(self.algorithm_parameters_dict,algorithm_parameters_file)
        self.write_setting_file(config_dict_to_save,config_file)
        self.write_setting_file(self.param_meta_data, param_metadata_file)

    def update_simulation_budget_used(self, simulation_count):
        """

        Upadte used simulation budget by adding simulation_count to current
        simulation budget

        :param simulation_count: simulation count to be added to current simulation budget
        :type simulation_count: int

        """
        self.algorithm_parameters_dict['simulation_budget_used'] += simulation_count

    @property
    def simulation_budget_used(self):
        """

        Get simulation budget used

        :return: current simulation budget
        :rtype: int

        """
        return self.algorithm_parameters_dict['simulation_budget_used']

    @property
    def no_iters_at_max_fidelity(self):
        """

        Get number of iterations where the fidelity have been at maximum fidelity

        :return: number of iterations where the fidelity have been at maximum fidelity
        :rtype: int

        """
        if "no_iters_at_max_fidelity" in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['no_iters_at_max_fidelity']
        return 0
    def increment_no_iters_at_max_fidelity(self):
        """

        Increment number of iterations where the fidelity have been at maximum fidelity

        """
        if "no_iters_at_max_fidelity" in self.algorithm_parameters_dict:
            self.algorithm_parameters_dict['no_iters_at_max_fidelity']+=1
        else:
            self.algorithm_parameters_dict['no_iters_at_max_fidelity'] = 1

    @property
    def radius_at_which_max_fidelity_reached(self):
        """

        Get trust region radius when the fidelity reached the maximum fidelity

        :return: trust region radius when the fidelity reached the maximum fidelity
        :rtype: float

        """
        if "radius_at_which_max_fidelity_reached" in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['radius_at_which_max_fidelity_reached']
        return None

    def set_radius_at_which_max_fidelity_reached(self,tr_radius):
        """

        Set trust region radius when the fidelity reached the maximum fidelity

        :param tr_radius: trust region radius when the fidelity reached the maximum fidelity
        :type tr_radius: float

        """
        if not "radius_at_which_max_fidelity_reached" in self.algorithm_parameters_dict:
            if self.fidelity>=self.max_fidelity:
                self.algorithm_parameters_dict['radius_at_which_max_fidelity_reached'] = tr_radius

    @property
    def optimization_parameters(self):
        """

        Get trust region subproblem optimziation parameters

        :return: trust region subproblem optimziation parameters
        :rtype: dict

        """
        return self.config_dict['f_structure']['parameters']['optimization']

    @property
    def tr_eta(self):
        """

        Get trust region parameter :math:`\eta`

        :return: :math:`\eta`
        :rtype: float

        """
        return self.algorithm_parameters_dict['tr']['eta']

    @property
    def tr_min_radius(self):
        """

        Get minimum allowed trust region radius

        :return: minimum allowed trust region radius
        :rtype: float

        """
        return self.algorithm_parameters_dict['tr']['min_radius']

    @property
    def tr_max_radius(self):
        """

        Get maximum allowed trust region radius

        :return: maximum allowed trust region radius
        :rtype: float

        """
        return self.algorithm_parameters_dict['tr']['max_radius']

    @property
    def min_gradient_norm(self):
        """

        Get minimum norm of projected gradient

        :return: minimum norm of projected gradient
        :rtype: float

        """
        return self.algorithm_parameters_dict['min_gradient_norm']
    @property
    def tr_center_scaled(self):
        """

        Get scaled trust region center

        :return: scaled trust region center
        :rtype: list

        """
        if 'tr_center_scaled' in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['tr_center_scaled']
        raise Exception("\"tr_center_scaled\" not set in algorithm state")

    @property
    def min_parameter_bounds_scaled(self):
        """

        Get minimum parameter parameter bounds (scaled)

        :return: minimum parameter parameter bounds (scaled)
        :rtype: list

        """
        if 'min_param_bounds_scaled' in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['min_param_bounds_scaled']
        raise Exception("\"min_param_bounds_scaled\" not set in algorithm state")

    @property
    def max_parameter_bounds_scaled(self):
        """

        Get maximum parameter parameter bounds (scaled)

        :return: maximum parameter parameter bounds (scaled)
        :rtype: list

        """
        if 'max_param_bounds_scaled' in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['max_param_bounds_scaled']
        raise Exception("\"max_param_bounds_scaled\" not set in algorithm state")

    def set_tr_center_scaled(self,tr_center_scaled):
        """

        Set trust region center (scaled)

        :param tr_center_scaled: trust region center (scaled)
        :type tr_center_scaled: list

        """
        self.algorithm_parameters_dict['tr_center_scaled'] = tr_center_scaled

    def set_scaled_min_max_parameter_bounds(self,min_bounds_scaled,max_bounds_scaled):
        """

        Set minimum and maximum parameter bounds

        :param min_bounds_scaled: minimum parameter bounds
        :type min_bounds_scaled: list
        :param max_bounds_scaled: maximum parameter bounds
        :type max_bounds_scaled: list

        """
        self.algorithm_parameters_dict['min_param_bounds_scaled'] = min_bounds_scaled
        self.algorithm_parameters_dict['max_param_bounds_scaled'] = max_bounds_scaled

    def change_mc_ran(self,val:bool):
        """

        Set whether MC ran (for script run caller type)

        :param val: true if MC ran, false otherwise
        :type val: bool

        """
        self.config_dict['mc_ran'] = val

    def update_fidelity(self,new_fidelity):
        """

        Update fidelity (add run fidelity to current fidelity)

        :param new_fidelity: run fidelity to add
        :type new_fidelity: float

        """
        self.algorithm_parameters_dict['fidelity'] = new_fidelity

    def update_close_to_min_condition(self,value=False):
        """

        Update close to minimum condition condition

        :param value: close to minimum condition condition
        :type value: boolean

        """
        self.algorithm_parameters_dict['close_to_min_condition'] = value

    def update_proj_grad_norm(self,pgnorm):
        """

        Update norm of projected gradient

        :param pgnorm: norm of projected gradient
        :type pgnorm: float

        """
        self.algorithm_parameters_dict['proj_grad_norm'] = pgnorm

    @property
    def data_names(self):
        """

        Get names of the components of the objective function
        generated from the MC

        :return: names of the components of the objective function
            generated from the MC
        :rtype: list

        """
        if 'data_names' in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['data_names']
        raise Exception("\"data_names\" not set in algorithm state")

    def set_data_names(self, dn):
        """

        Set names of the components of the objective function
        generated from the MC

        :param dn: names of the components of the objective function
            generated from the MC
        :type dn: list

        """
        if 'data_names' not in self.algorithm_parameters_dict:
            self.algorithm_parameters_dict['data_names'] = dn

    @property
    def proj_grad_norm(self):
        """

        Get norm of the projected gradient

        :return: norm of the projected gradient
        :rtype: float

        """
        if 'proj_grad_norm' in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['proj_grad_norm']
        return 1.0

    @property
    def close_to_min_condition(self):
        """
        Get close to minimum condition

        :return: close to minimum condition
        :rtype: bool

        """
        if 'close_to_min_condition' in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['close_to_min_condition']
        return False

    @property
    def mc_ran(self):
        """

        Get whether MC ran (for script run caller type)

        :return: whether MC ran (for script run caller type)
        :rtype: bool

        """
        if 'mc_ran' in self.config_dict:
            return self.config_dict['mc_ran']
        return False

    @property
    def next_step(self):
        """

        Get next step of the optimization task

        :return: next step of the optimization task
        :rtype: str

        """
        if 'next_step' in self.config_dict:
            return self.config_dict['next_step']
        return "ops_start"

    @property
    def meta_data_file(self):
        """

        Get meta data file path

        :return: meta data file path
        :rtype: str

        """
        if 'meta_data_file' in self.config_dict:
            return self.config_dict['meta_data_file']
        return self.working_directory.get_log_path("parameter_metadata_1_k{}.json".format(self.k))

    @property
    def usefixedfidelity(self):
        """

        Get use fixed fidelity boolean

        :return: true if fixed fidelity to be used, false for multi fidelity run
        :rtype: bool

        """
        if 'usefixedfidelity' in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['usefixedfidelity']
        return True

    @property
    def fidelity(self):
        """

        Get current fidelity

        :return: current fidelity
        :rtype: int

        """
        return self.algorithm_parameters_dict['fidelity']

    @property
    def max_iterations(self):
        """

        Get maximum iteration value

        :return: maximum iteration value
        :rtype: int

        """
        return self.algorithm_parameters_dict['max_iteration']

    @property
    def max_fidelity_iteration(self):
        """

        Get iteration allowed at max fidelity

        :return: iteration allowed at max fidelity
        :rtype: int

        """
        return self.algorithm_parameters_dict['max_fidelity_iteration']

    @property
    def max_simulation_budget(self):
        """

        Get maximum simulation budget

        :return: maximum simulation budget
        :rtype: int

        """
        return self.algorithm_parameters_dict['max_simulation_budget']

    @property
    def max_fidelity(self):
        """

        Get maximum fidelity value

        :return: maximum fidelity value
        :rtype: int

        """
        return np.infty if self.usefixedfidelity else self.algorithm_parameters_dict['max_fidelity']

    @property
    def min_fidelity(self):
        """

        Get minimum fidelity value

        :return: minimum fidelity value
        :rtype: int

        """
        if self.usefixedfidelity: return self.fidelity
        if 'min_fidelity' in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['min_fidelity']
        return 50

    @property
    def kappa(self):
        """

        Get algorithrm parameter :math:`\kappa`

        :return: algorithrm parameter :math:`\kappa`
        :rtype: float

        """
        return self.algorithm_parameters_dict['kappa']

    @property
    def tr_radius(self):
        """

        Get trust region radius

        :return: trust region radius
        :rtype: float

        """
        return self.algorithm_parameters_dict['tr']['radius']

    @property
    def previous_tr_radius(self):
        """

        Get previous trust region radius

        :return: previous trust region radius
        :rtype: float

        """
        return self.algorithm_parameters_dict['tr']['previous_radius']

    @property
    def previous_tr_center(self):
        """

        Get previous trust region center

        :return: previous trust region center
        :rtype: list

        """
        return self.algorithm_parameters_dict['tr']['previous_center']

    def update_tr_radius(self,new_radius):
        """

        Update trust region radius

        :param new_radius: new trust region radius value
        :type new_radius: float

        """
        self.algorithm_parameters_dict['tr']['previous_radius'] = self.tr_radius
        self.algorithm_parameters_dict['tr']['radius'] = new_radius

    def update_tr_center(self,new_center):
        """

        Update trust region center

        :param new_center: new trust region center
        :type new_center: list

        """
        self.algorithm_parameters_dict['tr']['previous_center'] = self.tr_center
        self.algorithm_parameters_dict['tr']['center'] = new_center

    @property
    def tr_mu(self):
        """

        Get trust region parameter :math:`\mu`

        :return: trust region parameter :math:`\mu`
        :rtype: float

        """
        return self.algorithm_parameters_dict['tr']['mu']

    @property
    def N_p(self):
        """

        Get number of parameters to sample

        :return: number of parameters to sample
        :rtype: int

        """
        return self.algorithm_parameters_dict['N_p']

    @property
    def dim(self):
        """

        Get dimensionality of the problem

        :return: dimensionality of the problem
        :rtype: int

        """
        return self.algorithm_parameters_dict['dim']

    @property
    def tr_center(self):
        """

        Get trust region center

        :return: trust region center
        :rtype: list

        """
        return self.algorithm_parameters_dict['tr']['center']

    @property
    def min_param_bounds(self):
        """

        Get minimum parameter bounds

        :return: minimum parameter bounds
        :rtype: list

        """
        return np.array(self.algorithm_parameters_dict['param_bounds'])[:,0]

    @property
    def max_param_bounds(self):
        """

        Get maximum parameter bounds

        :return: maximum parameter bounds
        :rtype: list

        """
        return np.array(self.algorithm_parameters_dict['param_bounds'])[:,1]

    @property
    def k(self):
        """

        Get current iteration number

        :return: current iteration number
        :rtype: int

        """
        return self.algorithm_parameters_dict['current_iteration']

    def increment_k(self):
        """

        Increment current iteration number by 1

        """
        self.algorithm_parameters_dict['current_iteration'] += 1

    @property
    def theta(self):
        """

        Get algorithm parameter :math:`\theta`

        :return: algorithm parameter :math:`\theta`
        :rtype: float

        """
        return self.algorithm_parameters_dict['theta']

    @property
    def thetaprime(self):
        """

        Get algorithm parameter :math:`\theta'`

        :return: algorithm parameter :math:`\theta'`
        :rtype: float

        """
        return self.algorithm_parameters_dict['thetaprime']

    @property
    def param_names(self):
        """

        Get names of the parameter dimensions

        :return: names of the parameter dimensions
        :rtype:  list

        """
        return self.algorithm_parameters_dict['param_names']

    @property
    def output_level(self):
        """

        Get output level

        :return: output level
        :rtype: int

        """
        return self.algorithm_parameters_dict['output_level']

    @property
    def working_directory(self):
        """

        Get working directory path

        :return: working directory path
        :rtype: WorkingDirectory

        """
        return self.config_dict['working_directory']

    def get_model_function_handle(self,data_name):
        """

        Get surrogate model function handle

        :param data_name: names of the component of the objective function
            generated from the MC  for which the  surrogate model function handle is desired
        :type data_name: str
        :return: surrogate model function handle
        :rtype: function

        """
        if data_name in self.config_dict['model']['function']:
            return self.config_dict['model']['function'][data_name]
        return None

    @property
    def model_parameters(self):
        """

        Get model parameters

        :return: model parameters
        :rtype: dict

        """
        return self.config_dict['model']['parameters']

    @property
    def f_structure_function_handle(self):
        """

        Get function structure function handle

        :return: function structure function handle
        :rtype: function

        """
        return self.config_dict['f_structure']['function']

    @property
    def f_structure_parameters(self):
        """

        Get function structure parameters

        :return: function structure parameters
        :rtype: dict

        """
        return self.config_dict['f_structure']['parameters']

    def update_f_structure_model_parameters(self, model_scaling_key:str,ds:object):
        """

        Update function parametrers with scaled and unscaled model artifacts

        :param model_scaling_key: scaled or unscaled model key
        :type model_scaling_key: str
        :param ds: model artifacts
        :type ds: dict

        """
        self.config_dict['f_structure']['parameters'][model_scaling_key].update(ds)

    def update_f_structure_parameters(self, key,value):
        """

        Update function structure parameters. If the key does not exist, a new one is created
        and if the key exists, the value associated with the value is updated

        :param key: key to be added or updated
        :type key: str
        :param value: value to be added or updated
        :type value: Any

        """
        self.config_dict['f_structure']['parameters'][key] = value

    @property
    def mc_run_folder_path(self):
        """

        Get MC run folder path (that the MC run sees)

        :return: MC run folder path
        :rtype: str

        """
        return self.working_directory.get_log_path("MC_RUN")

    @property
    def mc_parameters(self):
        """

        Get MC parameters

        :return: MC parameters
        :rtype:  dict

        """
        if "parameters" in self.config_dict['mc']:
            return  self.config_dict['mc']['parameters']
        return None

    @property
    def mc_object(self):
        """

        Get MC object

        :return: MC object
        :rtype: maestro.mc.MCTask

        """
        return self.config_dict['mc']['object']

    @property
    def mc_ranks(self):
        """

        Get MC ranks

        :return:  number of ranks to use in script run
        :rtype: int

        """
        if 'ranks' in self.config_dict['mc']:
            return self.config_dict['mc']['ranks']
        else:
            from maestro.mpi4py_ import MPI_
            comm = MPI_.COMM_WORLD
            return comm.Get_size()

    @property
    def mc_caller_type(self):
        """

        Get MC caller type

        :return: MC caller type
        :rtype: str

        """
        return self.config_dict['mc']['caller_type']

    @property
    def mc_call_on_workflow(self):
        """

        Is the MC Task call on workflow?

        :return: true if MC Task call on workflow
        :rtype: bool

        """
        try:
            import pyhenson as h
            pyhenson_found = True
        except: pyhenson_found = False
        is_workflow =  self.mc_caller_type == "workflow"
        if is_workflow and not pyhenson_found:
            raise Exception("Cannot run workflow without pyhenson")
        return is_workflow

    @property
    def mc_call_using_script_run(self):
        """

        Is the MC Task call using script run?

        :return: true if MC Task call using script run
        :rtype: bool

        """
        return self.mc_caller_type == "script run"

    @property
    def mc_call_using_function_call(self):
        """

        Is the MC Task call using function call?

        :return: true if MC Task call using function call
        :rtype: bool

        """
        return self.mc_caller_type == "function call"

    @property
    def algorithm_status(self):
        """

        Get algorithm status object

        :return: algorithm status
        :rtype: maestro.AlgorithmStatus

        """
        if 'algorithm_status' in self.config_dict:
            return self.config_dict['algorithm_status']
        else:
            self.config_dict['algorithm_status'] = AlgorithmStatus(0)
            return self.config_dict['algorithm_status']

    def update_parameter_metadata(self,k,p_type,param_ds):
        """

        Update parameter metadata

        :param k: iteration number
        :type k: int
        :param p_type: parameter type
        :type p_type: str
        :param param_ds: parameter metadata
        :type param_ds: dict

        """
        k_str = "k{}".format(k)
        if k_str not in self.param_meta_data:
            self.param_meta_data[k_str] = {}
        self.param_meta_data[k_str][p_type] = param_ds

    def get_paramerter_metadata(self,k,p_type):
        """

        Get parameter metadata

        :param k: iteration number
        :type k: int
        :param p_type: parameter type
        :type p_type: str
        :return: parameter metadata
        :rtype: dict

        """
        k_str = "k{}".format(k)
        if k_str in self.param_meta_data and p_type in self.param_meta_data[k_str]:
            return self.param_meta_data[k_str][p_type]
        else:
            raise Exception("Parameter metadata {} not found in iteration {}".format(p_type,k))

    def copy_type_in_paramerter_metadata(self,k,curr_type_key,new_type_key):
        """

        Copy parameter metadata from one parameter type to a new parameter type

        :param k: iteration number
        :type k: int
        :param curr_type_key: current parameter type key name
        :type curr_type_key: str
        :param new_type_key: new  parameter type key name
        :type new_type_key: str

        """
        k_str = "k{}".format(k)
        meta_data = copy.deepcopy(self.param_meta_data[k_str][curr_type_key])
        self.param_meta_data[k_str][new_type_key] = meta_data

    def delete_data_at_index_from_paramerter_metadata(self,k,p_type,index):
        """

        Delete data at an index of the parameter type and iteration number
        from parameter metadata

        :param k: iteration number
        :type k: int
        :param p_type: parameter type
        :type p_type: str
        :param index: index of the entry to delete
        :type index: int

        """
        k_str = "k{}".format(k)
        for key in self.param_meta_data[k_str][p_type].keys():
            del self.param_meta_data[k_str][p_type][key][index]


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "------ algorithm parameters dictionary ------\n"
        s+=pprint.pformat(self.algorithm_parameters_dict)
        s += "\n\n------ config dictionary ------\n"
        s+=pprint.pformat(self.config_dict)
        s += "\n\n------ param metadata ------\n"
        s += pprint.pformat(self.param_meta_data)
        return s


class WorkingDirectory():
    """
    Working directory utility
    """
    def __init__(self,working_dir):
        """

        Initialize working directory utility object

        :param working_dir: working directory path
        :type working_dir: str

        """
        self.working_directory = working_dir

    def get_log_path(self,path):
        """

        Get log path within working directory

        :param path: path within log directory of the working directory
        :type path: str
        :return: path consisting of working directory, log and path within log directory
        :rtype: str

        """
        return os.path.join(self.working_directory,"log",path)

    def get_conf_path(self,path):
        """

        Get conf path within working directory

        :param path: path within conf directory of the working directory
        :type path: str
        :return: path consisting of working directory, conf and path within conf directory
        :rtype: str

        """
        return os.path.join(self.working_directory,"conf",path)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "working directory set in the object of type WorkingDirectory is {}".format(self.working_directory)

class AlgorithmStatus():
    """
    Algorithm status class
    """
    def __init__(self,status,msg=None,tr_radius_messaage=None,tr_center_messaage=None,tr_update_code=None):
        """

        Initialize algorithm status object

        :param status: algorithm status value
        :type status: int | dict
        :param msg: algorithm status message
        :type msg: str
        :param tr_radius_messaage: trust region status message
        :type tr_radius_messaage: str
        :param tr_center_messaage: trust region center message
        :type tr_center_messaage: str
        :param tr_update_code: trust region update code
        :type tr_update_code: str

        """
        if type(status)==dict:
            self.from_dict(status)
        elif type(status)==int:
            self.status = status
            self.failure_message = msg
            self.tr_radius_messaage = tr_radius_messaage
            self.tr_center_messaage = tr_center_messaage
            self.tr_update_code = tr_update_code

    def as_dict(self):
        """

        Get algorithm status object as dictionary

        :return: algorithm status object as dictionary
        :rtype: dict

        """
        return {
            "status":self.status,
            "failure_message":self.failure_message,
            "tr_radius_messaage":self.tr_radius_messaage,
            "tr_center_messaage" :self.tr_center_messaage,
            "tr_update_code":self.tr_update_code
        }
    def from_dict(self, ds):
        """

        Construct algorithm status from dictionry

        :param ds: from dictionary
        :type ds: dict

        """
        self.status = ds['status']
        self.failure_message = ds['failure_message']
        self.tr_radius_messaage = ds['tr_radius_messaage']
        self.tr_center_messaage = ds['tr_center_messaage']
        self.tr_update_code = ds['tr_update_code']

    def update_tr_status(self,tr_radius_messaage,tr_center_messaage,tr_update_code):
        """

        Update trust region status

        :param tr_radius_messaage: trust region radius status message
        :type tr_radius_messaage: str
        :param tr_center_messaage: trust region center status message
        :type tr_center_messaage: str
        :param tr_update_code: trust region update code
        :type tr_update_code: str

        """
        self.tr_radius_messaage = tr_radius_messaage
        self.tr_center_messaage = tr_center_messaage
        self.tr_update_code = tr_update_code

    def update_status(self,status,msg=""):
        """

        Update status

        :param status: new status
        :type status: int
        :param msg: status message
        :type msg: str

        """
        self.status = status
        if status == 8:
            self.failure_message = msg

    @staticmethod
    def get_status_dict():
        """

        Get algorithm status to definition mapping

        :return: algorithm status to definition mapping
        :rtype: dict

        """
        status_dict = {
            0:"Ok to continue on to next iteration",
            1:"Success: norm of the projected gradient is sufficiently small",
            2:"Max iterations reached",
            3:"Simulation budget depleted",
            4:"Failure: MC task was successful on less than 1 or N_p parameters (error)",
            5:"Trust region radius is an order of magnitude smaller than the radius at which max "
              "fidelity was reached",
            6:"Fidelity has been at a maximum value for the specified number of iterations",
            7:"The usable MC output was less than what was needed for constructing a model."
              "It is possible that too many parameters yielded MC output that was either nan or infty",
            8:"Failure: ",
            9:"Trust region radius is less than the specified minimum bound",
            10:"The function structure solution indicates that the current iterate is very similar to the "
               "previous iterate. This could happen because the algorithm is near a stationary point due "
               "to which it fails to move from the current iterate. Then all of the same parameters "
               "from the previous iteration got selected within the trust region of the current iteration. "
               "In this state, the solver cannot continue. Quitting now. "
               "Some suggestions to get around this problem include:\n"
               "\t1. Try increasing number of parameters \"N_p\"\n"
               "\t2. Try increasing \"fidelity\" parameter\n"
               "\t3. Try using multiple levels of fidelity by setting \"usefixedfidelity\" parameter to \"false\"\n"
               "A feature that allows the user to set the minimum percentage of new parameters required in any iteration "
               "is in the works and will be released in future versions of this project."


        }
        return status_dict

    @property
    def status_def(self):
        """

        Get algorithm status definition

        :return: algorithm status definition
        :rtype:
        """
        defn = AlgorithmStatus.get_status_dict()[self.status]
        if self.status == 8:
            defn += " {}".format(self.failure_message)
        return defn

    @property
    def status_val(self):
        """

        Get status value

        :return: status value
        :rtype: int
        """
        return self.status

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = ""
        s += "status: "+str(self.status)
        s += "; failure message: "
        s += self.failure_message if self.failure_message is not None else "none"
        s += "; tr center message: "
        s += self.tr_center_messaage  if self.tr_center_messaage is not None else "none"
        s += "; tr radius message: "
        s += self.tr_radius_messaage  if self.tr_radius_messaage is not None else "none"
        s += "; tr update code: "
        s += self.tr_update_code if self.tr_update_code is not None else "none"
        return s










