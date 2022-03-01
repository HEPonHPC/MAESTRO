import copy
import pprint

import os,sys
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.16f')
import numpy as np


class Settings(object):

    def __init__(self,working_dir=None, algorithm_parameters_file=None, config_file=None):
        self.algorithm_parameters_dict = {}
        self.config_dict = {}
        if algorithm_parameters_file is None and config_file is None:
            WD = WorkingDirectory(working_dir)
            algorithm_parameters_file = WD.get_log_path('algorithm_parameters_dump.json')
            config_file = WD.get_log_path('config_dump.json')

        self.initialize_algorithm_parameters(algorithm_parameters_file)
        self.initialize_config(config_file, working_dir)

    def read_setting_file(self, file):
        try:
            with open(file, 'r') as f:
                ds = json.load(f)
                return ds
        except:
            raise Exception("setting file type not supported")

    def write_setting_file(self, ds, file):
        with open(file, 'w') as f:
            json.dump(ds, f, indent=4)

    def initialize_algorithm_parameters(self, file:str):
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
            import mfstrodf.mc
            mc_class = getattr(mfstrodf.mc,ds['mc']['class_str'])
            self.config_dict['mc']['object'] = mc_class(self.mc_run_folder_path,self.mc_parameters)
        except:
            raise Exception("MC class \""+ds['mc']['class_str']+"\" not found in mfstrodf.mc")
        if self.mc_call_using_script_run and 'location_str' not in ds['mc']:
            raise Exception("MC script location needs to be set in config_dict.mc.location_str for script call")

        # self.config_dict['model']['parameters'] = ds['model']['parameters']

        from mfstrodf import ModelConstruction
        self.config_dict['model']['function'] = {}
        for data_name in ds['model']['function_str'].keys():
            try:
                method_to_call = getattr(ModelConstruction,ds['model']['function_str'][data_name])
                self.config_dict['model']['function'][data_name] = method_to_call
            except:
                raise Exception("Model function \""+ds['model']['function_str'][data_name]+"\" not found in mfstrodf.ModelConstruction")

        # self.config_dict['f_structure']['parameters'] = ds['f_structure']['parameters']
        if 'model' not in self.config_dict['f_structure']['parameters']:
            self.config_dict['f_structure']['parameters']['model'] = {}
        if 'model_scaled' not in self.config_dict['f_structure']['parameters']:
            self.config_dict['f_structure']['parameters']['model_scaled'] = {}
        # self.config_dict['f_structure']['parameters']['optimization'] = ds['f_structure']['parameters']['optimization']

        try:
            from mfstrodf import Fstructure
            method_to_call = getattr(Fstructure,ds['f_structure']['function_str'])
            self.config_dict['f_structure']['function'] = method_to_call
        except:
            raise Exception("Fstructure function \""+ds['f_structure']['function_str']+"\" not found in mfstrodf.Fstructure")

        if 'algorithm_status_dict' in self.config_dict:
            self.algorithm_status.from_dict(self.config_dict['algorithm_status_dict'])
        else:
            self.algorithm_status.update_status(0)

    def save(self, meta_data_file=None,next_step=None, to_log = False):
        if to_log:
            algorithm_parameters_file = self.working_directory.get_log_path('algorithm_parameters_dump_k{}.json'.format(self.k))
            config_file = self.working_directory.get_log_path('config_dump_k{}.json'.format(self.k))
        else:
            algorithm_parameters_file = self.working_directory.get_log_path('algorithm_parameters_dump.json')
            config_file = self.working_directory.get_log_path('config_dump.json')

        if next_step is not None:
            self.config_dict['next_step'] = next_step
        if meta_data_file is not None:
            self.config_dict['meta_data_file'] = meta_data_file

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

    def update_simulation_budget_used(self, simulation_count):
        self.algorithm_parameters_dict['simulation_budget_used'] += simulation_count

    @property
    def simulation_budget_used(self):
        return self.algorithm_parameters_dict['simulation_budget_used']

    @property
    def no_iters_at_max_fidelity(self):
        if "no_iters_at_max_fidelity" in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['no_iters_at_max_fidelity']
        return 0
    def increment_no_iters_at_max_fidelity(self):
        if "no_iters_at_max_fidelity" in self.algorithm_parameters_dict:
            self.algorithm_parameters_dict['no_iters_at_max_fidelity']+=1
        else:
            self.algorithm_parameters_dict['no_iters_at_max_fidelity'] = 1

    @property
    def radius_at_which_max_fidelity_reached(self):
        if "radius_at_which_max_fidelity_reached" in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['radius_at_which_max_fidelity_reached']
        return None

    def set_radius_at_which_max_fidelity_reached(self,tr_radius):
        if not "radius_at_which_max_fidelity_reached" in self.algorithm_parameters_dict:
            if self.fidelity>=self.max_fidelity:
                self.algorithm_parameters_dict['radius_at_which_max_fidelity_reached'] = tr_radius

    @property
    def optimization_parameters(self):
        return self.config_dict['f_structure']['parameters']['optimization']

    @property
    def tr_eta(self):
        return self.algorithm_parameters_dict['tr']['eta']

    @property
    def tr_min_radius(self):
        return self.algorithm_parameters_dict['tr']['min_radius']

    @property
    def tr_max_radius(self):
        return self.algorithm_parameters_dict['tr']['max_radius']

    @property
    def min_gradient_norm(self):
        return self.algorithm_parameters_dict['min_gradient_norm']
    @property
    def tr_center_scaled(self):
        if 'tr_center_scaled' in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['tr_center_scaled']
        raise Exception("\"tr_center_scaled\" not set in algorithm state")

    @property
    def min_parameter_bounds_scaled(self):
        if 'min_param_bounds_scaled' in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['min_param_bounds_scaled']
        raise Exception("\"min_param_bounds_scaled\" not set in algorithm state")

    @property
    def max_parameter_bounds_scaled(self):
        if 'max_param_bounds_scaled' in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['max_param_bounds_scaled']
        raise Exception("\"max_param_bounds_scaled\" not set in algorithm state")

    def set_tr_center_scaled(self,tr_center_scaled):
        self.algorithm_parameters_dict['tr_center_scaled'] = tr_center_scaled

    def set_scaled_min_max_parameter_bounds(self,min_bounds_scaled,max_bounds_scaled):
        self.algorithm_parameters_dict['min_param_bounds_scaled'] = min_bounds_scaled
        self.algorithm_parameters_dict['max_param_bounds_scaled'] = max_bounds_scaled

    def change_mc_ran(self,val:bool):
        self.config_dict['mc_ran'] = val

    def update_fidelity(self,new_fidelity):
        self.algorithm_parameters_dict['fidelity'] = new_fidelity

    def update_close_to_min_condition(self,value=False):
        self.algorithm_parameters_dict['close_to_min_condition'] = value

    def update_proj_grad_norm(self,pgnorm):
        self.algorithm_parameters_dict['proj_grad_norm'] = pgnorm

    @property
    def data_names(self):
        if 'data_names' in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['data_names']
        raise Exception("\"data_names\" not set in algorithm state")

    def set_data_names(self, dn):
        if 'data_names' not in self.algorithm_parameters_dict:
            self.algorithm_parameters_dict['data_names'] = dn

    @property
    def proj_grad_norm(self):
        if 'proj_grad_norm' in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['proj_grad_norm']
        return 1.0

    @property
    def close_to_min_condition(self):
        if 'close_to_min_condition' in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['close_to_min_condition']
        return False

    @property
    def mc_ran(self):
        if 'mc_ran' in self.config_dict:
            return self.config_dict['mc_ran']
        return False

    @property
    def next_step(self):
        if 'next_step' in self.config_dict:
            return self.config_dict['next_step']
        return "ops_start"

    @property
    def meta_data_file(self):
        if 'meta_data_file' in self.config_dict:
            return self.config_dict['meta_data_file']
        return self.working_directory.get_log_path("parameter_metadata_1_k{}.json".format(self.k))

    @property
    def usefixedfidelity(self):
        if 'usefixedfidelity' in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['usefixedfidelity']
        return True

    @property
    def fidelity(self):
        return self.algorithm_parameters_dict['fidelity']

    @property
    def max_iterations(self):
        return self.algorithm_parameters_dict['max_iteration']

    @property
    def max_fidelity_iteration(self):
        return self.algorithm_parameters_dict['max_fidelity_iteration']

    @property
    def max_simulation_budget(self):
        return self.algorithm_parameters_dict['max_simulation_budget']

    @property
    def max_fidelity(self):
        return np.infty if self.usefixedfidelity else self.algorithm_parameters_dict['max_fidelity']

    @property
    def min_fidelity(self):
        if self.usefixedfidelity: return self.fidelity
        if 'min_fidelity' in self.algorithm_parameters_dict:
            return self.algorithm_parameters_dict['min_fidelity']
        return 50

    @property
    def kappa(self):
        return self.algorithm_parameters_dict['kappa']

    @property
    def tr_radius(self):
        return self.algorithm_parameters_dict['tr']['radius']

    @property
    def previous_tr_radius(self):
        return self.algorithm_parameters_dict['tr']['previous_radius']

    @property
    def previous_tr_center(self):
        return self.algorithm_parameters_dict['tr']['previous_center']

    def update_tr_radius(self,new_radius):
        self.algorithm_parameters_dict['tr']['previous_radius'] = self.tr_radius
        self.algorithm_parameters_dict['tr']['radius'] = new_radius

    def update_tr_center(self,new_center):
        self.algorithm_parameters_dict['tr']['previous_center'] = self.tr_center
        self.algorithm_parameters_dict['tr']['center'] = new_center

    @property
    def tr_mu(self):
        return self.algorithm_parameters_dict['tr']['mu']

    @property
    def N_p(self):
        return self.algorithm_parameters_dict['N_p']

    @property
    def dim(self):
        return self.algorithm_parameters_dict['dim']

    @property
    def tr_center(self):
        return self.algorithm_parameters_dict['tr']['center']

    @property
    def min_param_bounds(self):
        return np.array(self.algorithm_parameters_dict['param_bounds'])[:,0]

    @property
    def max_param_bounds(self):
        return np.array(self.algorithm_parameters_dict['param_bounds'])[:,1]

    @property
    def k(self):
        return self.algorithm_parameters_dict['current_iteration']

    def increment_k(self):
        self.algorithm_parameters_dict['current_iteration'] += 1

    @property
    def theta(self):
        return self.algorithm_parameters_dict['theta']

    @property
    def thetaprime(self):
        return self.algorithm_parameters_dict['thetaprime']

    @property
    def param_names(self):
        return self.algorithm_parameters_dict['param_names']

    @property
    def output_level(self):
        return self.algorithm_parameters_dict['output_level']

    @property
    def working_directory(self):
        return self.config_dict['working_directory']

    def get_model_function_handle(self,data_name):
        if data_name in self.config_dict['model']['function']:
            return self.config_dict['model']['function'][data_name]
        return None

    @property
    def model_parameters(self):
        return self.config_dict['model']['parameters']

    @property
    def f_structure_function_handle(self):
        return self.config_dict['f_structure']['function']

    @property
    def f_structure_parameters(self):
        return self.config_dict['f_structure']['parameters']

    def update_f_structure_model_parameters(self, model_scaling_key:str,ds:object):
        self.config_dict['f_structure']['parameters'][model_scaling_key].update(ds)

    def update_f_structure_parameters(self, key,value):
        self.config_dict['f_structure']['parameters'][key] = value

    @property
    def mc_run_folder_path(self):
        return self.working_directory.get_log_path("MC_RUN")

    @property
    def mc_parameters(self):
        if "parameters" in self.config_dict['mc']:
            return  self.config_dict['mc']['parameters']
        return None

    @property
    def mc_object(self):
        return self.config_dict['mc']['object']

    @property
    def mc_ranks(self):
        if 'ranks' in self.config_dict['mc']:
            return self.config_dict['mc']['ranks']
        else:
            from mfstrodf.mpi4py_ import MPI_
            comm = MPI_.COMM_WORLD
            return comm.Get_size()

    @property
    def mc_caller_type(self):
        return self.config_dict['mc']['caller_type']

    @property
    def mc_call_on_workflow(self):
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
        return self.mc_caller_type == "script run"

    @property
    def mc_call_using_function_call(self):
        return self.mc_caller_type == "function call"

    @property
    def algorithm_status(self):
        if 'algorithm_status' in self.config_dict:
            return self.config_dict['algorithm_status']
        else:
            self.config_dict['algorithm_status'] = AlgorithmStatus(0)
            return self.config_dict['algorithm_status']

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "------ algorithm parameters dictionary ------\n"
        s+=pprint.pformat(self.algorithm_parameters_dict)
        s += "\n\n------ config dictionary ------\n"
        s+=pprint.pformat(self.config_dict)
        return s


class WorkingDirectory():
    def __init__(self,working_dir):
        self.working_directory = working_dir

    def get_log_path(self,path):
        return os.path.join(self.working_directory,"log",path)

    def get_conf_path(self,path):
        return os.path.join(self.working_directory,"conf",path)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "working directory set in the object of type WorkingDirectory is {}".format(self.working_directory)

class AlgorithmStatus():
    def __init__(self,status,msg=None,tr_radius_messaage=None,tr_center_messaage=None,tr_update_code=None):
        if type(status)==dict:
            self.from_dict(status)
        elif type(status)==int:
            self.status = status
            self.failure_message = msg
            self.tr_radius_messaage = tr_radius_messaage
            self.tr_center_messaage = tr_center_messaage
            self.tr_update_code = tr_update_code

    def as_dict(self):
        return {
            "status":self.status,
            "failure_message":self.failure_message,
            "tr_radius_messaage":self.tr_radius_messaage,
            "tr_center_messaage" :self.tr_center_messaage,
            "tr_update_code":self.tr_update_code
        }
    def from_dict(self, ds):
        self.status = ds['status']
        self.failure_message = ds['failure_message']
        self.tr_radius_messaage = ds['tr_radius_messaage']
        self.tr_center_messaage = ds['tr_center_messaage']
        self.tr_update_code = ds['tr_update_code']

    def update_tr_status(self,tr_radius_messaage,tr_center_messaage,tr_update_code):
        self.tr_radius_messaage = tr_radius_messaage
        self.tr_center_messaage = tr_center_messaage
        self.tr_update_code = tr_update_code

    def update_status(self,status,msg=""):
        self.status = status
        if status == 8:
            self.failure_message = msg

    @staticmethod
    def get_status_dict():
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
        defn = AlgorithmStatus.get_status_dict()[self.status]
        if self.status == 8:
            defn += " {}".format(self.failure_message)
        return defn

    @property
    def status_val(self):
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










