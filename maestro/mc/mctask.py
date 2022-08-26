import sys

import pprint

import re,glob,os
import numpy as np
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.16f')
import math
from maestro import DiskUtil
class MCTask(object):
    """
    MC Task base class
    """
    def __init__(self,mc_working_directory, mc_parameters=None):
        """

        Create an MC Task object

        :param mc_working_directory: MC working directory path
        :type mc_working_directory: str
        :param mc_parameters: MC parameters set in the configuration input
        :type mc_parameters: dict

        """
        self.mc_run_folder = mc_working_directory
        self.mc_parmeters = mc_parameters

    # todo === add to doc
    # todo === can be called from __main__ or directly on object (should work as a blackbox -- like a task in the workflow)
    def run_mc(self):
        """

        Run the MC (abstract). This function needs to be implemented in a class that
        inherits this class

        """
        raise Exception("This function must be implemented in the derived class")

    # todo === add to doc
    # todo === return max sigma. If sigma cannot be found return None
    def merge_statistics_and_get_max_sigma(self):
        """

        Merge MC output statistics and find the maximum standard deviation of the
        MC output (abstract). This function needs to be implemented in a class that
        inherits this class

        :return: class that inherits this class should return the
            maximum standard deviation of the MC output
        :rtype: float

        """
        raise Exception("This function must be implemented in the derived class")

    # todo === add to doc
    # todo === return df and additional_data object (additional_data can be none)
    def convert_mc_output_to_df(self, all_param_directory):
        """

        Convert MC output to a pandas dataframe (abstract). This function needs to be implemented in a class that
        inherits this class

        :Example:
            Example of the returned pandas dataframe is given below.
            In the example below, there are three parameters and two terms of the objective function.
            Terms that end with ``.P`` are the parameters and those ending with ``.V`` are the values
            associated with either the ``MC``, i.e., MC sample values  or the ``DMC``, i.e., MC standard deviation
            values. You can add more rows, i.e, more sets of parameter and values  for additional terms in the objective function
            or more columns, i.e., more components of the each term of the objective that
            come from the MC simulator::

                >df
                                MC                      DMC
                Term1.P        [[1,2],[3,4],[6,3]]     [[1,2],[3,4],[6,3]]
                Term1.V        [19, 18, 17]            [99, 98, 97]
                Term2.P        [[1,2],[3,4],[6,3]]     [[1,2],[3,4],[6,3]]
                Term2.V        [29, 28, 27]            [89, 88, 87]

        :param all_param_directory: MC outout directory path
        :type all_param_directory: str
        :return: pandas dataframe formatted MC output
        :rtype: pandas.DataFrame

        """
        raise Exception("This function must be implemented in the derived class")

    def check_and_resolve_nan_inf(self,data, all_param_directory,term_names,group_names=None):
        """

        Check for NaNs/:math:`\infty` and try to resolve them.

        * If ``group_names`` is not None i.e., groups are specified, then try to interpolate NaNs/:math:`\infty`
          for terms within each group for each parameter
        * If ``group_names`` is None i.e., groups are not specified or if interpolation does not completely resolve
          the issue, then:

            * If the NaNs/:math:`\infty` was encountered in the start parameter,
              then raise an exception
            * If the NaNs/:math:`\infty` was encountered in the iterate (subproblem ``argmin``),
              then use the surrogate models from the current iteration to fill in the missing
              values
            * If the NaNs/:math:`\infty` was encountered in the adaptive sample, then
              drop the parameter from the corresponding term

        Resolved NaNs/:math:`\infty` are overwritten in the ``data`` argument (as ``data`` is passed
        as a reference to this function).

        :Example:
            Example of the ``data`` passed in as argument to this function is shown below.
            In the example below, there are three parameters and two terms of the objective function.
            Terms that end with ``.P`` are the parameters and those ending with ``.V`` are the values
            associated with either the ``MC``, i.e., MC sample values  or the ``DMC``, i.e., MC standard deviation
            values::

                >data

                {
                    "MC":{
                            "Term1.P":[[1,2],[3,4],[6,3]],
                            "Term1.V": [19, 18, 17],
                            "Term2.P":[[1,2],[3,4],[6,3]],
                            "Term2.V": [29, 28, 27]
                        },
                    "DMC":{
                            "Term1.P":[[1,2],[3,4],[6,3]],
                            "Term1.V": [99, 98, 97],
                            "Term2.P":[[1,2],[3,4],[6,3]],
                            "Term2.V": [89, 88, 87]
                        }
                }

        :param data: MC output data formatted into a dictionary
        :type data: dict
        :param all_param_directory: MC outout directory path
        :type all_param_directory: str
        :param term_names: term names e.g., [Term1, Term2]
        :type term_names: list
        :param group_names: groups of terms (if any) (default: None).
            If this argument is None, then this function will not try to
            interpolate missing values between terms
        :type group_names: list

        """
        self.did_middle = False
        self.did_left = False
        self.did_right = False
        self.did_model_eval = False
        self.did_discard = False
        def interpolate_nan_inf(data_array):
            resolved = True
            if np.isnan(data_array).all() or np.isinf(data_array).all():
                resolved = False
            elif np.isnan(data_array).any() or np.isinf(data_array).any():
                nan_inf = [i or j for (i,j) in zip(np.isnan(data_array), np.isinf(data_array))]
                # Do (any) middle
                not_ni_left = None
                ni_count = 0
                for ninum, ni in enumerate(nan_inf):
                    if not ni and ni_count == 0:
                        not_ni_left = data_array[ninum]
                    elif not ni and ni_count > 0:
                        if not_ni_left is not None:
                            increment = (data_array[ninum]-not_ni_left)/(ni_count+1)
                            for i in range(ninum-ni_count,ninum):
                                data_array[i] = data_array[i-1]+increment
                            ni_count = 0
                            not_ni_left = data_array[ninum]
                            self.did_middle = True
                        else:
                            not_ni_left = data_array[ninum]
                            ni_count = 0
                    else: ni_count+=1

                # Do (any) left
                nan_inf = [i or j for (i,j) in zip(np.isnan(data_array), np.isinf(data_array))]
                ni_count = 0
                for ninum, ni in enumerate(nan_inf):
                    if ni: ni_count+=1
                    elif ni_count>0:
                        if ninum < len(nan_inf)-1 and not nan_inf[ninum] and not nan_inf[ninum+1]:
                            difference = data_array[ninum+1] - data_array[ninum]
                            for i in range(ninum-1,-1,-1):
                                data_array[i] = max(data_array[i+1] - difference,0.0)
                            self.did_left = True
                        else:resolved = False
                        break

                # Do (any) right
                nan_inf = [i or j for (i,j) in zip(np.isnan(data_array), np.isinf(data_array))]
                ni_count = 0
                for ninum in range(len(nan_inf)-1,-1,-1):
                    ni = nan_inf[ninum]
                    if ni: ni_count+=1
                    elif ni_count>0:
                        if ninum > 0 and not nan_inf[ninum] and not nan_inf[ninum-1]:
                            difference = data_array[ninum] - data_array[ninum-1]
                            for i in range(ninum+1,len(nan_inf)):
                                data_array[i] = data_array[i-1] + difference
                            self.did_right = True
                        else:resolved = False
                        break

            return (data_array, resolved)

        def use_model_to_resolve_nan_inf_or_drop():
            import json
            from maestro.model import ModelConstruction
            apd_arr = os.path.basename(all_param_directory).split('_')
            log_dir = os.path.normpath(self.mc_run_folder + os.sep + os.pardir)
            # read algorithm_parameters_dump.json
            with open(os.path.join(log_dir,'algorithm_parameters_dump.json')) as f:
                algo_ds = json.load(f)
            k = algo_ds['current_iteration']
            with open(os.path.join(log_dir,'config_dump.json')) as f:
                config_ds = json.load(f)
            function_str_dict = config_ds['model']['function_str']
            for tname in term_names:
                for dno,dname in enumerate(data_names):
                    X_ = data[dname]["{}.P".format(tname)]
                    Y_ = data[dname]["{}.V".format(tname)]
                    if np.isnan(Y_).any() or np.isinf(Y_).any():
                        nan_inf = [i or j for (i,j) in zip(np.isnan(Y_), np.isinf(Y_))]
                        #   If only one parameter
                        if apd_arr[2] != 'Np':
                            if apd_arr[3] == 'k0':
                                raise Exception("Nan and Inf found at the start point. Algorithm cannot "
                                                "continue. Please select a new start point in \"tr_center\" and "
                                                "try again")
                            # If k > 0: read model of the corresponding bin and evaluate model at x where y is nan/inf
                            else:
                                with open(os.path.join(log_dir,"{}_model_k{}.json".format(dname,k))) as f:
                                    model_ds = json.load(f)
                                for ninum,ni in enumerate(nan_inf):
                                    if ni:
                                        x = X_[ninum]
                                        if tname not in model_ds and tname+"#1" in model_ds:
                                            model = ModelConstruction.get_model_object(function_str_dict[dname],model_ds[tname+"#1"])
                                        else: model = ModelConstruction.get_model_object(function_str_dict[dname],model_ds[tname])
                                        Y_[ninum] = model(x)
                                        self.did_model_eval = True
                    #   If more than one parameter:
                    if apd_arr[2] == 'Np':
                        if np.isnan(Y_).any() or np.isinf(Y_).any():
                            nan_inf = [i or j for (i,j) in zip(np.isnan(Y_), np.isinf(Y_))]
                            for dno_inner,dname_inner in enumerate(data_names):
                                X__ = np.array(data[dname_inner]["{}.P".format(tname)])
                                Y__ = np.array(data[dname_inner]["{}.V".format(tname)])
                                data[dname_inner]["{}.P".format(tname)] = X__[np.invert(nan_inf)].tolist()
                                data[dname_inner]["{}.V".format(tname)] = Y__[np.invert(nan_inf)].tolist()
                            self.did_discard = True

        # Try to interpolate
        # If not possible to interpolate:
        #   If only one parameter:
        #       find out the iteration number k of the run
        #       If k == 0: raise exception
        #       If k > 0: read model of the corresponding bin and evaluate model at x where y is nan/inf
        #   If more than one parameter:
        #       delete x and y where y is a nan/inf
        term_names = np.array(term_names)
        data_names = data.keys()
        if group_names is not None:
            resolved = True
            for gnum, gname in enumerate(group_names):
                # Find terms in the group
                terms_in_group = np.sort(term_names[np.flatnonzero(np.core.defchararray.find(term_names,gname)!=-1)])
                for pnum,param in enumerate(data['MC']["{}.P".format(terms_in_group[0])]):
                    for dnum,dname in enumerate(data_names):
                        Y = [data[dname]["{}.V".format(term)][pnum] for term in terms_in_group]
                        # Try to interpolate
                        (Y,Y_resolved) = interpolate_nan_inf(Y)
                        if Y_resolved:
                            for tno,term in enumerate(terms_in_group):
                                data[dname]["{}.V".format(term)][pnum] = Y[tno]
                        # Remember the results of interpolation
                        resolved = resolved and Y_resolved
        else: resolved = False
        if not resolved:
            use_model_to_resolve_nan_inf_or_drop()
        nan_inf_status_code = ""
        nan_inf_status_code += 'L' if self.did_left else '-'
        nan_inf_status_code += 'M' if self.did_middle else '-'
        nan_inf_status_code += 'R' if self.did_right else '-'
        nan_inf_status_code += 'E' if self.did_model_eval else '-'
        nan_inf_status_code += 'D' if self.did_discard else '-'

        print_nan_inf_status_code = self.mc_parmeters['print_nan_inf_status_code'] \
            if 'print_nan_inf_status_code' in self.mc_parmeters else False
        from maestro import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        if print_nan_inf_status_code and rank == 0:
            print("NaN/Inf Status Code: {}".format(nan_inf_status_code))
            sys.stdout.flush()

    @staticmethod
    def read_params_file(path):
        """

        Read parameters from parameters file.

        :Example:
            Example of the format of parameter file on the file path ``path`` is given below. The first
            column is the parameter name and the second column is the value associated with
            the parameter::

                dimension1 0.2
                dimension2 0.4

        :param path: parameter file path
        :type path: str
        :return: list of parameter values in the same order as found in the parameter file
        :rtype: list

        """
        parameters = []
        with open(path, "r") as f:
            L = [l.strip() for l in f if not l.startswith("#")]
            for num, line in enumerate(L):
                parts = line.split()
                if len(parts) == 2:
                    parameters.append(float(parts[1]))
                elif len(parts) == 1:
                    parameters.append(float(parts[0]))
                else:
                    raise Exception("Error in parameter input format")
        return parameters

    def get_param_from_directory(self,param_directory,fnamep="params.dat"):
        """

        Get all parameters from parameter file with name fnamep in directory
        param_directory

        :Example:
            Example of the format of parameter file with file name ``fnamep`` is given below. The first
            column is the parameter name and the second column is the value associated with
            the parameter::

                dimension1 0.2
                dimension2 0.4


        :param param_directory: path of the parameter directory
        :type param_directory: str
        :param fnamep: name of the parameter file
        :type fnamep: str
        :return: list of parameter values in the same order as found in the parameter file
        :rtype: list

        """
        from maestro import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        re_pfname = re.compile(fnamep) if fnamep else None
        param = None
        if rank == 0:
            files = glob.glob(os.path.join(param_directory, "*"))
            for f in files:
                if re_pfname and re_pfname.search(os.path.basename(f)):
                    param = MCTask.read_params_file(f)
                    break
        param = comm.bcast(param,root=0)
        if param is None:
            raise Exception("Something went wrong. Cannot get parameter")
        return param

    def get_param_from_metadata(self, param_metadata):
        """
        Get parameters from the metadata object

        :param param_metadata:  parameter metadata object
        :type param_metadata: dict
        :return: list of parameter values
        :rtype: list

        """
        return param_metadata['parameters']

    def write_param(self, parameters, parameter_names, at_fidelities, run_fidelities,
                    mc_run_folder, expected_folder_name,
                    fnamep="params.dat", fnamerf="run_fidelity.dat",
                    fnameaf="at_fidelity.dat"):
        """

        Write parameters to parameter directory and generate parameter metadata

        :param parameters: list of parameters points
        :type parameters: list of lists
        :param parameter_names: names of the parameter dimensions
        :type parameter_names: list
        :param at_fidelities: current fidelity of the parameters
        :type at_fidelities: list
        :param run_fidelities: expected fidelity of the parameters i.e., these are the
            fidelities at which the MC should be run at the corresponding parameters
        :type run_fidelities: list
        :param mc_run_folder: MC run folder path
        :type mc_run_folder: str
        :param expected_folder_name: expected MC run folder path with the
            type of run (single or sample) and iteration number
        :type expected_folder_name: str
        :param fnamep: name of the parameter file (default: params.dat)
        :type fnamep: str
        :param fnamerf: name of the run fidelity file (default: run_fidelity.dat)
        :type fnamerf: str
        :param fnameaf: name of the at fidelity file (default: at_fidelity.dat)
        :type fnameaf: str
        :return: parameter metadata object
        :rtype: dict

        """
        from maestro import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            DiskUtil.remove_directory(mc_run_folder)
            os.makedirs(mc_run_folder,exist_ok=True)
            param_dir = []
            mc_param_dir = []
            for num, (p, r_fid,a_fid) in enumerate(zip(parameters, run_fidelities,at_fidelities)):
                npad = "{}".format(num).zfill(1+int(np.ceil(np.log10(len(parameters)))))
                outd_mc_run_folder = os.path.join(mc_run_folder, npad)
                outd_expected_folder = os.path.join(expected_folder_name, npad)
                mc_param_dir.append(outd_mc_run_folder)
                param_dir.append(outd_expected_folder)
                os.makedirs(outd_mc_run_folder,exist_ok=True)
                outfparams = os.path.join(outd_mc_run_folder, fnamep)
                with open(outfparams, "w") as pf:
                    for k, v in zip(parameter_names, p):
                        pf.write("{name} {val:.16e}\n".format(name=k, val=v))
                outffidelities = os.path.join(outd_mc_run_folder, fnamerf)
                with open(outffidelities, "w") as ff:
                    ff.write("{}".format(r_fid))
                outffidelities = os.path.join(outd_mc_run_folder, fnameaf)
                with open(outffidelities, "w") as ff:
                    ff.write("{}".format(a_fid))
            ds = {
                "parameters": parameters,
                "at fidelity": at_fidelities,
                "run fidelity": run_fidelities,
                "param directory": param_dir,
                "mc param directory":mc_param_dir
            }
            return ds
            # with open(file,'w') as f:
            #     json.dump(ds, f, indent=4)

    def set_current_iterate_as_next_iterate(self,
                                            current_iterate_parameter_data,
                                            next_iterate_parameter_data=None,
                                            next_iterate_mc_directory=None
                                            ):
        """
        Copy current iterate as the next iterate

        :param current_iterate_parameter_data: parameter metadata of the current iterate (from the current
            iteration
        :type current_iterate_parameter_data: dict
        :param next_iterate_parameter_data: parameter metadata of the next iterate (from the next
            iteration (default: None)
        :type next_iterate_parameter_data: dict
        :param next_iterate_mc_directory: MC directory path of the next iterate (default: None)
        :type next_iterate_mc_directory: str
        :return: updated current iterate parameter metadata
        :rtype: dict
        """

        from maestro import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        comm.barrier()
        curr_mc_dir_name = os.path.join(os.path.dirname(current_iterate_parameter_data['param directory'][0]),
                                        '__' + os.path.basename(current_iterate_parameter_data['param directory'][0]))
        if next_iterate_parameter_data is not None:
            new_mc_dir_name = next_iterate_parameter_data['param directory'][0]
            current_iterate_parameter_data['param directory'] = next_iterate_parameter_data['param directory']
        else:
            nex_iterate_basename = os.path.basename(current_iterate_parameter_data['param directory'][0])
            if nex_iterate_basename.startswith('__'):nex_iterate_basename = nex_iterate_basename[2:]
            new_mc_dir_name = os.path.join(next_iterate_mc_directory, nex_iterate_basename)
            current_iterate_parameter_data['param directory'] = [new_mc_dir_name]

        if rank == 0:
            DiskUtil.copyanything(curr_mc_dir_name, new_mc_dir_name)

        return current_iterate_parameter_data


    def update_current_fidelities(self,param_metadata):
        """

        Update current fidelity in parameter metadata of all parameters
        as the sum of current fidelity and run fidelity

        :param param_metadata: parameter metadata
        :type param_metadata: dict
        :return: list of updated fidelities
        :rtype: list

        """
        new_at_fid = [i+j for (i,j) in zip(param_metadata['at fidelity'], param_metadata['run fidelity'])]
        param_metadata['at fidelity'] = new_at_fid
        return new_at_fid

    def get_updated_current_fidelities(self,param_metadata):
        """

        Get updated current fidelities after adding the run fidelity

        :param param_metadata: parameter metadata
        :type param_metadata: dict
        :return: list of updated fidelities
        :rtype: list

        """
        return self.update_current_fidelities(param_metadata)

    def get_current_fidelities(self,param_metadata):
        """

        Get current fidelity from parameter metadata

        :param param_metadata: parameter metadata
        :type param_metadata: dict
        :return: list of current fidelities
        :rtype: list

        """
        return param_metadata['at fidelity']

    def get_parameter_data_from_metadata(self,param_metadata,param_index,include_parameter = False):
        """

        Get parameter data at a particular index in the metadata

        :param param_metadata: parameter metadata of all parameters
        :type param_metadata: dict
        :param param_index: index of the parameter of interest
        :type param_index: int
        :param include_parameter: true if parameter value at index should also to be included (default: False)
        :type include_parameter: bool
        :return: parameter metadata at the index of interest
        :rtype: dict

        """
        ds = {
            'at fidelity': param_metadata['at fidelity'][param_index],
            'run fidelity': param_metadata['run fidelity'][param_index],
            'param directory':param_metadata['param directory'][param_index],
            'mc param directory':param_metadata['mc param directory'][param_index]
        }
        if include_parameter:
            ds['parameters'] = param_metadata['parameters'][param_index]
        return ds

    def get_fidelity_from_directory(self,param_directory,fnamef="run_fidelity.dat"):
        """

        Get fidelity (current or run fidelity) from a parameter directory

        :param param_directory: path of the parameter directory
        :type param_directory: str
        :param fnamef: path of the fidelity file (current or run fidelity) (default: run_fidelity.dat)
        :type fnamef: str
        :return: fidelity value
        :rtype: int

        """
        re_fnamerf = re.compile(fnamef) if fnamef else None
        from maestro import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        files = None
        if rank == 0:
            files = glob.glob(os.path.join(param_directory, "*"))
        files = comm.bcast(files, root=0)
        fid = None
        from maestro import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            for file in files:
                if re_fnamerf and re_fnamerf.search(os.path.basename(file)):
                    with open(file) as f:
                        fid = int(float(next(f)))
        fid = comm.bcast(fid, root=0)
        if fid is None:
            if rank == 0:
                raise Exception("Something went wrong. Cannot get fidelity")
            else: exit(8)
        return fid

    def get_param_directory_array(self,all_param_directory):
        """

        From the MC run directory, get a list of parameter directory

        :Example:
            If <path>/MC_RUN is the MC run directory (``all_param_directory``), and it has the the following contents::

                <path>/MC_RUN/00
                <path>/MC_RUN/01
                <path>/MC_RUN/02

            Then this function return::

                [
                    <path>/MC_RUN/00,
                    <path>/MC_RUN/01,
                    <path>/MC_RUN/02
                ]

        :param all_param_directory: MC run directory path
        :type all_param_directory: str
        :return: paths of sub directories within the MC run directory
        :rtype: list

        """
        from maestro import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        newINDIRSLIST = None
        if rank == 0:
            newINDIRSLIST = glob.glob(os.path.join(all_param_directory, "*"))
            newINDIRSLIST = newINDIRSLIST if len(newINDIRSLIST)==1 else \
                sorted(newINDIRSLIST, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        newINDIRSLIST = comm.bcast(newINDIRSLIST, root=0)
        return newINDIRSLIST

    def write_fidelity_to_metadata_and_directory(self,param_metadata,fidelities,metadata_file_key='run fidelity',
                                                 fnamef="run_fidelity.dat"):
        """

        Write (updated) current or run fidelity to metadata and in the fidelity file in the
        parameter directory

        :param param_metadata: all parameter metadata
        :type param_metadata: dict
        :param fidelities: fidelity values for all parameters
        :type fidelities: list
        :param metadata_file_key: metadata file key (corresponding to current or run fidelity) (default: run fidelity)
        :type metadata_file_key: str
        :param fnamef: file name of fidelity file (corresponding to current or run fidelity) (default: run_fidelity.dat)
        :type fnamef: str

        """
        param_metadata[metadata_file_key] = fidelities

        from maestro import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            for (fid,exp_d,mc_d) in zip(fidelities,param_metadata['param directory'],param_metadata['mc param directory']):
                d=exp_d if os.path.exists(exp_d) else mc_d
                outffidelities = os.path.join(d, fnamef)
                with open(outffidelities, "w") as ff:
                    ff.write("{}".format(fid))


    def get_run_fidelity_from_metadata(self,param_metadata):
        """

        Get run fidelity from parameter metadata

        :param param_metadata: parameter metadata
        :type param_metadata: dict
        :return: run fidelity values
        :rtype: list

        """
        return param_metadata['run fidelity']

    def save_mc_out_as_csv(self,header,term_names,data,out_path):
        """

        Save MC output in CSV format

        :Example:
            If::

                header = ["name", "MC", "DMC"]
                term_names = ["Term1", "Term2"]
                # data: two arrays for MC and DMC respectively and within each
                # array, two values for the two terms
                data = [[19,29],[99,89]]

            Then CSV output is::

                name,MC,DMC
                Term1,19,99
                Term2,29,89

        :param header: CSV header line
        :type header: str
        :param term_names:  objective function term names to use in the
        :type term_names: list
        :param data: parameter and value data
        :type data: list
        :param out_path: path of the output CSV file
        :type out_path: str

        """
        from maestro import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            s = header+"\n"
            for tno,t in enumerate(term_names):
                s+=t+","
                for dno,d in enumerate(data):
                    s+=str(d[tno])
                    s+="\n" if dno == len(data)-1 else ","
            with open(out_path, "w") as ff:
                ff.write(s)

    def inject_nan(self,data,all_param_directory,term_names):
        """

        Inject NaNs into the MC data.

        :Example:
            Example of the ``data`` passed in as argument to this function is shown below.
            In the example below, there are three parameters and two terms of the objective function.
            Terms that end with ``.P`` are the parameters and those ending with ``.V`` are the values
            associated with either the ``MC``, i.e., MC sample values  or the ``DMC``, i.e., MC standard deviation
            values::

                >data

                {
                    "MC":{
                            "Term1.P":[[1,2],[3,4],[6,3]],
                            "Term1.V": [19, 18, 17],
                            "Term2.P":[[1,2],[3,4],[6,3]],
                            "Term2.V": [29, 28, 27]
                        },
                    "DMC":{
                            "Term1.P":[[1,2],[3,4],[6,3]],
                            "Term1.V": [99, 98, 97],
                            "Term2.P":[[1,2],[3,4],[6,3]],
                            "Term2.V": [89, 88, 87]
                        }
                }

        :param data: MC output data formatted into a dictionary
        :type data: dict
        :param all_param_directory: MC outout directory path
        :type all_param_directory: str
        :param term_names: term names e.g., [Term1, Term2]
        :type term_names: list

        """
        chosen_to_set_nan = True
        nan_injection_fraction = self.mc_parmeters['nan_injection_fraction'] \
            if 'nan_injection_fraction' in self.mc_parmeters else 0.0
        apd_arr = os.path.basename(all_param_directory).split('_')
        if nan_injection_fraction >0 and apd_arr[2] != 'Np' and apd_arr[3] != 'k0':
            term_names = np.array(term_names)
            data_names = data.keys()
            total_vals = 0
            for tname in term_names:
                for dno,dname in enumerate(data_names):
                    Y_ = data[dname]["{}.V".format(tname)]
                    total_vals += len(Y_)
            nan_inf_bound = nan_injection_fraction * total_vals
            number_of_nan_inf_per_line = np.ceil(nan_inf_bound/(len(term_names)*len(data_names)))
            for tname in term_names:
                for dno,dname in enumerate(data_names):
                    Y_ = data[dname]["{}.V".format(tname)]
                    nan_inf = [i or j for (i,j) in zip(np.isnan(Y_), np.isinf(Y_))]
                    index = number_of_nan_inf_per_line
                    for ninum,ni in enumerate(nan_inf):
                        if not ni:
                            if chosen_to_set_nan:
                                choice = False
                                Y_[ninum] = np.nan
                                index -= 1
                                nan_inf_bound -= 1
                            else: chosen_to_set_nan = True
                        else:
                            index -= 1
                            nan_inf_bound -= 1
                        if index <= 0 or nan_inf_bound <= 0: break
                    if nan_inf_bound <= 0: break
                if nan_inf_bound <= 0: break

    def convert_csv_data_to_df(self,all_param_directory,mc_out_file_name):
        """

        Read CSV data from all parameter directories and convert them
        into pandas dataframe

        :Example:
            Example of the returned pandas dataframe is given below.
            In the example below, there are three parameters and two terms of the objective function.
            Terms that end with ``.P`` are the parameters and those ending with ``.V`` are the values
            associated with either the ``MC``, i.e., MC sample values  or the ``DMC``, i.e., MC standard deviation
            values. You can add more rows, i.e, more sets of parameter and values  for additional terms in the objective function
            or more columns, i.e., more components of the each term of the objective that
            come from the MC simulator::

                >df
                                MC                      DMC
                Term1.P        [[1,2],[3,4],[6,3]]     [[1,2],[3,4],[6,3]]
                Term1.V        [19, 18, 17]            [99, 98, 97]
                Term2.P        [[1,2],[3,4],[6,3]]     [[1,2],[3,4],[6,3]]
                Term2.V        [29, 28, 27]            [89, 88, 87]

        :param all_param_directory: MC run directory path
        :type all_param_directory: str
        :param mc_out_file_name: MC out CSV file name
        :type mc_out_file_name: str

        """
        import pandas as pd
        from maestro import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        dirlist = self.get_param_directory_array(all_param_directory)
        main_object = {}
        term_names = []
        for dno,d in enumerate(dirlist):
            param = self.get_param_from_directory(d)
            mc_out_path = os.path.join(d,mc_out_file_name)
            df = None
            if rank == 0:
                df = pd.read_csv(mc_out_path)
            df = comm.bcast(df, root=0)
            rownames = list(df.columns.values)
            columnnames = list(df.index)
            for rno in range(1,len(rownames)):
                if dno==0:
                    main_object[rownames[rno]] = {}
                    for cno in range(len(columnnames)):
                        name = df[rownames[0]][columnnames[cno]]
                        term_names.append(name)
                        main_object[rownames[rno]]["{}.P".format(name)] = []
                        main_object[rownames[rno]]["{}.V".format(name)] = []
                for cno in range(len(columnnames)):
                    name = df[rownames[0]][columnnames[cno]]
                    val = df[rownames[rno]][columnnames[cno]]
                    main_object[rownames[rno]]["{}.P".format(name)].append(param)
                    main_object[rownames[rno]]["{}.V".format(name)].append(val)
        self.inject_nan(main_object,all_param_directory,term_names)
        self.check_and_resolve_nan_inf(main_object,all_param_directory,term_names)
        return pd.DataFrame(main_object)





# class A(MCTask):
#     def write_param(self, parameter_array, file):
#         super().write_param(parameter_array, file)



