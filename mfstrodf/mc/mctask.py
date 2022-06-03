import sys

import pprint

import re,glob,os
import numpy as np
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.16f')
import math
from mfstrodf import DiskUtil
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
        from mfstrodf import MPI_
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
        from mfstrodf import MPI_
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

        from mfstrodf import MPI_
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
        from mfstrodf import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        files = None
        if rank == 0:
            files = glob.glob(os.path.join(param_directory, "*"))
        files = comm.bcast(files, root=0)
        fid = None
        from mfstrodf import MPI_
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
        from mfstrodf import MPI_
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

        from mfstrodf import MPI_
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
        from mfstrodf import MPI_
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
        from mfstrodf import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        dirlist = self.get_param_directory_array(all_param_directory)
        main_object = {}
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
                        main_object[rownames[rno]]["{}.P".format(name)] = []
                        main_object[rownames[rno]]["{}.V".format(name)] = []
                for cno in range(len(columnnames)):
                    name = df[rownames[0]][columnnames[cno]]
                    val = df[rownames[rno]][columnnames[cno]]
                    if not math.isnan(val) and not math.isinf(val):
                        main_object[rownames[rno]]["{}.P".format(name)].append(param)
                        main_object[rownames[rno]]["{}.V".format(name)].append(val)
        return pd.DataFrame(main_object)





# class A(MCTask):
#     def write_param(self, parameter_array, file):
#         super().write_param(parameter_array, file)



