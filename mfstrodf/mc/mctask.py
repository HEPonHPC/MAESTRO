import pprint

import re,glob,os
import numpy as np
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.16f')
import math
from mfstrodf import DiskUtil
class MCTask(object):
    def __init__(self,mc_working_directory, mc_parameters=None):
        self.mc_run_folder = mc_working_directory
        self.mc_parmeters = mc_parameters

    # todo === add to doc
    # todo === can be called from __main__ or directly on object (should work as a blackbox -- like a task in the workflow)
    def run_mc(self):
        raise Exception("This function must be implemented in the derived class")

    # todo === add to doc
    # todo === return max sigma. If sigma cannot be found return None
    def merge_statistics_and_get_max_sigma(self):
        raise Exception("This function must be implemented in the derived class")

    # todo === add to doc
    # todo === return df and additional_data object (additional_data can be none)
    def convert_mc_output_to_df(self, all_param_directory):
        raise Exception("This function must be implemented in the derived class")

    @staticmethod
    def read_params_file(path):
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
        Get all parameters from MC run directory
        :param pfname: parameter file name with extension (to search for)
        :type pfname: str
        :return: parameter values (in order of how they a listed in fnamep)
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
        return param_metadata['parameters']

    def write_param(self, parameters, parameter_names, at_fidelities, run_fidelities,
                    mc_run_folder, expected_folder_name,
                    fnamep="params.dat", fnamerf="run_fidelity.dat",
                    fnameaf="at_fidelity.dat"):
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

        from mfstrodf import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
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
        new_at_fid = [i+j for (i,j) in zip(param_metadata['at fidelity'], param_metadata['run fidelity'])]
        param_metadata['at fidelity'] = new_at_fid
        return new_at_fid

    def get_updated_current_fidelities(self,param_metadata):
        return self.update_current_fidelities(param_metadata)

    def get_current_fidelities(self,param_metadata):
        return param_metadata['at fidelity']

    def get_parameter_data_from_metadata(self,param_metadata,param_index,include_parameter = False):
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
        return param_metadata['run fidelity']

    def save_mc_out_as_csv(self,header,term_names,data,out_path):
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



