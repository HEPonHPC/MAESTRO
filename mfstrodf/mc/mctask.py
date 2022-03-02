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
        from mfstrodf import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        parameters = None
        if rank == 0:
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
        parameters = comm.bcast(parameters, root=0)
        return parameters

    def get_param_from_directory(self,param_directory,fnamep="params.dat"):
        """
        Get all parameters from MC run directory
        :param pfname: parameter file name with extension (to search for)
        :type pfname: str
        :return: parameter values (in order of how they a listed in fnamep)
        :rtype: list

        """
        re_pfname = re.compile(fnamep) if fnamep else None
        files = glob.glob(os.path.join(param_directory, "*"))
        param = None
        for f in files:
            if re_pfname and re_pfname.search(os.path.basename(f)):
                param = MCTask.read_params_file(f)
        if param is None:
            raise Exception("Something went wrong. Cannot get parameter")
        return param

    def get_param_from_metadata(self, metadata_file):
        from mfstrodf import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        parameters = None
        if rank == 0:
            with open(metadata_file, 'r') as f:
                ds = json.load(f)
            parameters = ds['parameters']
        parameters = comm.bcast(parameters, root=0)
        return parameters

    def write_param(self, parameters, parameter_names, at_fidelities, run_fidelities,
                    file, mc_run_folder, expected_folder_name,
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
            with open(file,'w') as f:
                json.dump(ds, f, indent=4)

    def set_current_iterate_as_next_iterate(self,
                                            current_iterate_meta_data_file,
                                            next_iterate_meta_data_file,
                                            next_iterate_mc_directory=None
                                            ):

        from mfstrodf import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            with open(current_iterate_meta_data_file,'r') as f:
                curr_md_ds = json.load(f)
            curr_mc_dir_name = os.path.join(os.path.dirname(curr_md_ds['param directory'][0]),
                                            '__'+os.path.basename(curr_md_ds['param directory'][0]))
            if os.path.exists(next_iterate_meta_data_file):
                with open(next_iterate_meta_data_file,'r') as f:
                    next_md_ds = json.load(f)
                new_mc_dir_name = next_md_ds['param directory'][0]
                curr_md_ds['param directory'] = next_md_ds['param directory']
            else:
                new_mc_dir_name = os.path.join(next_iterate_mc_directory,os.path.basename(curr_md_ds['param directory'][0]))
                curr_md_ds['param directory'] = [new_mc_dir_name]

            with open(next_iterate_meta_data_file,'w') as f:
                json.dump(curr_md_ds,f,indent=4)

            DiskUtil.copyanything(curr_mc_dir_name,new_mc_dir_name)

    def update_current_fidelities(self,metadata_file):
        from mfstrodf import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        new_at_fid = None
        if rank == 0:
            with open(metadata_file, 'r') as f:
                ds = json.load(f)
            new_at_fid = [i+j for (i,j) in zip(ds['at fidelity'], ds['run fidelity'])]
            ds['at fidelity'] = new_at_fid
            with open(metadata_file,'w') as f:
                json.dump(ds, f, indent=4)
        new_at_fid = comm.bcast(new_at_fid, root=0)
        return new_at_fid

    def get_updated_current_fidelities(self,metadata_file):
        return self.update_current_fidelities(metadata_file)

    def get_current_fidelities(self,metadata_file):
        from mfstrodf import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        at_fid = None
        if rank == 0:
            with open(metadata_file, 'r') as f:
                ds = json.load(f)
            at_fid = ds['at fidelity']
        at_fid = comm.bcast(at_fid, root=0)
        return at_fid

    def get_fidelity_from_directory(self,param_directory,fnamef="run_fidelity.dat"):
        re_fnamerf = re.compile(fnamef) if fnamef else None
        files = glob.glob(os.path.join(param_directory, "*"))
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
        newINDIRSLIST = glob.glob(os.path.join(all_param_directory, "*"))
        return newINDIRSLIST if len(newINDIRSLIST)==1 else \
            sorted(newINDIRSLIST, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

    def write_fidelity_to_metadata_and_directory(self,metadata_file,fidelities,metadata_file_key='run fidelity',
                                                 fnamef="run_fidelity.dat"):
        from mfstrodf import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            with open(metadata_file, 'r') as f:
                ds = json.load(f)
            ds[metadata_file_key] = fidelities
            with open(metadata_file,'w') as f:
                json.dump(ds, f, indent=4)
            for (fid,exp_d,mc_d) in zip(fidelities,ds['param directory'],ds['mc param directory']):
                d=exp_d if os.path.exists(exp_d) else mc_d
                outffidelities = os.path.join(d, fnamef)
                with open(outffidelities, "w") as ff:
                    ff.write("{}".format(fid))


    def get_run_fidelity_from_metadata(self,metadata_file):
        from mfstrodf import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        run_fid = None
        if rank == 0:
            with open(metadata_file, 'r') as f:
                ds = json.load(f)
            run_fid = ds['run fidelity']
        run_fid = comm.bcast(run_fid, root=0)
        return run_fid

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
        dirlist = self.get_param_directory_array(all_param_directory)
        main_object = {}
        for dno,d in enumerate(dirlist):
            param = self.get_param_from_directory(d)
            mc_out_path = os.path.join(d,mc_out_file_name)
            df = pd.read_csv(mc_out_path)
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



