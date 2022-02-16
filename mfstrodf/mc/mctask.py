import pprint

import re,glob,os
import apprentice
import numpy as np
import json
import math
from mfstrodf import DiskUtil
class MCTask(object):
    def __init__(self,mc_working_directory,param_names):
        self.mc_run_folder = mc_working_directory
        #TODO remove param_names and get this info from params.dat
        self.param_names = param_names

    # todo add to doc
    # todo can be called from __main__ or directly on object (should work as a blackbox -- like a task in the workflow)
    def run_mc(self):
        raise Exception("The function objective must be implemented in the derived class")

    #todo add to doc
    #todo return max sigma. If sigma cannot be found return None
    def merge_statistics_and_get_max_sigma(self):
        raise Exception("The function objective must be implemented in the derived class")

    #todo add to doc
    #todo return df and additional_data object
    def convert_mc_output_to_df(self, all_param_directory):
        raise Exception("The function objective must be implemented in the derived class")

    def get_param_from_directory(self,param_directory,fnamep="params.dat"):
        """
        Get all parameters from MC run directory
        :param pfname: parameter file name with extension (to search for)
        :type pfname: str
        :return: parameter values (in order of param_names)
        :rtype: list

        """
        re_pfname = re.compile(fnamep) if fnamep else None
        files = glob.glob(os.path.join(param_directory, "*"))
        param = None
        for f in files:
            if re_pfname and re_pfname.search(os.path.basename(f)):
                param = apprentice.io.read_paramsfile(f)
        if param is None:
            raise Exception("Something went wrong. Cannot get parameter")
        pp = [param[pn] for pn in self.param_names]
        return pp

    def get_param_from_metadata(self, metadata_file):
        with open(metadata_file, 'r') as f:
            ds = json.load(f)
        return ds['parameters']

    def write_param(self, parameters, parameter_names, at_fidelities, run_fidelities,
                    file, mc_run_folder, expected_folder_name,
                    fnamep="params.dat", fnamef="fidelity.dat", **kwargs):
        DiskUtil.remove_directory(mc_run_folder)
        os.makedirs(mc_run_folder,exist_ok=True)
        param_dir = []
        mc_param_dir = []
        for num, (p, fid) in enumerate(zip(parameters, run_fidelities)):
            npad = "{}".format(num).zfill(1+int(np.ceil(np.log10(len(parameters)))))
            outd_mc_run_folder = os.path.join(mc_run_folder, npad)
            outd_expected_folder = os.path.join(expected_folder_name, npad)
            mc_param_dir.append(outd_mc_run_folder)
            param_dir.append(outd_expected_folder)
            os.makedirs(outd_mc_run_folder,exist_ok=True)
            outfparams = os.path.join(outd_mc_run_folder, fnamep)
            with open(outfparams, "w") as pf:
                for k, v in zip(parameter_names, p):
                    pf.write("{name} {val:e}\n".format(name=k, val=v))
            outffidelities = os.path.join(outd_mc_run_folder, fnamef)
            with open(outffidelities, "w") as ff:
                ff.write("{}".format(fid))
        ds = {
            "parameters": parameters,
            "at fidelity": at_fidelities,
            "run fidelity": run_fidelities,
            "param directory": param_dir,
            "mc param directory":mc_param_dir
        }
        with open(file,'w') as f:
            json.dump(ds, f, indent=4)

    def set_current_iterate_as_next_iterate(self, current_iterate_meta_data_file, next_iterate_meta_data_file):

        with open(next_iterate_meta_data_file,'r') as f:
            next_md_ds = json.load(f)
        with open(current_iterate_meta_data_file,'r') as f:
            curr_md_ds = json.load(f)

        curr_mc_dir_name = os.path.join(os.path.dirname(curr_md_ds['param directory'][0]),
                                        '__'+os.path.basename(curr_md_ds['param directory'][0]))
        new_mc_dir_name = next_md_ds['param directory'][0]
        curr_md_ds['param directory'] = next_md_ds['param directory']

        with open(next_iterate_meta_data_file,'w') as f:
            json.dump(curr_md_ds,f,indent=4)

        DiskUtil.copyanything(curr_mc_dir_name,new_mc_dir_name)

    def update_current_fidelities(self,metadata_file):
        with open(metadata_file, 'r') as f:
            ds = json.load(f)
        new_at_fid = [i+j for (i,j) in zip(ds['at fidelity'], ds['run fidelity'])]
        ds['at fidelity'] = new_at_fid
        with open(metadata_file,'w') as f:
            json.dump(ds, f, indent=4)
        return new_at_fid

    def get_updated_current_fidelities(self,metadata_file):
        return self.update_current_fidelities(metadata_file)

    def get_current_fidelities(self,metadata_file):
        with open(metadata_file, 'r') as f:
            ds = json.load(f)
        return ds['at fidelity']

    def get_run_fidelity_from_directory(self,param_directory,fnamef="fidelity.dat"):
        re_fnamef = re.compile(fnamef) if fnamef else None
        files = glob.glob(os.path.join(param_directory, "*"))
        fid = None
        for file in files:
            if re_fnamef and re_fnamef.search(os.path.basename(file)):
                with open(file) as f:
                    fid = int(next(f))
        if fid is None:
            raise Exception("Something went wrong. Cannot get fidelity")
        return fid

    def get_param_directory_array(self,all_param_directory):
        newINDIRSLIST = glob.glob(os.path.join(all_param_directory, "*"))
        return newINDIRSLIST if len(newINDIRSLIST)==1 else \
            sorted(newINDIRSLIST, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

    def write_run_fidelity_to_metadata_and_directory(self,metadata_file,run_fidelities,fnamef="fidelity.dat"):
        with open(metadata_file, 'r') as f:
            ds = json.load(f)
        ds['run fidelity'] = run_fidelities
        with open(metadata_file,'w') as f:
            json.dump(ds, f, indent=4)
        for (fid,exp_d,mc_d) in zip(run_fidelities,ds['param directory'],ds['mc param directory']):
            d=exp_d if os.path.exists(exp_d) else mc_d
            outffidelities = os.path.join(d, fnamef)
            with open(outffidelities, "w") as ff:
                ff.write("{}".format(fid))


    def get_run_fidelity_from_metadata(self,metadata_file):
        with open(metadata_file, 'r') as f:
            ds = json.load(f)
        return ds['run fidelity']

    def save_mc_out_as_csv(self,header,term_names,data,out_path):
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



