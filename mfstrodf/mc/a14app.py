import sys

from mfstrodf.mc import MCTask,DiskUtil
import numpy as np
import pprint,os
from mfstrodf.mpi4py_ import MPI_
from subprocess import Popen, PIPE
import apprentice

class A14App(MCTask):
    rivett_analysis = {
        "qcd":["ATLAS_2011_S8924791", "ATLAS_2011_S8971293","ATLAS_2011_I919017","ATLAS_2011_S9128077","ATLAS_2012_I1125575","ATLAS_2014_I1298811","ATLAS_2012_I1094564"],
        # "qcd":["ATLAS_2011_S8924791"], #shortened
        "z":["ATLAS_2011_S9131140","ATLAS_2014_I1300647"],
        # "z":["ATLAS_2011_S9131140"], #shortened
        "ttbar":["ATLAS_2012_I1094568","ATLAS_2013_I1243871"]
    }
    @staticmethod
    def __chunk_fidelity(run_at_fidelity,min_fidelity=50):
        import warnings
        warnings.warn("RUN MC for A14 depricated")
        sys.stdout.flush()
        return None
        # comm = MPI_.COMM_WORLD
        # size = comm.Get_size()
        #
        # split_fidelity = np.ceil(run_at_fidelity/size)
        # if split_fidelity >min_fidelity:
        #     run_fidelity_arr = [int(split_fidelity)] * size
        # else:
        #     run_fidelity_arr = [0] * size
        #     fidelity_remaining = run_at_fidelity
        #     for rank in range(size):
        #         if fidelity_remaining < min_fidelity:
        #             run_fidelity_arr[rank] = min_fidelity
        #             break
        #         run_fidelity_arr[rank] = min_fidelity
        #         fidelity_remaining -= min_fidelity
        # return run_fidelity_arr

    def __run_mc_command(self,runcard,fidelity,anaylysis_name,seed,output_loc):
        """
        Run the miniapp MC command

        :param pp: parameter values
        :param fidelity: number of events to use
        :param output_loc: output location
        :type pp: list
        :type fidelity: int
        :type loc: str
        :return: return code obtained after running miniapp
        :rtype: int

        """
        import warnings
        warnings.warn("RUN MC for A14 depricated")
        sys.stdout.flush()
        return
        # runcardstr = "{}".format(runcard)
        # fidstr = "{}".format(fidelity)
        # seedstr = "{}".format(str(seed))
        # outstr = "{}".format(output_loc)
        # argarr = [self.mc_parmeters['mc_location'], "-p",runcardstr, "-n",fidstr, "-s",seedstr, "-o",outstr]
        # for ra in A14App.rivett_analysis[anaylysis_name]:
        #     argarr.append("-a")
        #     argarr.append(ra)
        # p = Popen(argarr,stdin=PIPE, stdout=PIPE, stderr=PIPE)
        # p.communicate(b"input data that is passed to subprocess' stdin")
        # return p.returncode

    def run_mc(self):
        raise Exception("A14 MC cannot be run using a function call")

        # import warnings
        # warnings.warn("RUN MC for A14 depricated")
        # sys.stdout.flush()
        # comm = MPI_.COMM_WORLD
        # rank = comm.Get_rank()
        # dirlist = self.get_param_directory_array(self.mc_run_folder) # from super class
        # for dno,d in enumerate(dirlist):
        #     # param = self.get_param_from_directory(d) # from super class
        #     run_fidelity = self.get_fidelity_from_directory(d) # from super class
        #     rank_run_fidelity = None
        #     if rank==0:
        #         min_f = self.mc_parmeters['min_fidelity'] \
        #             if 'min_fidelity' in self.mc_parmeters else 50
        #         rank_run_fidelity = A14App.__chunk_fidelity(run_fidelity,min_f)
        #     rank_run_fidelity = comm.scatter(rank_run_fidelity,root=0)
        #     if rank_run_fidelity !=0:
        #         for ano, anlysis_name in enumerate(A14App.rivett_analysis.keys()):
        #             runcard = os.path.join(d, "main30_rivet.{}.cmnd".format(anlysis_name))
        #             outfile = os.path.join(d,"out_{}_curr_r{}.yoda".format(anlysis_name,rank))
        #             seed = np.random.randint(1,9999999)
        #             self.__run_mc_command(runcard,rank_run_fidelity,anlysis_name,seed,outfile)
        # comm.barrier()


    def __merge_yoda_files(self,yoda_files,outfile):
        """
        Merge yoda file statistics into a single yoda file

        :param yoda_files: yoda files to merge
        :type yoda_files: list
        :param outfile: output filename
        :type outfile: str

        """
        from subprocess import Popen, PIPE
        if len(yoda_files) == 1:
            DiskUtil.copyanything(yoda_files[0], outfile)
        else:
            argarr = [self.mc_parmeters['rivetmerge_location'],'-o',outfile]
            for file in yoda_files:
                argarr.append(file)
            argarr.append("-e")
            p = Popen(
                argarr,stdin=PIPE, stdout=PIPE, stderr=PIPE)
            p.communicate(b"input data that is passed to subprocess' stdin")

    def __check_and_resolve_nan_inf(self,data, binids,all_param_directory):
        self.did_middle = False
        self.did_left = False
        self.did_right = False
        self.did_model_eval = False
        self.did_V_discard = False
        self.did_DV_discard = False
        def interpolate_nan_inf(data_array):
            resolved = True
            if np.isnan(data_array).any() or np.isinf(data_array).any():
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
            for binid in binids:
                X_V = data['MC']["{}.P".format(binid)]
                V = data['MC']["{}.V".format(binid)]
                X_DV = data['DMC']["{}.P".format(binid)]
                DV = data['DMC']["{}.V".format(binid)]

                import json
                from mfstrodf.model import ModelConstruction
                apd_arr = os.path.basename(all_param_directory).split('_')
                for (X,data_array,mc_prefix) in zip([X_V,X_DV],[V,DV],['MC','DMC']):
                    if np.isnan(data_array).any() or np.isinf(data_array).any():
                        nan_inf = [i or j for (i,j) in zip(np.isnan(data_array), np.isnan(data_array))]
                        #   If only one parameter
                        if apd_arr[2] != 'Np':
                            if apd_arr[3] == 'k0':
                                raise Exception("Nan and Inf found at the start point. Algorithm cannot "
                                                "continue. Please select a new start point in \"tr_center\" and "
                                                "try again")
                            # If k > 0: read model of the corresponding bin and evaluate model at x where y is nan/inf
                            else:
                                log_dir = os.path.normpath(self.mc_run_folder + os.sep + os.pardir)
                                # read algorithm_parameters_dump.json
                                with open(os.path.join(log_dir,'algorithm_parameters_dump.json')) as f:
                                    algo_ds = json.load(f)
                                k = algo_ds['current_iteration']
                                with open(os.path.join(log_dir,'config_dump.json')) as f:
                                    config_ds = json.load(f)
                                function_str_dict = config_ds['model']['function_str']
                                with open(os.path.join(log_dir,"{}_model_k{}.json".format(mc_prefix,k))) as f:
                                    model_ds = json.load(f)
                                for ninum,ni in enumerate(nan_inf):
                                    if ni:
                                        x = X[ninum]
                                        binid = bins_in_obs[ninum]
                                        bin_model_ds = model_ds[binid]
                                        model = ModelConstruction.get_model_object(function_str_dict[mc_prefix],bin_model_ds)
                                        data_array[ninum] = model(x)
                                        self.did_model_eval = True
                #   If more than one parameter:
                if apd_arr[2] == 'Np':
                    if np.isnan(V).any() or np.isinf(V).any():
                        nan_inf = [i or j for (i,j) in zip(np.isnan(V), np.isinf(V))]
                        X_V = np.array(X_V)[np.invert(nan_inf)].tolist()
                        V =  np.array(V)[np.invert(nan_inf)].tolist()
                        X_DV = np.array(X_DV)[np.invert(nan_inf)].tolist()
                        DV = np.array(DV)[np.invert(nan_inf)].tolist()
                        self.did_V_discard = True
                    if np.isnan(DV).any() or np.isinf(DV).any():
                        nan_inf = [i or j for (i,j) in zip(np.isnan(DV), np.isinf(DV))]
                        X_V = np.array(X_V)[np.invert(nan_inf)].tolist()
                        V =  np.array(V)[np.invert(nan_inf)].tolist()
                        X_DV = np.array(X_DV)[np.invert(nan_inf)].tolist()
                        DV = np.array(DV)[np.invert(nan_inf)].tolist()
                        self.did_DV_discard = True
                data['MC']["{}.P".format(binid)] =X_V
                data['MC']["{}.V".format(binid)] =V
                data['DMC']["{}.P".format(binid)] =X_DV
                data['DMC']["{}.V".format(binid)] =DV


        # Try to interpolate
        # If not possible to interpolate:
        #   If only one parameter:
        #       find out the iteration number k of the run
        #       If k == 0: raise exception
        #       If k > 0: read model of the corresponding bin and evaluate model at x where y is nan/inf
        #   If more than one parameter:
        #       delete x and y where y is a nan/inf
        observables = np.unique([b.split("#")[0]  for b in binids])
        binids = np.array(binids)
        resolved = True
        for onum, obs in enumerate(observables):
            # Find bins in the observable
            bins_in_obs = np.sort(binids[np.flatnonzero(np.core.defchararray.find(binids,obs)!=-1)])
            for pnum,param in enumerate(data['MC']["{}.P".format(bins_in_obs[0])]):
                V = [data['MC']["{}.V".format(binid)] for binid in bins_in_obs]
                DV = [data['DMC']["{}.V".format(binid)] for binid in bins_in_obs]
                # Try to interpolate
                (V,V_resolved) = interpolate_nan_inf(V)
                (DV,DV_resolved) = interpolate_nan_inf(DV)
                if V_resolved and DV_resolved:
                    for bno,binid in enumerate(bins_in_obs):
                        data['MC']["{}.V".format(binid)] = V[bno]
                        data['DMC']["{}.V".format(binid)] = DV[bno]
                # If not possible to interpolate
                else:
                    resolved = resolved and False
        if not resolved:
            use_model_to_resolve_nan_inf_or_drop()
        nan_inf_status_code = ""
        nan_inf_status_code += 'L' if self.did_left else '-'
        nan_inf_status_code += 'M' if self.did_middle else '-'
        nan_inf_status_code += 'R' if self.did_right else '-'
        nan_inf_status_code += 'E' if self.did_model_eval else '-'
        nan_inf_status_code += 'D' if self.did_V_discard else '-'
        nan_inf_status_code += 'DD' if self.did_DV_discard else '--'

        print_nan_inf_status_code = self.mc_parmeters['print_nan_inf_status_code'] \
            if 'print_nan_inf_status_code' in self.mc_parmeters else False
        if print_nan_inf_status_code:
            print("NaN/Inf Status Code: {}".format(nan_inf_status_code))

    def merge_statistics_and_get_max_sigma(self):
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        # If pythia8-diy-050522 USES the filename from the -o option passed to it
        # rivet_filenames = [
        #     "out_rivet_qcd.yoda",
        #     "out_rivet_z.yoda",
        #     "out_rivet_ttbar.yoda"
        # ]
        # If pythia8-diy-050522 DOES NOT USE the filename from the -o option passed to it
        rivet_filenames = {
            "qcd":"main30_rivet.qcd.cmnd.yoda",
            "z":"main30_rivet.z.cmnd.yoda",
            "ttbar":"main30_rivet.ttbar.cmnd.yoda"
        }
        dirlist = self.get_param_directory_array(self.mc_run_folder)
        rank_dirs = None
        if rank == 0:
            rank_dirs = MPI_.chunk_it(dirlist)
        rank_dirs = comm.scatter(rank_dirs, root=0)
        rank_max_sigma = 0.
        wtfile = self.mc_parmeters['weights'] if 'weights' in self.mc_parmeters else None
        for dno,d in enumerate(rank_dirs):
            """
            Rivet files from MC: See rivet_filenames array above
            out_{}.yoda (if exists) and corresponding file name from rivet_filenames (if exists) merges into out_{}_curr.yoda
            move out_{}_curr.yoda to out_{}.yoda 
            """
            rank_max_sigma = 0.
            for analysis_name in A14App.rivett_analysis.keys():

                yodafiles = []
                mainfile = os.path.join(d, "out_{}.yoda".format(analysis_name))
                if os.path.exists(mainfile):
                    yodafiles.append(mainfile)
                if os.path.exists(rivet_filenames[analysis_name]):
                    yodafiles.append(rivet_filenames[analysis_name])
                outfile = os.path.join(d, "out_{}_curr.yoda".format(analysis_name))
                self.__merge_yoda_files(yodafiles,outfile)
                for i in range(len(yodafiles)):
                    if i == 0 and mainfile in yodafiles[i]: continue
                    file = yodafiles[i]
                    os.remove(file)
                DiskUtil.moveanything(outfile,mainfile)
                (DATA,BNAMES) = apprentice.io.readSingleYODAFile(d, "params.dat", wtfile)
                sigma = [_E[0] for mcnum, (_X, _Y, _E) in enumerate(DATA)]
                rank_max_sigma = max(rank_max_sigma,max(sigma))
        all_sigma = comm.gather(rank_max_sigma,root=0)
        max_sigma = None
        if rank == 0:
            max_sigma = max(all_sigma)
        max_sigma = comm.bcast(max_sigma,root=0)
        return max_sigma

    def check_df_structure_sanity(self,df):
        rownames = list(df.columns.values)
        columnnames = list(df.index)
        if len(rownames)>1 and ('.P' not in rownames[0] and '.V' not in rownames[1]) and \
                len(columnnames)>1 and ('.P' not in columnnames[0] and '.V' not in columnnames[1]):
            raise Exception('The MC data frame does not have a parameter index that ends in \".P\" '
                            'and value index that ends in \".V\"')
        if len(rownames)>1 and ('.P' in rownames[0] and '.V' in rownames[1]):
            df = df.transpose()
        return df

    def convert_mc_output_to_df(self, all_param_directory):
        import pandas as pd
        main_object = {}
        wtfile = self.mc_parmeters['weights'] if 'weights' in self.mc_parmeters else None
        main_object['MC'] = {}
        main_object['DMC'] = {}
        DATA, binids, pnames, xmin, xmax = apprentice.io.read_input_data_YODA_on_all_ranks(
            [all_param_directory], "params.dat",wtfile,storeAsH5=None)
        for num, (X, Y, E) in enumerate(DATA):
            bin = binids[num]
            main_object["MC"]["{}.P".format(bin)] = X
            main_object["MC"]["{}.V".format(bin)] = Y
            main_object["DMC"]["{}.P".format(bin)] = X
            main_object["DMC"]["{}.V".format(bin)] = E
        self.__check_and_resolve_nan_inf(main_object,binids,all_param_directory)
        df = pd.DataFrame(main_object)
        df = self.check_df_structure_sanity(df)
        additional_data = {"MC":{"xmin":xmin,"xmax":xmax},"DMC":{"xmin":xmin,"xmax":xmax}}
        return (df,additional_data)

    def write_param(self, parameters, parameter_names, at_fidelities, run_fidelities,
                    mc_run_folder, expected_folder_name,
                    fnamep="params.dat", fnamerf="run_fidelity.dat",
                    fnameaf="at_fidelity.dat"):
        ds = super().write_param(parameters,parameter_names,at_fidelities,run_fidelities,mc_run_folder,
                            expected_folder_name,fnamep,fnamerf,fnameaf)
        from mfstrodf import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            dirlist = self.get_param_directory_array(mc_run_folder)
            for d in dirlist:
                fin = open(os.path.join(d,fnamep), "r")
                param_data = fin.read()
                fin.close()
                for rc_path in self.mc_parmeters['run_card_paths']:
                    dst = os.path.join(d,os.path.basename(rc_path))
                    DiskUtil.copyanything(rc_path,dst)
                    fout = open(dst, "a")
                    fout.write("\n")
                    fout.write(param_data)
                    fout.close()

        return ds